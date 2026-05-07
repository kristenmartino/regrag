"""Stage 6: upsert documents + chunks into Postgres (Neon + pgvector).

Idempotent on chunk_id and accession_number. Skips embedding for chunks
whose chunk_content_hash + embedding_model already exist with that hash —
makes re-runs after corpus growth fast.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Iterable

import psycopg
from pgvector.psycopg import register_vector

from .chunk import CHUNKER_VERSION, Chunk
from .embed import EMBEDDING_MODEL, embed_texts
from .fetch import FetchResult
from .manifest import ManifestEntry

log = logging.getLogger(__name__)


def get_conn() -> psycopg.Connection:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL not set in environment")
    conn = psycopg.connect(url, autocommit=False)
    register_vector(conn)
    return conn


def upsert_document(conn: psycopg.Connection, entry: ManifestEntry, fetch: FetchResult) -> None:
    accession = entry.accession_number or entry.slug
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents (
                accession_number, order_number, docket_numbers,
                document_type, issue_date, title, source_url,
                fetched_at, content_hash
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (accession_number) DO UPDATE SET
                order_number   = EXCLUDED.order_number,
                docket_numbers = EXCLUDED.docket_numbers,
                document_type  = EXCLUDED.document_type,
                issue_date     = EXCLUDED.issue_date,
                title          = EXCLUDED.title,
                source_url     = EXCLUDED.source_url,
                fetched_at     = EXCLUDED.fetched_at,
                content_hash   = EXCLUDED.content_hash
            """,
            (
                accession,
                entry.order_number,
                entry.docket_numbers,
                entry.document_type,
                entry.issue_date,
                entry.title,
                entry.pdf_url,
                datetime.utcnow(),
                fetch.content_hash,
            ),
        )


def upsert_chunks(
    conn: psycopg.Connection,
    chunks: list[Chunk],
    *,
    skip_existing: bool = True,
) -> tuple[int, int]:
    """Embed-then-upsert. Returns (embedded_count, skipped_count).

    skip_existing: when True, chunks whose chunk_content_hash already exists in
    the table with the same embedding_model are skipped (no re-embedding).
    """
    if not chunks:
        return 0, 0

    if skip_existing:
        existing_hashes = _existing_hashes(conn, [c.chunk_content_hash for c in chunks])
    else:
        existing_hashes = set()

    to_embed = [c for c in chunks if c.chunk_content_hash not in existing_hashes]
    skipped = len(chunks) - len(to_embed)

    if to_embed:
        log.info("embedding %d new chunks (%d skipped as already-embedded)", len(to_embed), skipped)
        vectors = embed_texts([c.chunk_text for c in to_embed], input_type="document")
        if len(vectors) != len(to_embed):
            raise RuntimeError(f"voyage returned {len(vectors)} vectors for {len(to_embed)} chunks")
        chunk_to_vector = dict(zip([c.chunk_id for c in to_embed], vectors))
    else:
        chunk_to_vector = {}
        log.info("all %d chunks already embedded — skipping voyage call", skipped)

    # Two-phase insert to satisfy parent_chunk_id self-references:
    # body chunks first (parent_chunk_id IS NULL), then footnote chunks.
    body_chunks = [c for c in chunks if c.parent_chunk_id is None]
    footnote_chunks = [c for c in chunks if c.parent_chunk_id is not None]
    for c in body_chunks + footnote_chunks:
        emb = chunk_to_vector.get(c.chunk_id)
        _insert_chunk(conn, c, emb)

    return len(to_embed), skipped


def _insert_chunk(conn: psycopg.Connection, c: Chunk, embedding: list[float] | None) -> None:
    """Upsert one chunk. If embedding is None, the existing row's embedding is preserved."""
    with conn.cursor() as cur:
        if embedding is not None:
            cur.execute(
                """
                INSERT INTO chunks (
                    chunk_id, accession_number, section_heading, paragraph_range,
                    chunk_text, chunk_content_hash, embedding,
                    embedding_model, chunker_version, chunk_index, parent_chunk_id
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    accession_number  = EXCLUDED.accession_number,
                    section_heading   = EXCLUDED.section_heading,
                    paragraph_range   = EXCLUDED.paragraph_range,
                    chunk_text        = EXCLUDED.chunk_text,
                    chunk_content_hash= EXCLUDED.chunk_content_hash,
                    embedding         = EXCLUDED.embedding,
                    embedding_model   = EXCLUDED.embedding_model,
                    chunker_version   = EXCLUDED.chunker_version,
                    chunk_index       = EXCLUDED.chunk_index,
                    parent_chunk_id   = EXCLUDED.parent_chunk_id
                """,
                (
                    c.chunk_id, c.accession_number, c.section_heading, c.paragraph_range,
                    c.chunk_text, c.chunk_content_hash, embedding,
                    EMBEDDING_MODEL, CHUNKER_VERSION, c.chunk_index, c.parent_chunk_id,
                ),
            )
        else:
            # row exists with same content_hash → only refresh non-embedding metadata
            cur.execute(
                """
                INSERT INTO chunks (
                    chunk_id, accession_number, section_heading, paragraph_range,
                    chunk_text, chunk_content_hash,
                    embedding_model, chunker_version, chunk_index, parent_chunk_id
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    section_heading = EXCLUDED.section_heading,
                    paragraph_range = EXCLUDED.paragraph_range,
                    chunker_version = EXCLUDED.chunker_version,
                    chunk_index     = EXCLUDED.chunk_index,
                    parent_chunk_id = EXCLUDED.parent_chunk_id
                """,
                (
                    c.chunk_id, c.accession_number, c.section_heading, c.paragraph_range,
                    c.chunk_text, c.chunk_content_hash,
                    EMBEDDING_MODEL, CHUNKER_VERSION, c.chunk_index, c.parent_chunk_id,
                ),
            )


def _existing_hashes(conn: psycopg.Connection, hashes: Iterable[str]) -> set[str]:
    hashes = list(set(hashes))
    if not hashes:
        return set()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT chunk_content_hash FROM chunks "
            "WHERE chunk_content_hash = ANY(%s) AND embedding_model = %s "
            "AND embedding IS NOT NULL",
            (hashes, EMBEDDING_MODEL),
        )
        return {row[0] for row in cur.fetchall()}
