"""Hybrid retrieval: vector similarity + keyword matching, fused via RRF,
with an identifier-match recall floor.

Implements the spec from docs/implementation-plan.md §2.3:

  hybrid_retrieve(query, k=10):
    1. Extract regulatory identifiers from query
    2. Vector search: top-20 by cosine similarity (pgvector HNSW)
    3. Keyword search: top-10 by ts_rank('simple', chunk_text)
    4. RRF fusion (k_const=60) over both lists
    5. Identifier-match floor: any chunk containing an exact identifier
       from the query is guaranteed to be in the result set
    6. Dedupe by chunk_id, return top-k by fused score

The "recall floor" framing from CS §7 is implemented as the explicit step
5: queries that name an identifier always get the matching chunks, regardless
of vector or keyword scores. Cap floor at FLOOR_MAX chunks to bound result-set
size on broad-identifier queries.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Iterable

import psycopg
import voyageai
from pgvector.psycopg import register_vector

from .identifiers import ExtractedIdentifiers, extract_identifiers

VECTOR_TOPK = 20
KEYWORD_TOPK = 10
RRF_K_CONST = 60
FLOOR_MAX = 5
EMBEDDING_MODEL = "voyage-3.5-lite"
EMBEDDING_DIM = 512

log = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    chunk_id: str
    accession_number: str
    section_heading: str | None
    paragraph_range: str | None
    chunk_text: str
    parent_chunk_id: str | None
    # Scoring details — useful for inspection / debugging
    vector_rank: int | None       # rank in vector top-K (1-indexed), None if not in list
    keyword_rank: int | None      # rank in keyword top-K (1-indexed), None if not in list
    cosine_sim: float | None      # cosine similarity (1 - cosine distance)
    ts_rank: float | None         # Postgres ts_rank
    floor_match: bool             # True if chunk forced in via identifier-match floor
    rrf_score: float              # final RRF score


def hybrid_retrieve(
    query: str,
    *,
    k: int = 10,
    conn: psycopg.Connection | None = None,
    voyage_client: voyageai.Client | None = None,
) -> list[RetrievedChunk]:
    """Run hybrid retrieval against the chunks table. Returns top-k chunks
    with scoring details. If conn or voyage_client are None, they are
    constructed from environment variables (DATABASE_URL, VOYAGE_API_KEY)."""
    own_conn = conn is None
    if conn is None:
        conn = _connect_pg()
    if voyage_client is None:
        voyage_client = _voyage_client()

    try:
        ids = extract_identifiers(query)
        log.debug("extracted identifiers: %s", ids)

        query_emb = _embed_query(voyage_client, query)
        vector_hits = _vector_topk(conn, query_emb, k=VECTOR_TOPK)
        keyword_hits = _keyword_topk(conn, query, ids, k=KEYWORD_TOPK)
        floor_hits = _identifier_floor(conn, ids, max_n=FLOOR_MAX) if not ids.is_empty else []

        fused = _rrf_fuse(vector_hits, keyword_hits, floor_hits)
        return fused[:k]
    finally:
        if own_conn:
            conn.close()


# ---- internal helpers ----


def _connect_pg() -> psycopg.Connection:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL not set")
    conn = psycopg.connect(url)
    register_vector(conn)
    return conn


def _voyage_client() -> voyageai.Client:
    key = os.environ.get("VOYAGE_API_KEY")
    if not key:
        raise RuntimeError("VOYAGE_API_KEY not set")
    return voyageai.Client(api_key=key)


def _embed_query(client: voyageai.Client, query: str) -> list[float]:
    result = client.embed(
        [query], model=EMBEDDING_MODEL, output_dimension=EMBEDDING_DIM, input_type="query"
    )
    return result.embeddings[0]


def _vector_topk(conn: psycopg.Connection, query_emb: list[float], *, k: int) -> list[dict]:
    """Top-k chunks by cosine distance to query embedding."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT chunk_id, accession_number, section_heading, paragraph_range,
                   chunk_text, parent_chunk_id,
                   1 - (embedding <=> %s::vector) AS cosine_sim
            FROM chunks
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_emb, query_emb, k),
        )
        rows = cur.fetchall()
    return [
        {
            "chunk_id": r[0], "accession_number": r[1], "section_heading": r[2],
            "paragraph_range": r[3], "chunk_text": r[4], "parent_chunk_id": r[5],
            "cosine_sim": float(r[6]),
        }
        for r in rows
    ]


_QUERY_WORD_RE = __import__("re").compile(r"\w+")
# 'simple' tokenizer keeps everything lowercase, no stop-word removal — we
# strip only ultra-short tokens (1 char) which are tsquery noise.
_TS_STOP = {"a", "an", "the", "is", "of", "in", "for", "and", "or", "to", "what",
            "does", "do", "did", "how", "why", "when", "where", "who", "which",
            "me", "my", "i", "you", "we", "us", "our", "find"}


def _build_or_tsquery(natural_query: str, identifier_terms: list[str]) -> str:
    """Build an OR tsquery string suitable for to_tsquery('simple', ...).

    Natural-query words are OR'd; each identifier becomes an AND group OR'd
    with the rest. This gives loose recall (any matching word counts) while
    still ranking exact-identifier hits highest via ts_rank.
    """
    natural_tokens = [t.lower() for t in _QUERY_WORD_RE.findall(natural_query)
                      if len(t) > 1 and t.lower() not in _TS_STOP]
    parts: list[str] = list(dict.fromkeys(natural_tokens))  # dedupe, preserve order
    for term in identifier_terms:
        id_tokens = [t.lower() for t in _QUERY_WORD_RE.findall(term) if len(t) > 1]
        if id_tokens:
            parts.append("(" + " & ".join(id_tokens) + ")")
    return " | ".join(parts) if parts else "''"


def _keyword_topk(
    conn: psycopg.Connection, query: str, ids: ExtractedIdentifiers, *, k: int
) -> list[dict]:
    """Top-k chunks by ts_rank using an OR'd tsquery over query words +
    identifier-AND-groups. Identifiers boost rank because chunks containing
    the full identifier match more terms in the OR set."""
    ts_query_str = _build_or_tsquery(query, ids.all_terms)
    if ts_query_str == "''":
        return []

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT chunk_id, accession_number, section_heading, paragraph_range,
                   chunk_text, parent_chunk_id,
                   ts_rank(to_tsvector('simple', chunk_text), to_tsquery('simple', %s)) AS rank
            FROM chunks
            WHERE to_tsvector('simple', chunk_text) @@ to_tsquery('simple', %s)
            ORDER BY rank DESC
            LIMIT %s
            """,
            (ts_query_str, ts_query_str, k),
        )
        rows = cur.fetchall()
    return [
        {
            "chunk_id": r[0], "accession_number": r[1], "section_heading": r[2],
            "paragraph_range": r[3], "chunk_text": r[4], "parent_chunk_id": r[5],
            "ts_rank": float(r[6]),
        }
        for r in rows
    ]


def _identifier_floor(
    conn: psycopg.Connection, ids: ExtractedIdentifiers, *, max_n: int
) -> list[dict]:
    """Chunks containing any exact identifier from the query. Capped at max_n
    to bound result-set size on broad queries (e.g. a docket with hundreds of hits)."""
    if ids.is_empty:
        return []

    # Build OR of LIKE patterns. Use ILIKE for case insensitivity on docket numbers.
    # Each identifier becomes one LIKE clause.
    where_clauses = []
    params: list = []
    for term in ids.all_terms:
        where_clauses.append("chunk_text ILIKE %s")
        params.append(f"%{term}%")

    sql = f"""
        SELECT chunk_id, accession_number, section_heading, paragraph_range,
               chunk_text, parent_chunk_id
        FROM chunks
        WHERE ({" OR ".join(where_clauses)})
        ORDER BY chunk_index
        LIMIT %s
    """
    params.append(max_n)

    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    return [
        {
            "chunk_id": r[0], "accession_number": r[1], "section_heading": r[2],
            "paragraph_range": r[3], "chunk_text": r[4], "parent_chunk_id": r[5],
        }
        for r in rows
    ]


def _rrf_fuse(
    vector_hits: list[dict],
    keyword_hits: list[dict],
    floor_hits: list[dict],
) -> list[RetrievedChunk]:
    """Reciprocal Rank Fusion over the three lists, deduped by chunk_id.

    RRF score = sum over each list it appears in of 1 / (k_const + rank).
    Floor hits get a generous synthetic rank (FLOOR_MAX) so they always
    score above 0 but don't dominate.
    """
    # Build maps from chunk_id → (rank, payload) per list
    vmap = {h["chunk_id"]: (i + 1, h) for i, h in enumerate(vector_hits)}
    kmap = {h["chunk_id"]: (i + 1, h) for i, h in enumerate(keyword_hits)}
    fmap = {h["chunk_id"]: (FLOOR_MAX, h) for h in floor_hits}

    all_ids = set(vmap) | set(kmap) | set(fmap)
    fused: list[RetrievedChunk] = []
    for cid in all_ids:
        vrank, vrow = (vmap[cid][0], vmap[cid][1]) if cid in vmap else (None, None)
        krank, krow = (kmap[cid][0], kmap[cid][1]) if cid in kmap else (None, None)
        in_floor = cid in fmap
        # RRF score
        score = 0.0
        if vrank is not None:
            score += 1.0 / (RRF_K_CONST + vrank)
        if krank is not None:
            score += 1.0 / (RRF_K_CONST + krank)
        if in_floor and (vrank is None and krank is None):
            score += 1.0 / (RRF_K_CONST + FLOOR_MAX)

        # Use whichever payload has the most fields populated (vector wins
        # since it carries cosine_sim; falls back to keyword or floor)
        payload = vrow or krow or fmap[cid][1]
        fused.append(RetrievedChunk(
            chunk_id=payload["chunk_id"],
            accession_number=payload["accession_number"],
            section_heading=payload.get("section_heading"),
            paragraph_range=payload.get("paragraph_range"),
            chunk_text=payload["chunk_text"],
            parent_chunk_id=payload.get("parent_chunk_id"),
            vector_rank=vrank,
            keyword_rank=krank,
            cosine_sim=vrow.get("cosine_sim") if vrow else None,
            ts_rank=krow.get("ts_rank") if krow else None,
            floor_match=in_floor,
            rrf_score=score,
        ))
    fused.sort(key=lambda c: c.rrf_score, reverse=True)
    return fused
