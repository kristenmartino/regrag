"""Hybrid retrieval: vector similarity + keyword matching, fused via RRF,
with an identifier-match recall floor + document-anchored retrieval.

Pipeline (see docs/implementation-plan.md §2.3):

  hybrid_retrieve(query, k=10):
    1. Extract regulatory identifiers from query
    2. Vector search: top-20 by cosine similarity (pgvector HNSW)
    3. Keyword search: top-10 by ts_rank('simple', chunk_text)
    4. Identifier-match floor: any chunk containing an exact identifier
       from the query is guaranteed to be in the result set
    5. Document-anchored retrieval: when the query names a specific order
       number, also retrieve top-K chunks restricted to that order's
       accession_number(s). Fixes the failure mode where a chunk that
       *references* Order N (e.g. Order N-A's discussion of N) outranks
       chunks *from* Order N itself. (Review finding #9, 2026-05-22.)
    6. RRF fusion (k_const=60) over all four lists
    7. Dedupe by chunk_id, return top-k by fused score
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import psycopg
import voyageai
from pgvector.psycopg import register_vector

from .identifiers import ExtractedIdentifiers, extract_identifiers

VECTOR_TOPK = 20
KEYWORD_TOPK = 10
ANCHORED_TOPK = 8       # per-order anchored retrieval
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
    anchored_rank: int | None     # rank in document-anchored top-K (when query named an order)
    cosine_sim: float | None      # cosine similarity (1 - cosine distance)
    ts_rank: float | None         # Postgres ts_rank
    floor_match: bool             # True if chunk forced in via identifier-match floor
    anchored_match: bool          # True if chunk's accession matches a named order in the query
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

        # Document-anchored retrieval: when query names an order, restrict
        # a separate top-K search to chunks from that order's accession_number(s).
        # Guarantees chunks FROM the order are in the fusion pool even if
        # chunks ABOUT the order from other docs would otherwise outrank them.
        anchored_accessions = _accessions_for_named_orders(conn, ids)
        anchored_hits = (
            _vector_topk_within_accessions(conn, query_emb, anchored_accessions, k=ANCHORED_TOPK)
            if anchored_accessions
            else []
        )

        fused = _rrf_fuse(vector_hits, keyword_hits, floor_hits, anchored_hits, anchored_accessions)
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


# ──────────────────────────────────────────────────────────────────────
# Document-anchored retrieval (added 2026-05-22 per review finding #9)
# ──────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _order_to_accessions_cache_key() -> int:
    """Cache invalidator. Currently no invalidation; restart the process to refresh."""
    return 1


@lru_cache(maxsize=1)
def _load_order_to_accessions() -> dict[str, list[str]]:
    """Query Neon for the order_number → [accession_numbers] mapping. Cached
    for the process lifetime; restart the service after corpus changes."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        return {}
    out: dict[str, list[str]] = {}
    try:
        with psycopg.connect(url, connect_timeout=10) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT order_number, accession_number FROM documents WHERE order_number IS NOT NULL"
                )
                for order_number, accession_number in cur.fetchall():
                    out.setdefault(order_number, []).append(accession_number)
    except Exception as e:
        log.warning("order→accession mapping load failed: %s", e)
        return {}
    log.info("loaded order→accession mapping: %d orders, %d accessions", len(out), sum(len(v) for v in out.values()))
    return out


def _accessions_for_named_orders(conn: psycopg.Connection, ids: ExtractedIdentifiers) -> list[str]:
    """For each Order N named in the query, return the canonical accession_number(s)
    for that order. Rehearings get folded in too: a query for Order 841 also returns
    Order 841-A's accession if it's in the corpus, because they're the same rulemaking.
    """
    if not ids.orders:
        return []
    mapping = _load_order_to_accessions()
    out: list[str] = []
    seen: set[str] = set()
    for order in ids.orders:
        # Match the order exactly, plus any rehearing variant (Order 841, Order 841-A, 841-B...)
        for k, v in mapping.items():
            if k == order or k.startswith(order + "-"):
                for acc in v:
                    if acc not in seen:
                        seen.add(acc)
                        out.append(acc)
    return out


def _vector_topk_within_accessions(
    conn: psycopg.Connection,
    query_emb: list[float],
    accessions: list[str],
    *,
    k: int,
) -> list[dict]:
    """Vector search restricted to specific accession_numbers. Guarantees
    chunks FROM the named order(s) are in the candidate pool."""
    if not accessions:
        return []
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT chunk_id, accession_number, section_heading, paragraph_range,
                   chunk_text, parent_chunk_id,
                   1 - (embedding <=> %s::vector) AS cosine_sim
            FROM chunks
            WHERE embedding IS NOT NULL
              AND accession_number = ANY(%s)
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_emb, accessions, query_emb, k),
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


def _rrf_fuse(
    vector_hits: list[dict],
    keyword_hits: list[dict],
    floor_hits: list[dict],
    anchored_hits: list[dict],
    anchored_accessions: list[str],
) -> list[RetrievedChunk]:
    """Reciprocal Rank Fusion over four lists, deduped by chunk_id.

    RRF score = sum over each list it appears in of 1 / (k_const + rank).
    Floor hits get a generous synthetic rank (FLOOR_MAX) so they always
    score above 0 but don't dominate. Anchored hits get RRF scoring AND
    a flag set on the resulting chunk for inspection.
    """
    # Build maps from chunk_id → (rank, payload) per list
    vmap = {h["chunk_id"]: (i + 1, h) for i, h in enumerate(vector_hits)}
    kmap = {h["chunk_id"]: (i + 1, h) for i, h in enumerate(keyword_hits)}
    fmap = {h["chunk_id"]: (FLOOR_MAX, h) for h in floor_hits}
    amap = {h["chunk_id"]: (i + 1, h) for i, h in enumerate(anchored_hits)}
    anchored_acc_set = set(anchored_accessions)

    all_ids = set(vmap) | set(kmap) | set(fmap) | set(amap)
    fused: list[RetrievedChunk] = []
    for cid in all_ids:
        vrank, vrow = (vmap[cid][0], vmap[cid][1]) if cid in vmap else (None, None)
        krank, krow = (kmap[cid][0], kmap[cid][1]) if cid in kmap else (None, None)
        arank, arow = (amap[cid][0], amap[cid][1]) if cid in amap else (None, None)
        in_floor = cid in fmap

        # RRF score
        score = 0.0
        if vrank is not None:
            score += 1.0 / (RRF_K_CONST + vrank)
        if krank is not None:
            score += 1.0 / (RRF_K_CONST + krank)
        if arank is not None:
            score += 1.0 / (RRF_K_CONST + arank)
        if in_floor and (vrank is None and krank is None and arank is None):
            score += 1.0 / (RRF_K_CONST + FLOOR_MAX)

        # Use whichever payload has the most fields populated. Anchored and
        # vector both carry cosine_sim; either is fine. Keyword carries
        # ts_rank. Floor carries the minimum.
        payload = vrow or arow or krow or fmap[cid][1]
        anchored_match = payload.get("accession_number") in anchored_acc_set
        fused.append(RetrievedChunk(
            chunk_id=payload["chunk_id"],
            accession_number=payload["accession_number"],
            section_heading=payload.get("section_heading"),
            paragraph_range=payload.get("paragraph_range"),
            chunk_text=payload["chunk_text"],
            parent_chunk_id=payload.get("parent_chunk_id"),
            vector_rank=vrank,
            keyword_rank=krank,
            anchored_rank=arank,
            cosine_sim=(vrow or arow or {}).get("cosine_sim"),
            ts_rank=krow.get("ts_rank") if krow else None,
            floor_match=in_floor,
            anchored_match=anchored_match,
            rrf_score=score,
        ))
    fused.sort(key=lambda c: c.rrf_score, reverse=True)
    return fused
