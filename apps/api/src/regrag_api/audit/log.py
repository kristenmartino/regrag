"""Append-only audit log writes — one row per `/chat` invocation.

Per docs/regrag-case-study.md §5 and infra/neon-schema.sql:
  query_log captures, for every interaction: timestamp, user_id, raw_query,
  classification decision, decomposed sub_queries (if any), retrieved chunks
  with provenance, prompt sent to generation, model identifier, raw response,
  verified response, latency and token counts per stage.

Audit writes are best-effort: if Postgres is unreachable or the write fails,
we log a warning and let the chat response return successfully. The user's
answer is more important than the audit row, and the failure mode is
recoverable (the row is missing, not corrupt).
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any

import psycopg

from ..orchestration.state import GraphState

log = logging.getLogger(__name__)

# Cap stored prompt at this many bytes — synthesis prompts include all retrieved
# chunks and can be 10-30 KB. The model_id and answer are the audit-load-bearing
# fields; the prompt is here for replay debugging, not legal evidence.
MAX_PROMPT_BYTES = 8 * 1024


def write_query_log(state: GraphState, *, conn: psycopg.Connection | None = None) -> str | None:
    """Insert one row into query_log from a completed graph state. Returns the
    generated query_id on success, None on failure (logged but not raised)."""
    own_conn = conn is None
    try:
        if conn is None:
            url = os.environ.get("DATABASE_URL")
            if not url:
                log.debug("audit skipped: DATABASE_URL not set")
                return None
            conn = psycopg.connect(url)
        query_id = uuid.uuid4()
        row = _build_row(state, query_id)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO query_log (
                    query_id, user_id, raw_query, classification, sub_queries,
                    retrieved_chunks, prompt_sent, model_id,
                    raw_response, verified_response,
                    citations_stripped, refusal_emitted, refusal_reason,
                    latency_ms_total, latency_ms_by_stage, token_counts
                ) VALUES (
                    %(query_id)s, %(user_id)s, %(raw_query)s, %(classification)s, %(sub_queries)s,
                    %(retrieved_chunks)s, %(prompt_sent)s, %(model_id)s,
                    %(raw_response)s, %(verified_response)s,
                    %(citations_stripped)s, %(refusal_emitted)s, %(refusal_reason)s,
                    %(latency_ms_total)s, %(latency_ms_by_stage)s, %(token_counts)s
                )
                """,
                row,
            )
        conn.commit()
        return str(query_id)
    except Exception as e:
        log.warning("audit log write failed (%s) — chat response unaffected", e)
        return None
    finally:
        if own_conn and conn is not None:
            conn.close()


def _build_row(state: GraphState, query_id: uuid.UUID) -> dict[str, Any]:
    timings = state.get("timings") or {}
    model_ids = state.get("model_ids_used") or {}
    chunks = state.get("retrieved_chunks") or []

    prompt = state.get("synthesize_prompt") or ""
    if len(prompt) > MAX_PROMPT_BYTES:
        prompt = prompt[:MAX_PROMPT_BYTES] + f"\n\n[...truncated; original was {len(prompt)} bytes]"

    # If the synthesizer never ran (pre-gen refusal), use a placeholder. The
    # schema declares prompt_sent and model_id NOT NULL.
    if not prompt:
        prompt = "(synthesizer not invoked — pre-generation refusal)"
    primary_model = model_ids.get("synthesize") or model_ids.get("classify") or "(none)"

    return {
        "query_id": query_id,
        "user_id": state.get("user_id"),
        "raw_query": state.get("query") or "",
        "classification": state.get("classification"),
        "sub_queries": json.dumps(state.get("sub_queries") or []),
        "retrieved_chunks": json.dumps(_compact_chunks(chunks)),
        "prompt_sent": prompt,
        "model_id": primary_model,
        "raw_response": state.get("draft_answer") or "",
        "verified_response": state.get("final_answer") or "",
        "citations_stripped": int(state.get("citations_stripped") or 0),
        "refusal_emitted": bool(state.get("refusal_emitted")),
        "refusal_reason": state.get("refusal_reason"),
        "latency_ms_total": int(sum(timings.values())) if timings else 0,
        "latency_ms_by_stage": json.dumps(timings),
        "token_counts": json.dumps(state.get("token_counts") or {}),
    }


def _compact_chunks(chunks: list[dict]) -> list[dict]:
    """Snapshot only the audit-relevant fields per chunk. Drops the per-call
    scoring details (vector_rank, ts_rank, etc.) since those don't replay."""
    return [
        {
            "chunk_id": c.get("chunk_id"),
            "accession_number": c.get("accession_number"),
            "section_heading": c.get("section_heading"),
            "paragraph_range": c.get("paragraph_range"),
            "chunk_text": c.get("chunk_text"),
            "parent_chunk_id": c.get("parent_chunk_id"),
        }
        for c in chunks
    ]
