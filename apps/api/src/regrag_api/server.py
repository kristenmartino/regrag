"""FastAPI surface for RegRAG.

Day 15 scope: POST /chat returns the full response as JSON. Streaming +
intermediate stage events come in Day 16. /audit endpoints come in Day 17.

Run locally:
  cd apps/api
  .venv/bin/uvicorn regrag_api.server:app --reload --port 8000
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import psycopg
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .orchestration.graph import run as run_graph
from .orchestration.graph import run_streaming

# Local-dev convenience: load .env from the repo root if present. In production
# (Railway, Vercel, etc.) env vars come from the platform; this no-ops there.
try:
    _REPO_ROOT = Path(__file__).resolve().parents[4]
    _ENV_PATH = _REPO_ROOT / ".env"
    if _ENV_PATH.exists():
        load_dotenv(_ENV_PATH, override=True)
except (IndexError, OSError):
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
log = logging.getLogger(__name__)


app = FastAPI(
    title="RegRAG API",
    description="Hybrid retrieval + agentic orchestration over FERC orders",
    version="0.1.0",
)

# CORS configuration:
#   - Local dev: any localhost:<port> (Next.js may pick 3000-3010)
#   - Production: comma-separated list in ALLOWED_ORIGINS env var, e.g.
#     ALLOWED_ORIGINS="https://regrag.kristenmartino.ai,https://regrag.vercel.app"
_ALLOWED = os.environ.get("ALLOWED_ORIGINS", "").strip()
_explicit_origins = [o.strip() for o in _ALLOWED.split(",") if o.strip()] if _ALLOWED else []

app.add_middleware(
    CORSMiddleware,
    allow_origins=_explicit_origins,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
log.info("CORS allow_origins=%s + localhost regex", _explicit_origins or "(none)")


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    user_id: str | None = None


class ChunkSummary(BaseModel):
    chunk_id: str
    accession_number: str
    section_heading: str | None = None
    paragraph_range: str | None = None
    chunk_text_preview: str       # first 300 chars
    rrf_score: float | None = None
    cosine_sim: float | None = None


class ChatResponse(BaseModel):
    classification: str | None
    classification_confidence: float | None
    sub_queries: list[str] | None
    retrieved_chunks: list[ChunkSummary]
    final_answer: str
    refusal_emitted: bool
    refusal_reason: str | None
    citations_stripped: int
    regeneration_count: int
    timings_ms: dict[str, int]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    log.info("/chat query=%r user_id=%r", req.query, req.user_id)
    try:
        state = run_graph(req.query, user_id=req.user_id)
    except Exception as e:
        log.exception("chat invocation failed")
        raise HTTPException(status_code=500, detail=f"chat failed: {type(e).__name__}: {e}") from e

    chunks_raw = state.get("retrieved_chunks") or []
    chunks = [
        ChunkSummary(
            chunk_id=c.get("chunk_id", ""),
            accession_number=c.get("accession_number", ""),
            section_heading=c.get("section_heading"),
            paragraph_range=c.get("paragraph_range"),
            chunk_text_preview=(c.get("chunk_text") or "")[:300],
            rrf_score=c.get("rrf_score"),
            cosine_sim=c.get("cosine_sim"),
        )
        for c in chunks_raw
    ]

    return ChatResponse(
        classification=state.get("classification"),
        classification_confidence=state.get("classification_confidence"),
        sub_queries=state.get("sub_queries"),
        retrieved_chunks=chunks,
        final_answer=state.get("final_answer") or "",
        refusal_emitted=bool(state.get("refusal_emitted")),
        refusal_reason=state.get("refusal_reason"),
        citations_stripped=int(state.get("citations_stripped") or 0),
        regeneration_count=int(state.get("regeneration_count") or 0),
        timings_ms=state.get("timings") or {},
    )


# ─── /audit endpoints ──────────────────────────────────────────────


class AuditRowSummary(BaseModel):
    query_id: str
    timestamp: datetime
    user_id: str | None
    raw_query: str
    classification: str | None
    refusal_emitted: bool
    citations_stripped: int
    latency_ms_total: int
    n_chunks: int


class AuditRowDetail(BaseModel):
    query_id: str
    timestamp: datetime
    user_id: str | None
    raw_query: str
    classification: str | None
    sub_queries: list[str] | None
    retrieved_chunks: list[dict[str, Any]]
    prompt_sent: str
    model_id: str
    raw_response: str
    verified_response: str
    citations_stripped: int
    refusal_emitted: bool
    refusal_reason: str | None
    latency_ms_total: int
    latency_ms_by_stage: dict[str, int]
    token_counts: dict[str, dict[str, int]]


def _get_audit_conn() -> psycopg.Connection:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise HTTPException(status_code=503, detail="DATABASE_URL not configured")
    return psycopg.connect(url)


@app.get("/audit", response_model=list[AuditRowSummary])
def audit_list(
    limit: int = Query(default=50, ge=1, le=200),
    user_id: str | None = Query(default=None, description="Filter by user_id (e.g. 'eval-runner')"),
):
    """Recent query_log rows, newest first. Compact summary per row."""
    where_clauses = []
    params: list[Any] = []
    if user_id is not None:
        where_clauses.append("user_id = %s")
        params.append(user_id)
    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    params.append(limit)

    with _get_audit_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT query_id, timestamp, user_id, raw_query, classification,
                       refusal_emitted, citations_stripped, latency_ms_total,
                       jsonb_array_length(retrieved_chunks) AS n_chunks
                FROM query_log
                {where_sql}
                ORDER BY timestamp DESC
                LIMIT %s
                """,
                params,
            )
            rows = cur.fetchall()
    return [
        AuditRowSummary(
            query_id=str(r[0]),
            timestamp=r[1],
            user_id=r[2],
            raw_query=r[3][:500],
            classification=r[4],
            refusal_emitted=r[5],
            citations_stripped=r[6],
            latency_ms_total=r[7],
            n_chunks=r[8] or 0,
        )
        for r in rows
    ]


@app.get("/audit/{query_id}", response_model=AuditRowDetail)
def audit_detail(query_id: str):
    """Full record for one query_log row."""
    with _get_audit_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT query_id, timestamp, user_id, raw_query, classification,
                       sub_queries, retrieved_chunks, prompt_sent, model_id,
                       raw_response, verified_response,
                       citations_stripped, refusal_emitted, refusal_reason,
                       latency_ms_total, latency_ms_by_stage, token_counts
                FROM query_log
                WHERE query_id = %s
                """,
                (query_id,),
            )
            row = cur.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"query_id {query_id} not found")
    return AuditRowDetail(
        query_id=str(row[0]),
        timestamp=row[1],
        user_id=row[2],
        raw_query=row[3],
        classification=row[4],
        sub_queries=row[5] or None,  # JSONB → list/None
        retrieved_chunks=row[6] or [],
        prompt_sent=row[7],
        model_id=row[8],
        raw_response=row[9],
        verified_response=row[10],
        citations_stripped=row[11],
        refusal_emitted=row[12],
        refusal_reason=row[13],
        latency_ms_total=row[14],
        latency_ms_by_stage=row[15] or {},
        token_counts=row[16] or {},
    )


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """SSE stream of stage events as the LangGraph executes.

    Each event is one `data: <json>\\n\\n` block:
      - {type: "started", query: str}
      - {type: "stage_complete", stage: str, delta_summary: dict, elapsed_ms: int}
      - {type: "done", state: dict}    # final state for client rendering
      - {type: "error", message: str}  # on exception
    """
    log.info("/chat/stream query=%r user_id=%r", req.query, req.user_id)

    def event_iter():
        try:
            for event in run_streaming(req.query, user_id=req.user_id):
                yield f"data: {json.dumps(event, default=str)}\n\n"
        except Exception as e:
            log.exception("streaming chat invocation failed")
            yield f"data: {json.dumps({'type': 'error', 'message': f'{type(e).__name__}: {e}'})}\n\n"

    return StreamingResponse(
        event_iter(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",  # disable nginx buffering if proxied
        },
    )
