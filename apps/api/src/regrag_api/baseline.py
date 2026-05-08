"""Naive RAG baseline for measurement comparison.

Bypasses everything that distinguishes RegRAG from a plain prompt-with-chunks
pipeline: no classifier, no decomposition, no hybrid retrieval (pure vector),
no identifier floor, no citation verification, no substantive-support judge.

The point is not to ship this. The point is to have a fair "what does the
same model + corpus look like without the agentic layer and verification" so
the eval harness can quantify what the rest of the system buys us. Returns
the same GraphState shape the eval runner already understands.

Wired into eval via `regrag-eval run --baseline`. Not exposed by the FastAPI
server.
"""

from __future__ import annotations

import logging
import os
import time

import psycopg
import voyageai
from pgvector.psycopg import register_vector

from .orchestration._anthropic import SYNTHESIS_MODEL, get_client
from .orchestration.state import GraphState, initial_state
from .retrieval.hybrid import EMBEDDING_DIM, EMBEDDING_MODEL

log = logging.getLogger(__name__)

BASELINE_K = 10  # match the order of magnitude of the agentic path's per-query retrieval

# Two baseline variants, selectable via env var BASELINE_PROMPT_VARIANT:
#   "thin"   — minimal instruction; tests "what does vanilla RAG produce out of the box"
#   "matched" — same citation discipline as production prompt, just no agentic / no verification;
#              isolates JUST the architectural contribution

THIN_PROMPT = """\
You are answering questions about FERC regulatory orders. Use the chunks below to answer.

Cite chunks using their chunk_id in [[chunk_id]] format after sentences supported by a chunk. Copy chunk_ids verbatim from the chunk headers.

CHUNKS:
{chunks_block}
"""

MATCHED_PROMPT = """\
You are answering questions about FERC regulatory orders. Use ONLY the chunks below; do not draw on outside knowledge or training data.

CITATION DISCIPLINE — read this carefully:

Before you write any sentence with a citation, find the specific phrase in the cited chunk that asserts what you're claiming. If the chunk discusses the topic but does not make the specific assertion you want to make — DO NOT MAKE THE CLAIM. Drop it. A shorter answer with grounded claims is much better than a longer answer with paraphrased-from-context claims.

Cite every substantive claim using the chunk_id in double square brackets, exactly as shown in the chunk's header. Place the citation at the end of the sentence the chunk supports.

CITATION FORMAT — copy the chunk_id verbatim, including all colons and digits:
  Correct:   [[20200917-3084:c0111]]
  WRONG:     [[Order 2222]]
  WRONG:     [[20200917-3084]]              ← must include the cNNNN suffix

If the chunks do NOT address the question, say so briefly rather than inventing details.

CHUNKS:
{chunks_block}
"""

BASELINE_PROMPT = (
    MATCHED_PROMPT
    if os.environ.get("BASELINE_PROMPT_VARIANT", "matched") == "matched"
    else THIN_PROMPT
)


def run_baseline(query: str, *, user_id: str | None = None, k: int = BASELINE_K) -> GraphState:
    """Single-pass naive RAG: pure vector retrieval + one Sonnet call."""
    state = initial_state(query, user_id=user_id)
    state["classification"] = "baseline"  # marker so the eval can distinguish runs

    # ─── Pure vector retrieval (no RRF, no keyword, no identifier floor) ───
    t_retrieve = time.perf_counter()
    chunks = _pure_vector_topk(query, k)
    state["retrieved_chunks"] = chunks
    state["top_cosine_sim"] = chunks[0]["cosine_sim"] if chunks else 0.0
    state["timings"] = dict(state.get("timings") or {})
    state["timings"]["retrieve_baseline"] = int((time.perf_counter() - t_retrieve) * 1000)

    if not chunks:
        state["refusal_emitted"] = True
        state["refusal_reason"] = "no_relevant_chunks"
        state["final_answer"] = "No relevant chunks retrieved."
        return state

    # ─── Single Sonnet call with chunks + bare prompt ───
    t_synth = time.perf_counter()
    chunks_block = "\n\n".join(
        f"[chunk_id={c['chunk_id']} | accession={c['accession_number']} | section={c.get('section_heading') or '?'}]\n"
        f"{(c.get('chunk_text') or '').strip()}"
        for c in chunks
    )
    system = BASELINE_PROMPT.format(chunks_block=chunks_block)

    client = get_client()
    response = client.messages.create(
        model=SYNTHESIS_MODEL,
        max_tokens=2048,
        system=system,
        messages=[{"role": "user", "content": query}],
    )
    answer = response.content[0].text.strip()
    state["draft_answer"] = answer
    state["final_answer"] = answer  # no verification step → final == draft
    state["timings"]["synthesize_baseline"] = int((time.perf_counter() - t_synth) * 1000)
    state["model_ids_used"] = {"synthesize_baseline": SYNTHESIS_MODEL}
    state["token_counts"] = {
        "synthesize_baseline": {
            "in": response.usage.input_tokens,
            "out": response.usage.output_tokens,
        }
    }
    return state


def _pure_vector_topk(query: str, k: int) -> list[dict]:
    db_url = os.environ.get("DATABASE_URL")
    voyage_key = os.environ.get("VOYAGE_API_KEY")
    if not db_url or not voyage_key:
        raise RuntimeError("baseline requires DATABASE_URL and VOYAGE_API_KEY")

    voyage_client = voyageai.Client(api_key=voyage_key)
    emb_result = voyage_client.embed(
        [query], model=EMBEDDING_MODEL, output_dimension=EMBEDDING_DIM, input_type="query"
    )
    query_emb = emb_result.embeddings[0]

    with psycopg.connect(db_url) as conn:
        register_vector(conn)
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
            "chunk_id": r[0],
            "accession_number": r[1],
            "section_heading": r[2],
            "paragraph_range": r[3],
            "chunk_text": r[4],
            "parent_chunk_id": r[5],
            "cosine_sim": float(r[6]),
            "vector_rank": i + 1,
            "keyword_rank": None,
            "ts_rank": None,
            "floor_match": False,
            "rrf_score": None,
        }
        for i, r in enumerate(rows)
    ]
