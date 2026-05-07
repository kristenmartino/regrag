"""Retrieve nodes — single-doc and parallel-over-sub-queries paths.

Pre-generation refusal trigger (per docs/implementation-plan.md §2.8): if the
top vector hit is below COSINE_REFUSAL_THRESHOLD, no chunks are sufficient and
the system refuses without invoking generation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict
from typing import Iterable

from ...retrieval.hybrid import RetrievedChunk, hybrid_retrieve
from ..state import GraphState

log = logging.getLogger(__name__)

PER_QUERY_K = 8                # chunks per single retrieval call
# TODO: recalibrate COSINE_REFUSAL_THRESHOLD against the eval set distribution
# once it exists (see docs/implementation-plan.md §2.8). Empirical observation
# from Days 6-7 smoke tests: identifier-heavy in-scope queries naturally have
# top-cosine in 0.43-0.54 range (lower than purely semantic queries which hit
# 0.61-0.68). The 0.55 the plan originally specified blocks real answers; 0.35
# catches obviously-irrelevant retrievals while letting the LLM's own refusal
# flag handle the nuanced cases.
COSINE_REFUSAL_THRESHOLD = 0.35


def retrieve_single(state: GraphState) -> dict:
    return _retrieve_for_queries([state["query"]], state, stage_name="retrieve_single")


def retrieve_parallel(state: GraphState) -> dict:
    sub_queries = state.get("sub_queries") or [state["query"]]
    return _retrieve_for_queries(sub_queries, state, stage_name="retrieve_parallel")


def _retrieve_for_queries(
    queries: list[str], state: GraphState, *, stage_name: str
) -> dict:
    t0 = time.perf_counter()
    all_chunks: list[RetrievedChunk] = []
    seen: set[str] = set()
    top_cosine = 0.0
    for q in queries:
        results = hybrid_retrieve(q, k=PER_QUERY_K)
        for r in results:
            if r.chunk_id in seen:
                continue
            seen.add(r.chunk_id)
            all_chunks.append(r)
            if r.cosine_sim is not None and r.cosine_sim > top_cosine:
                top_cosine = r.cosine_sim

    # Convert to serializable dicts for state (audit log will snapshot these)
    chunks_serializable = [_to_dict(c) for c in all_chunks]

    refusal_emitted = top_cosine < COSINE_REFUSAL_THRESHOLD
    refusal_reason = "no_relevant_chunks" if refusal_emitted else None

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    timings = dict(state.get("timings", {}))
    timings[stage_name] = elapsed_ms

    log.info(
        "%s: %d queries → %d unique chunks, top_cosine=%.3f%s, in %dms",
        stage_name, len(queries), len(all_chunks), top_cosine,
        " [REFUSE]" if refusal_emitted else "",
        elapsed_ms,
    )

    update = {
        "retrieved_chunks": chunks_serializable,
        "top_cosine_sim": top_cosine,
        "refusal_emitted": refusal_emitted,
        "refusal_reason": refusal_reason,
        "timings": timings,
    }
    if refusal_emitted:
        update["final_answer"] = (
            f"This question doesn't appear to be answerable from the FERC corpus. "
            f"The closest retrieved passage had a similarity score of {top_cosine:.2f}, "
            f"below the relevance threshold. The system declines to answer rather than "
            f"speculate beyond what the corpus supports."
        )
    return update


def _to_dict(c: RetrievedChunk) -> dict:
    return {
        "chunk_id": c.chunk_id,
        "accession_number": c.accession_number,
        "section_heading": c.section_heading,
        "paragraph_range": c.paragraph_range,
        "chunk_text": c.chunk_text,
        "parent_chunk_id": c.parent_chunk_id,
        "vector_rank": c.vector_rank,
        "keyword_rank": c.keyword_rank,
        "cosine_sim": c.cosine_sim,
        "ts_rank": c.ts_rank,
        "floor_match": c.floor_match,
        "rrf_score": c.rrf_score,
    }
