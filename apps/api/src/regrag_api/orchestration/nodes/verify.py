"""Verify node — runs citation verification on the draft answer and either
finalizes the response or routes back to synthesize for regeneration.
"""

from __future__ import annotations

import logging
import time

from ...verification.citations import verify_citations
from ..state import GraphState

log = logging.getLogger(__name__)


def verify(state: GraphState) -> dict:
    t0 = time.perf_counter()

    # If synthesize already emitted a refusal (LLM declined to answer),
    # verify is a no-op — preserve the refusal text and don't try to clean
    # citations that don't exist.
    if state.get("refusal_emitted"):
        log.info("verify: refusal already emitted, skipping citation check")
        return {"timings": dict(state.get("timings", {}))}

    draft = state.get("draft_answer", "") or ""
    retrieved = state.get("retrieved_chunks") or []
    retrieved_chunk_ids = {c["chunk_id"] for c in retrieved}
    regen_count = state.get("regeneration_count", 0)

    result = verify_citations(draft, retrieved_chunk_ids, regeneration_count=regen_count)

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    timings = dict(state.get("timings", {}))
    timings["verify"] = timings.get("verify", 0) + elapsed_ms

    update: dict = {
        "verification_result": result,
        "cited_chunk_ids": result.valid_citations,
        "citations_stripped": len(result.invalid_citations) if result.action == "finalize" else 0,
        "timings": timings,
    }

    if result.action == "regenerate":
        update["regeneration_count"] = regen_count + 1
        # final_answer stays None; the graph routes back to synthesize
    else:
        update["final_answer"] = result.cleaned_text

    return update


def regenerate_or_finalize(state: GraphState) -> str:
    """Conditional edge selector: returns the next node name."""
    result = state.get("verification_result")
    if result is None:
        return "finalize"
    return "regenerate" if result.action == "regenerate" else "finalize"
