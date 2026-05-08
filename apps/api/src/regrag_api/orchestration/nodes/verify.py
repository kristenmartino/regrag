"""Verify node — two-step citation verification on the draft answer.

Step 1 (cheap, no LLM): verify cited chunk_ids exist in the retrieved set.
  Strip unknown chunk_ids; regenerate if drift > 30%.

Step 2 (one Haiku call): verify each cited chunk SUBSTANTIVELY supports its
  claim. Strip unsupported sentences; regenerate if > 50% are stripped.

The two steps are sequenced so step 2 only runs on drafts that pass step 1.
This keeps the cheap regen path fast while adding substantive checking for
drafts that survive citation-id verification.
"""

from __future__ import annotations

import logging
import time

from ...verification.citations import verify_citations
from ...verification.substantive import check_substantive_support
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

    # ─── Step 1: chunk-id presence check ───
    result = verify_citations(draft, retrieved_chunk_ids, regeneration_count=regen_count)

    update: dict = {
        "verification_result": result,
        "cited_chunk_ids": result.valid_citations,
        "citations_stripped": len(result.invalid_citations) if result.action == "finalize" else 0,
    }

    if result.action == "regenerate":
        # Step 2 doesn't run when we're regenerating
        update["regeneration_count"] = regen_count + 1
        timings = dict(state.get("timings", {}))
        timings["verify"] = timings.get("verify", 0) + int((time.perf_counter() - t0) * 1000)
        update["timings"] = timings
        return update

    # ─── Step 2: substantive support check (LLM judge) ───
    sub = check_substantive_support(
        query=state.get("query", ""),
        draft=result.cleaned_text,
        retrieved_chunks=retrieved,
    )

    timings = dict(state.get("timings", {}))
    timings["verify"] = timings.get("verify", 0) + int((time.perf_counter() - t0) * 1000)
    update["timings"] = timings
    update["sentences_stripped"] = sub.sentences_stripped
    update["substantive_citations_stripped"] = sub.citations_stripped
    update["judge_notes"] = sub.judge_notes

    if sub.should_regenerate and regen_count < 2:
        log.info("substantive judge: high strip rate → regenerate")
        update["regeneration_count"] = regen_count + 1
        # Override the verification_result action so the graph routing edge
        # ('regenerate_or_finalize') sees a regen signal. Without this, the
        # action from step 1 (which finalized) wins and the graph ends instead
        # of looping back to synthesize.
        from dataclasses import replace
        update["verification_result"] = replace(
            result, action="regenerate", notes=f"substantive judge: {sub.sentences_stripped}/{len(sub.judge_notes) or 1} sentences stripped"
        )
        # final_answer stays unset; graph routes back to synthesize
        return update

    update["final_answer"] = sub.cleaned_text
    return update


def regenerate_or_finalize(state: GraphState) -> str:
    """Conditional edge selector: returns the next node name."""
    result = state.get("verification_result")
    if result is None:
        return "finalize"
    return "regenerate" if result.action == "regenerate" else "finalize"
