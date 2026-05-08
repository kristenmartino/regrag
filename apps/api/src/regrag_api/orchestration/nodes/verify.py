"""Verify node — three-step citation verification on the draft answer.

Step 1 (no LLM): quote verification — for each structured claim, substring-check
  the supporting_quote against the chunk_text it cites. Drop claims whose
  quote isn't actually in the chunk. Regenerate if keep ratio < 50%.

Step 2 (no LLM): chunk-id presence check — strip cited chunk_ids that aren't
  in the retrieved set. Regenerate if drift > 30%. (Mostly redundant with
  step 1 now that the synthesizer emits structured claims, but kept as a
  defensive layer in case rendered text gets out of sync with structured claims.)

Step 3 (one Haiku call): LLM-judge substantive support — score each (sentence,
  cited_chunk) pair 0/1; drop sentences whose all citations score 0. This
  catches the case where the quote IS in the chunk but the claim paraphrases
  too aggressively. Regenerate if > 50% of sentences are stripped.

The three steps are sequenced cheapest-first. Each can independently trigger
regeneration; the existing 2-attempt cap bounds the overall loop.
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace

from ...verification.citations import verify_citations
from ...verification.quotes import StructuredClaim, verify_claim_quotes
from ...verification.substantive import check_substantive_support
from ..state import GraphState

log = logging.getLogger(__name__)

# If quote verification keeps fewer than this fraction of claims, regenerate.
QUOTE_KEEP_REGEN_THRESHOLD = 0.5


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
    structured_claims_raw = state.get("structured_claims") or []

    # ─── Step 0: quote verification on structured claims ───
    # Only runs if synthesize emitted structured claims (new path). If absent,
    # this is a no-op — falls through to the legacy chunk-id + substantive checks
    # so older drafts and the test fallback still work.
    quote_update: dict = {}
    if structured_claims_raw:
        as_dataclasses = [
            StructuredClaim(
                claim=c.get("claim", ""),
                chunk_id=c.get("chunk_id", ""),
                supporting_quote=c.get("supporting_quote", ""),
            )
            for c in structured_claims_raw
        ]
        qv = verify_claim_quotes(as_dataclasses, retrieved)
        quote_update["claims_kept_after_quotes"] = len(qv.kept_claims)
        quote_update["claims_dropped_for_bad_quote"] = (
            len(qv.dropped_for_quote_not_found)
            + len(qv.dropped_for_unknown_chunk)
            + len(qv.dropped_for_short_quote)
        )

        if qv.keep_ratio < QUOTE_KEEP_REGEN_THRESHOLD and regen_count < 2:
            log.info(
                "verify step 0: quote keep_ratio %.0f%% < threshold → regenerate (attempt %d)",
                qv.keep_ratio * 100, regen_count + 1,
            )
            timings = dict(state.get("timings", {}))
            timings["verify"] = timings.get("verify", 0) + int((time.perf_counter() - t0) * 1000)
            from ..state import VerificationResult
            return {
                **quote_update,
                "verification_result": VerificationResult(
                    valid_citations=[c.chunk_id for c in qv.kept_claims],
                    invalid_citations=[c.chunk_id for c in (
                        qv.dropped_for_quote_not_found
                        + qv.dropped_for_unknown_chunk
                        + qv.dropped_for_short_quote
                    )],
                    cleaned_text="",
                    action="regenerate",
                    notes=f"quote verification: {qv.keep_ratio:.0%} keep ratio",
                ),
                "regeneration_count": regen_count + 1,
                "timings": timings,
            }

        # Re-render the draft from the surviving claims so step 1+2 see the cleaned text
        kept_text_parts: list[str] = []
        for c in qv.kept_claims:
            kept_text_parts.append(f"{c.claim.rstrip(' .')}. [[{c.chunk_id}]]")
        draft = " ".join(kept_text_parts)

    # ─── Step 1: chunk-id presence check ───
    result = verify_citations(draft, retrieved_chunk_ids, regeneration_count=regen_count)

    update: dict = {
        **quote_update,
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
