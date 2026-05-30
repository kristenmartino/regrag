"""Verify node — two-step citation verification on the draft answer.

Step 1 (cheap, no LLM): verify cited chunk_ids exist in the retrieved set.
  Strip unknown chunk_ids; regenerate if drift > 30%.

Step 2 (one Haiku call): verify each cited chunk SUBSTANTIVELY supports its
  claim. Strip unsupported sentences; regenerate if > 50% are stripped.

The two steps are sequenced so step 2 only runs on drafts that pass step 1.
This keeps the cheap regen path fast while adding substantive checking for
drafts that survive citation-id verification.

Fail-closed policy (issue #12): the verifier never finalizes empty, uncited, or
unverified text. When a draft can't be grounded — no citations after the regen
budget, all cited support stripped, or the substantive judge unavailable — the
node emits a refusal instead of passing unsupported text through. The
finalize-time backstop (the `_CITATION_RE` check) is authoritative even if an
earlier branch is later refactored away.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import replace

from ...verification.citations import verify_citations
from ...verification.scope import check_accession_scope
from ...verification.substantive import check_substantive_support
from ..state import GraphState, RefusalReason, VerificationResult

log = logging.getLogger(__name__)

# Citation marker — used by the finalize backstop to refuse uncited final text.
_CITATION_RE = re.compile(r"\[\[[^\]]+\]\]")

# User-facing refusal messages (kept short and safe).
_REFUSAL_UNRECOVERABLE = (
    "I couldn't verify enough cited support in the retrieved FERC corpus to answer "
    "reliably, so I'm declining rather than returning an unsupported answer."
)
_REFUSAL_UNAVAILABLE = (
    "The citation verifier is temporarily unavailable, so I can't safely return an "
    "answer right now. Please try again in a moment."
)


def _verification_refusal(
    state: GraphState,
    *,
    reason: RefusalReason,
    message: str,
    t0: float,
    verification_result: VerificationResult | None = None,
) -> dict:
    """Build a refusal update for the verify node.

    Sets refusal_emitted + reason + final_answer and accumulates the verify
    timing. The caller merges this over the node's accumulated update, whose
    verification_result has action='finalize' — so the graph routes to END,
    never back into a regen loop.
    """
    timings = dict(state.get("timings", {}))
    timings["verify"] = timings.get("verify", 0) + int((time.perf_counter() - t0) * 1000)
    update: dict = {
        "refusal_emitted": True,
        "refusal_reason": reason,
        "final_answer": message,
        "timings": timings,
    }
    if verification_result is not None:
        update["verification_result"] = verification_result
    return update


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
        # Step 2/3 don't run when we're regenerating
        update["regeneration_count"] = regen_count + 1
        timings = dict(state.get("timings", {}))
        timings["verify"] = timings.get("verify", 0) + int((time.perf_counter() - t0) * 1000)
        update["timings"] = timings
        return update

    # ─── Path 1: no-citation draft (issue #12) ───
    # A non-refused draft with zero citations is unverifiable. Give the model one
    # more attempt to cite — a no-citation draft is often a recoverable formatting
    # failure — and refuse once the regen budget is exhausted. (The finalize
    # backstop below is the authoritative guard; catching it here just avoids a
    # wasted judge call on an uncited draft.)
    if not result.valid_citations and not result.invalid_citations:
        if regen_count < 2:
            log.info("verify: draft has no citations → regenerate (attempt %d)", regen_count + 1)
            update["regeneration_count"] = regen_count + 1
            update["verification_result"] = replace(result, action="regenerate", notes="no citations in draft")
            timings = dict(state.get("timings", {}))
            timings["verify"] = timings.get("verify", 0) + int((time.perf_counter() - t0) * 1000)
            update["timings"] = timings
            return update
        log.warning("verify: draft still uncited after regen budget → refuse")
        return {**update, **_verification_refusal(
            state, reason="verification_unrecoverable", message=_REFUSAL_UNRECOVERABLE, t0=t0)}

    # ─── Step 1.5: accession-scope check (no LLM, cheap) ───
    # Enforces per-sentence in-scope citation rule when the query named an
    # order. Safety net for cases where the synthesis SCOPE prompt didn't
    # constrain the model fully. Runs BEFORE the substantive judge because
    # this is a cheap rule that can short-circuit a regen.
    named_orders = state.get("named_orders") or []
    anchored_roles = state.get("anchored_roles") or {}
    scope = check_accession_scope(
        result.cleaned_text,
        named_orders,
        anchored_roles,
        regeneration_count=regen_count,
    )
    if scope.sentences_checked > 0:
        log.info("scope check: %s", scope.notes)
    if scope.should_regenerate and regen_count < 2:
        log.info(
            "scope: %d/%d in-subject sentences violate scope → regenerate",
            scope.sentences_violating, scope.sentences_checked,
        )
        update["scope_citations_stripped"] = len(scope.citations_stripped)
        update["scope_sentences_dropped"] = len(scope.sentences_dropped)
        update["regeneration_count"] = regen_count + 1
        update["verification_result"] = replace(
            result, action="regenerate",
            notes=f"scope: {scope.sentences_violating}/{scope.sentences_checked} sentences violate in-scope rule",
        )
        timings = dict(state.get("timings", {}))
        timings["verify"] = timings.get("verify", 0) + int((time.perf_counter() - t0) * 1000)
        update["timings"] = timings
        return update

    # Apply scope stripping (cleaned_text now has out-of-scope cites removed)
    update["scope_citations_stripped"] = len(scope.citations_stripped)
    update["scope_sentences_dropped"] = len(scope.sentences_dropped)

    # ─── Step 2: substantive support check (LLM judge) ───
    sub = check_substantive_support(
        query=state.get("query", ""),
        draft=scope.cleaned_text,  # use scope-cleaned text, not raw result
        retrieved_chunks=retrieved,
    )

    timings = dict(state.get("timings", {}))
    timings["verify"] = timings.get("verify", 0) + int((time.perf_counter() - t0) * 1000)
    update["timings"] = timings
    update["sentences_stripped"] = sub.sentences_stripped
    update["substantive_citations_stripped"] = sub.citations_stripped
    update["judge_notes"] = sub.judge_notes

    # ─── Path 2: judge unavailable → refuse (transient) (issue #12) ───
    # A judge that couldn't run is not a judge that passed. Don't regenerate — an
    # unavailable verifier stays unavailable on retry — refuse instead.
    if sub.judge_failed:
        log.warning("verify: substantive judge unavailable → refuse (verification_unavailable)")
        return {**update, **_verification_refusal(
            state, reason="verification_unavailable", message=_REFUSAL_UNAVAILABLE, t0=t0)}

    if sub.should_regenerate and regen_count < 2:
        log.info("substantive judge: high strip rate → regenerate")
        update["regeneration_count"] = regen_count + 1
        # Override the verification_result action so the graph routing edge
        # ('regenerate_or_finalize') sees a regen signal. Without this, the
        # action from step 1 (which finalized) wins and the graph ends instead
        # of looping back to synthesize.
        update["verification_result"] = replace(
            result, action="regenerate",
            notes=f"substantive judge: {sub.sentences_stripped}/{len(sub.judge_notes) or 1} sentences stripped",
        )
        return update

    # ─── Path 3 + authoritative backstop (issue #12): never finalize empty/uncited text ───
    # If every citation was stripped (Path 3) or the text otherwise lost all
    # cited support, refuse rather than reverting to an unsupported answer. This
    # check is the last word — it catches any path that produced ungrounded text.
    final_text = sub.cleaned_text
    if not final_text.strip() or not _CITATION_RE.search(final_text):
        log.warning("verify: post-verification text empty or uncited → refuse (verification_unrecoverable)")
        return {**update, **_verification_refusal(
            state, reason="verification_unrecoverable", message=_REFUSAL_UNRECOVERABLE, t0=t0)}

    update["final_answer"] = final_text
    return update


def regenerate_or_finalize(state: GraphState) -> str:
    """Conditional edge selector: returns the next node name."""
    result = state.get("verification_result")
    if result is None:
        return "finalize"
    return "regenerate" if result.action == "regenerate" else "finalize"
