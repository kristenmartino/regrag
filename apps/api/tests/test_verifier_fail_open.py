"""Tests for the verifier fail-closed policy (issue #12).

Three fail-open paths are closed: a no-citation draft, a judge that can't run,
and an all-citations-stripped draft must NOT finalize unsupported/unverified
text — the verify node refuses instead. Plus a happy-path test so this stays a
safety fix and not an over-refusal one.

The verify node calls verify_citations + check_accession_scope (pure, no LLM)
and check_substantive_support (one Haiku call). Tests patch the substantive
layer so no network/API key is needed.
"""

from __future__ import annotations

from regrag_api.orchestration.nodes import verify as verify_mod
from regrag_api.orchestration.nodes.verify import (
    _REFUSAL_UNAVAILABLE,
    _REFUSAL_UNRECOVERABLE,
    verify,
)
from regrag_api.verification import substantive as substantive_mod
from regrag_api.verification.substantive import SubstantiveCheckResult, check_substantive_support

CHUNKS = [
    {"chunk_id": "acc1:c1", "accession_number": "acc1", "chunk_text": "Order 2222 requires RTOs to revise tariffs."},
    {"chunk_id": "acc1:c2", "accession_number": "acc1", "chunk_text": "The compliance deadline is 270 days."},
]


def _state(draft: str, *, regen: int = 0) -> dict:
    return {
        "query": "What does Order 2222 require?",
        "draft_answer": draft,
        "retrieved_chunks": CHUNKS,
        "regeneration_count": regen,
        "refusal_emitted": False,
        "timings": {},
    }


def _patch_substantive(monkeypatch, result: SubstantiveCheckResult) -> None:
    monkeypatch.setattr(verify_mod, "check_substantive_support", lambda **kw: result)


# ─── Path 1: no-citation draft ───────────────────────────────────────


def test_no_citation_draft_regenerates_when_budget_remains(monkeypatch):
    _patch_substantive(monkeypatch, SubstantiveCheckResult("x [[acc1:c1]]", 0, 0, [], False))
    out = verify(_state("Order 2222 requires tariff revisions.", regen=0))
    assert out["verification_result"].action == "regenerate"
    assert out["regeneration_count"] == 1
    assert not out.get("refusal_emitted")
    assert "final_answer" not in out


def test_no_citation_draft_refuses_when_budget_exhausted(monkeypatch):
    _patch_substantive(monkeypatch, SubstantiveCheckResult("x [[acc1:c1]]", 0, 0, [], False))
    draft = "Order 2222 requires tariff revisions."
    out = verify(_state(draft, regen=2))
    assert out["refusal_emitted"] is True
    assert out["refusal_reason"] == "verification_unrecoverable"
    assert out["final_answer"] == _REFUSAL_UNRECOVERABLE
    assert out["final_answer"] != draft
    # routing must go to END, not a regen loop
    assert out["verification_result"].action == "finalize"


# ─── Path 2: judge failure ───────────────────────────────────────────


def test_judge_failure_refuses_unavailable(monkeypatch):
    _patch_substantive(
        monkeypatch,
        SubstantiveCheckResult(
            cleaned_text="claim [[acc1:c1]]", sentences_stripped=0, citations_stripped=0,
            judge_notes=[{"warning": "judge_unavailable"}], should_regenerate=False, judge_failed=True,
        ),
    )
    draft = "Order 2222 requires X [[acc1:c1]]."
    out = verify(_state(draft, regen=0))
    assert out["refusal_emitted"] is True
    assert out["refusal_reason"] == "verification_unavailable"
    assert out["final_answer"] == _REFUSAL_UNAVAILABLE
    assert out["final_answer"] != draft
    assert out["verification_result"].action == "finalize"


def test_invoke_judge_empty_sets_judge_failed(monkeypatch):
    # _invoke_judge returns {} on any judge exception / parse failure.
    monkeypatch.setattr(substantive_mod, "_invoke_judge", lambda *a, **k: {})
    res = check_substantive_support(query="q", draft="claim [[acc1:c1]].", retrieved_chunks=CHUNKS)
    assert res.judge_failed is True


# ─── Path 3: all-stripped ────────────────────────────────────────────


def test_all_stripped_returns_empty_not_draft(monkeypatch):
    # Judge scores every pair 0 → all sentences stripped → cleaned_text == "".
    monkeypatch.setattr(
        substantive_mod, "_invoke_judge",
        lambda *a, **k: {0: {"score": 0, "reason": ""}, 1: {"score": 0, "reason": ""}},
    )
    res = check_substantive_support(
        query="q",
        draft="Claim one [[acc1:c1]]. Claim two [[acc1:c2]].",
        retrieved_chunks=CHUNKS,
    )
    assert res.cleaned_text == ""
    assert res.sentences_stripped == 2


def test_all_stripped_refuses_at_max_regen(monkeypatch):
    _patch_substantive(
        monkeypatch,
        SubstantiveCheckResult(
            cleaned_text="", sentences_stripped=2, citations_stripped=0,
            judge_notes=[], should_regenerate=True, judge_failed=False,
        ),
    )
    draft = "Claim [[acc1:c1]]."
    out = verify(_state(draft, regen=2))
    assert out["refusal_emitted"] is True
    assert out["refusal_reason"] == "verification_unrecoverable"
    assert out["final_answer"] != draft
    assert out["verification_result"].action == "finalize"


# ─── Happy path: no over-refusal ─────────────────────────────────────


def test_supported_draft_finalizes(monkeypatch):
    final = "Order 2222 requires tariff revisions [[acc1:c1]]."
    _patch_substantive(
        monkeypatch,
        SubstantiveCheckResult(
            cleaned_text=final, sentences_stripped=0, citations_stripped=0,
            judge_notes=[{"sentence_idx": 0, "chunk_id": "acc1:c1", "score": 1}],
            should_regenerate=False, judge_failed=False,
        ),
    )
    out = verify(_state(final, regen=0))
    assert not out.get("refusal_emitted")
    assert out["final_answer"] == final
