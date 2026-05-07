"""Eval metrics for RegRAG (per docs/implementation-plan.md §2.9):

  1. Retrieval recall — for each `answer`-expected question, fraction of
     `expected_passages_keywords` that appear (case-insensitive substring)
     in the union of retrieved chunk texts.
  2. Citation faithfulness — judge per-claim with Sonnet (see judge.py),
     averaged per question then macro-averaged across questions.
  3. Refusal accuracy — for each `refuse`-expected question, did the system
     refuse? For `answer`-expected, did it NOT refuse? Binary, averaged.

All metrics report macro averages with per-persona breakdown.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class QuestionResult:
    id: str
    persona: str
    expected_behavior: str

    # System output captured for scoring
    classification: str | None = None
    sub_queries: list[str] | None = None
    retrieved_chunks: list[dict] = field(default_factory=list)
    final_answer: str = ""
    refusal_emitted: bool = False
    refusal_reason: str | None = None
    citations_stripped: int = 0
    regeneration_count: int = 0
    timings: dict[str, int] = field(default_factory=dict)

    # Scores populated by metric functions
    retrieval_recall: float | None = None       # 0.0–1.0; None if not applicable
    refusal_correct: bool | None = None         # True / False; None if N/A
    citation_faithfulness: float | None = None  # 0.0–1.0; None if no citations to judge
    judge_notes: list[dict] = field(default_factory=list)
    error: str | None = None  # populated if the chat invocation crashed


def score_retrieval_recall(
    chunks: list[dict], expected_keywords: list[str]
) -> tuple[float, list[str]]:
    """Fraction of expected_keywords present (case-insensitive substring) in any
    retrieved chunk. Returns (recall_fraction, list_of_missing_keywords)."""
    if not expected_keywords:
        return 1.0, []
    haystack = " \n ".join((c.get("chunk_text") or "") for c in chunks).lower()
    found = [kw for kw in expected_keywords if kw.lower() in haystack]
    missing = [kw for kw in expected_keywords if kw.lower() not in haystack]
    return len(found) / len(expected_keywords), missing


def score_refusal(expected_behavior: str, refusal_emitted: bool) -> bool:
    """True if the system did the right thing (refused when expected, answered when expected)."""
    if expected_behavior == "refuse":
        return refusal_emitted
    if expected_behavior == "answer":
        return not refusal_emitted
    # 'clarify' or unknown: treat as not-refused expected
    return not refusal_emitted


@dataclass
class AggregateReport:
    n_questions: int
    n_errors: int
    retrieval_recall_macro: float | None
    citation_faithfulness_macro: float | None
    refusal_accuracy: float
    by_persona: dict[str, dict[str, float]]
    by_question: list[QuestionResult]


def aggregate(results: Iterable[QuestionResult]) -> AggregateReport:
    results = list(results)
    n = len(results)
    n_err = sum(1 for r in results if r.error is not None)

    # Retrieval recall — only applies to answer-expected, non-error questions
    rr_vals = [r.retrieval_recall for r in results
               if r.expected_behavior == "answer" and r.retrieval_recall is not None and r.error is None]
    rr_macro = sum(rr_vals) / len(rr_vals) if rr_vals else None

    # Citation faithfulness — only when we ran the judge (non-refused, non-error)
    cf_vals = [r.citation_faithfulness for r in results
               if r.citation_faithfulness is not None and r.error is None]
    cf_macro = sum(cf_vals) / len(cf_vals) if cf_vals else None

    # Refusal accuracy — over all questions (excluding errors)
    refusal_vals = [r.refusal_correct for r in results
                    if r.refusal_correct is not None and r.error is None]
    refusal_acc = sum(1 for v in refusal_vals if v) / len(refusal_vals) if refusal_vals else 0.0

    # Per-persona breakdown
    by_persona: dict[str, dict[str, float]] = defaultdict(lambda: {"n": 0, "rr": 0.0, "rr_n": 0,
                                                                   "cf": 0.0, "cf_n": 0,
                                                                   "ref": 0.0, "ref_n": 0})
    for r in results:
        if r.error is not None:
            continue
        b = by_persona[r.persona]
        b["n"] += 1
        if r.retrieval_recall is not None:
            b["rr"] += r.retrieval_recall
            b["rr_n"] += 1
        if r.citation_faithfulness is not None:
            b["cf"] += r.citation_faithfulness
            b["cf_n"] += 1
        if r.refusal_correct is not None:
            if r.refusal_correct:
                b["ref"] += 1
            b["ref_n"] += 1

    by_persona_clean = {}
    for p, b in by_persona.items():
        by_persona_clean[p] = {
            "n": b["n"],
            "retrieval_recall": (b["rr"] / b["rr_n"]) if b["rr_n"] else None,
            "citation_faithfulness": (b["cf"] / b["cf_n"]) if b["cf_n"] else None,
            "refusal_accuracy": (b["ref"] / b["ref_n"]) if b["ref_n"] else None,
        }

    return AggregateReport(
        n_questions=n,
        n_errors=n_err,
        retrieval_recall_macro=rr_macro,
        citation_faithfulness_macro=cf_macro,
        refusal_accuracy=refusal_acc,
        by_persona=by_persona_clean,
        by_question=results,
    )
