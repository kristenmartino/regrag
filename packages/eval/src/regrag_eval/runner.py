"""Eval runner — walks eval_set.yaml, calls the chat graph for each question,
scores it, and accumulates a report."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Iterable

import yaml

from regrag_api.orchestration.graph import run as run_graph

from .judge import judge_citations
from .metrics import (
    QuestionResult,
    aggregate,
    AggregateReport,
    score_refusal,
    score_retrieval_recall,
)

log = logging.getLogger(__name__)


def load_eval_set(path: Path) -> list[dict]:
    raw = yaml.safe_load(path.read_text())
    return raw.get("questions", [])


def run_one_question(question: dict, *, run_judge: bool = True) -> QuestionResult:
    qid = question["id"]
    persona = question["persona"]
    expected_behavior = question["expected_behavior"]
    query = question["query"]

    result = QuestionResult(id=qid, persona=persona, expected_behavior=expected_behavior)

    try:
        state = run_graph(query, user_id="eval-runner")
    except Exception as e:
        log.error("question %s crashed: %s", qid, e)
        result.error = f"{type(e).__name__}: {e}"
        return result

    result.classification = state.get("classification")
    result.sub_queries = state.get("sub_queries")
    result.retrieved_chunks = state.get("retrieved_chunks") or []
    result.final_answer = state.get("final_answer") or ""
    result.refusal_emitted = bool(state.get("refusal_emitted"))
    result.refusal_reason = state.get("refusal_reason")
    result.citations_stripped = state.get("citations_stripped", 0)
    result.sentences_stripped = state.get("sentences_stripped", 0)
    result.substantive_citations_stripped = state.get("substantive_citations_stripped", 0)
    result.regeneration_count = state.get("regeneration_count", 0)
    result.timings = state.get("timings", {})

    # Refusal correctness
    result.refusal_correct = score_refusal(expected_behavior, result.refusal_emitted)

    # Retrieval recall (only for answer-expected and only if we got chunks)
    if expected_behavior == "answer":
        expected_kw = question.get("expected_passages_keywords") or []
        if expected_kw and result.retrieved_chunks:
            recall, missing = score_retrieval_recall(result.retrieved_chunks, expected_kw)
            result.retrieval_recall = recall
            if missing:
                result.judge_notes.append({"missing_keywords": missing})

    # Citation faithfulness (only for answers, only if judge enabled)
    if (
        run_judge
        and expected_behavior == "answer"
        and not result.refusal_emitted
        and result.final_answer
    ):
        try:
            cf, notes = judge_citations(query, result.final_answer, result.retrieved_chunks)
            result.citation_faithfulness = cf
            if notes:
                result.judge_notes.extend(notes)
        except Exception as e:
            log.warning("judge crashed on %s: %s", qid, e)
            result.judge_notes.append({"judge_error": f"{type(e).__name__}: {e}"})

    return result


def run_all(
    questions: Iterable[dict],
    *,
    run_judge: bool = True,
    progress_callback=None,
) -> AggregateReport:
    results: list[QuestionResult] = []
    questions = list(questions)
    for i, q in enumerate(questions, 1):
        if progress_callback:
            progress_callback(i, len(questions), q["id"])
        t0 = time.perf_counter()
        r = run_one_question(q, run_judge=run_judge)
        elapsed = int((time.perf_counter() - t0) * 1000)
        log.info(
            "[%d/%d] %s persona=%s expected=%s → refused=%s recall=%s cf=%s in %dms",
            i, len(questions), r.id, r.persona, r.expected_behavior,
            r.refusal_emitted, r.retrieval_recall, r.citation_faithfulness, elapsed,
        )
        results.append(r)
    return aggregate(results)


def report_to_dict(report: AggregateReport) -> dict:
    """Serializable form for writing to JSON."""
    return {
        "summary": {
            "n_questions": report.n_questions,
            "n_errors": report.n_errors,
            "retrieval_recall_macro": report.retrieval_recall_macro,
            "citation_faithfulness_macro": report.citation_faithfulness_macro,
            "refusal_accuracy": report.refusal_accuracy,
        },
        "by_persona": report.by_persona,
        "by_question": [
            {
                "id": r.id,
                "persona": r.persona,
                "expected_behavior": r.expected_behavior,
                "classification": r.classification,
                "refusal_emitted": r.refusal_emitted,
                "refusal_correct": r.refusal_correct,
                "retrieval_recall": r.retrieval_recall,
                "citation_faithfulness": r.citation_faithfulness,
                "citations_stripped": r.citations_stripped,
                "sentences_stripped": r.sentences_stripped,
                "substantive_citations_stripped": r.substantive_citations_stripped,
                "regeneration_count": r.regeneration_count,
                "n_chunks_retrieved": len(r.retrieved_chunks),
                "timings_ms": r.timings,
                "answer_preview": (r.final_answer[:300] + "...") if len(r.final_answer) > 300 else r.final_answer,
                "judge_notes": r.judge_notes,
                "error": r.error,
            }
            for r in report.by_question
        ],
    }
