"""Answerability gate — a flagged pre-synthesis check.

STATUS (issue #3 A/B, 2026-05-29): OFF by default; an accuracy-neutral rebalance.
The 55-question A/B (restricted to single_doc queries) found the gate is a *wash*
on refusal accuracy (90.9% = 90.9% vs self-flag) — it trades −6.3pp precision
(90.9% → 84.6%) for +8.7pp recall (87.0% → 95.7%). The synthesizer's existing
`refused: true` self-flag ALREADY catches every false_premise (6/6) and
order_conflation (5/5) case, so the gate's entire contribution is closing the
jurisdiction_boundary gap (2/4 → 4/4). Worth enabling only if a deployment's query
mix is jurisdiction-heavy and favors recall ("I don't know" over a confident wrong
answer); otherwise leave off. See docs/eval-results.md "Answerability gate experiment".

Runs between retrieve and synthesize when REGRAG_ANSWERABILITY_GATE is set.
Asks a cheap Haiku call: do the retrieved chunks actually contain what's needed
to answer the question AS ASKED? This catches the failure mode the synthesizer's
own `refused: true` self-flag misses — questions that name an in-corpus order (so
retrieval returns high-cosine chunks) but whose *premise* the corpus doesn't
support:

  - false premise: "Under Order 2222, how much do distribution utilities pay
    aggregators per MWh?" (Order 2222 doesn't set such a rate)
  - order conflation: "What are Order 841's generator-interconnection queue
    reforms?" (841 is storage participation; interconnection is 845/2023)
  - jurisdiction boundary: "Does Order 2222 set retail net-metering rates?"
    (retail is state jurisdiction)

Design notes:
  - OFF by default. The flag lets the eval A/B the gated path against the current
    synthesizer-self-flag path (see docs/eval-results.md, issue #3).
  - No-op when retrieval already refused (low-cosine) — that path ends before us.
  - Conservative by construction: the prompt biases toward `answerable: true` so
    the gate protects refusal *precision* (doesn't false-refuse in-scope queries)
    while still catching the clear no-support cases that hurt refusal *recall*.
"""

from __future__ import annotations

import logging
import os
import time

from .._anthropic import CLASSIFIER_MODEL, get_client, parse_json_response
from ..state import GraphState

log = logging.getLogger(__name__)

GATE_ENV_FLAG = "REGRAG_ANSWERABILITY_GATE"
MAX_CHUNKS_IN_PROMPT = 10
MAX_CHARS_PER_CHUNK = 500

SYSTEM_PROMPT = """\
You are a gatekeeper for a FERC regulatory question-answering system. Given a user question and the passages retrieved for it, decide ONE thing: do these passages contain the specific information needed to answer the question AS ASKED?

"As asked" is the crux. A question can be about an order that appears in the passages and STILL be unanswerable, when the question presupposes something the passages don't support. Watch for three patterns:

1. FALSE PREMISE — the question asks for a specific fact (a rate, penalty, deadline, quantity, mandate) that the order does not actually establish. Example: "Under Order 2222, how much must distribution utilities pay aggregators per MWh?" — Order 2222 routes aggregator compensation through RTO/ISO markets; it sets no such per-MWh payment from distribution utilities. The passages are about Order 2222 but contain no such rate. NOT answerable.

2. ORDER CONFLATION — the question attributes a subject to an order that doesn't cover it. Example: "What are Order 841's generator-interconnection queue reforms?" — Order 841 is electric-storage market participation; interconnection queue reform is Orders 845/2023. Passages from 841 won't contain interconnection-queue reforms. NOT answerable.

3. JURISDICTION / SCOPE BOUNDARY — the question asks about something outside what these federal orders address (retail rates, net metering, state PUC procedures, pipeline permits, agency budgets). NOT answerable from this corpus.

Decision rule:
- answerable=true if the passages contain facts that substantively address the question as asked (even if only partially — the downstream system will scope the answer).
- answerable=false if answering would require inventing a fact the passages don't contain, correcting a false premise, or going outside what the passages cover.

IMPORTANT — protect against over-refusal: when the passages plausibly support a real answer, choose answerable=true. Only choose answerable=false when the passages clearly lack the specific information the question asks for. A borderline-but-supported question is answerable.

Return ONLY valid JSON, no commentary, no markdown fences:
{"answerable": true, "reason": "one short sentence"}
or
{"answerable": false, "reason": "one short sentence naming what's missing or the false premise"}
"""


def answerability_gate(state: GraphState) -> dict:
    """Flagged gate. Returns {} (pass-through) unless the flag is on AND the
    chunks don't support the question, in which case it emits a refusal."""
    # Flag off → no-op. Keep the node in the graph always; gate behavior is
    # purely runtime so the eval can toggle it without recompiling.
    if os.environ.get(GATE_ENV_FLAG, "").strip().lower() not in ("1", "true", "yes", "on"):
        return {}

    # Retrieval already refused (low cosine) → nothing to gate; let routing end it.
    if state.get("refusal_emitted"):
        return {"answerability_checked": False}

    # Only gate single-document queries. Data-driven (issue #3 A/B on the 55-q
    # set): gating ALL queries hits 100% refusal recall but collapses precision
    # 90.9%→63.9% by false-refusing 9 multi-doc synthesis questions — for a
    # cross-document question, each retrieved chunk only partially covers the
    # ask, so the gate misjudges distributed-but-present answers as unanswerable.
    # Restricting to single_doc makes it accuracy-neutral vs the self-flag
    # (90.9% = 90.9%): precision 84.6%, recall 95.7% — a −6.3pp/+8.7pp rebalance
    # that fully closes the jurisdiction_boundary gap (2/4 → 4/4). Multi-doc
    # queries fall back to the synthesizer's own refusal self-flag.
    if state.get("classification") == "multi_doc":
        return {"answerability_checked": False}

    chunks = state.get("retrieved_chunks") or []
    if not chunks:
        # No chunks but retrieval didn't flag refusal — treat as unanswerable.
        return _refuse(state, "No passages were retrieved for this question.", t0=time.perf_counter())

    t0 = time.perf_counter()
    chunks_block = _format_chunks(chunks[:MAX_CHUNKS_IN_PROMPT])
    user_msg = f"QUESTION:\n{state['query']}\n\nRETRIEVED PASSAGES:\n{chunks_block}"

    client = get_client()
    try:
        response = client.messages.create(
            model=CLASSIFIER_MODEL,
            max_tokens=200,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
    except Exception as e:
        # Fail OPEN: if the gate call errors, don't block the answer — pass
        # through to synthesize (which has its own refusal path). A flaky gate
        # must never make the system worse than the ungated path.
        log.warning("answerability gate call failed (%s: %s) — passing through", type(e).__name__, e)
        return {"answerability_checked": False}

    text = response.content[0].text
    try:
        parsed = parse_json_response(text)
        answerable = bool(parsed.get("answerable", True))
        reason = str(parsed.get("reason", "")).strip()
    except (ValueError, Exception) as e:
        # Unparseable verdict → fail open (treat as answerable).
        log.warning("answerability gate parse failed (%s) — treating as answerable", type(e).__name__)
        answerable = True
        reason = ""

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    timings = dict(state.get("timings", {}))
    timings["answerability_gate"] = elapsed_ms
    model_ids = dict(state.get("model_ids_used", {}))
    model_ids["answerability_gate"] = CLASSIFIER_MODEL
    token_counts = dict(state.get("token_counts", {}))
    token_counts["answerability_gate"] = {
        "in": response.usage.input_tokens,
        "out": response.usage.output_tokens,
    }

    base = {
        "answerability_checked": True,
        "answerability_verdict": answerable,
        "answerability_reason": reason or None,
        "timings": timings,
        "model_ids_used": model_ids,
        "token_counts": token_counts,
    }

    if answerable:
        log.info("answerability gate: ANSWERABLE — %s", reason)
        return base

    log.info("answerability gate: UNANSWERABLE — %s", reason)
    refusal_text = (
        "This question can't be answered from the FERC corpus as posed"
        + (f": {reason}" if reason else ".")
    )
    return {
        **base,
        "refusal_emitted": True,
        "refusal_reason": "unanswerable_from_corpus",
        "final_answer": refusal_text,
    }


def _refuse(state: GraphState, reason: str, *, t0: float) -> dict:
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    timings = dict(state.get("timings", {}))
    timings["answerability_gate"] = elapsed_ms
    return {
        "answerability_checked": True,
        "answerability_verdict": False,
        "answerability_reason": reason,
        "refusal_emitted": True,
        "refusal_reason": "unanswerable_from_corpus",
        "final_answer": f"This question can't be answered from the FERC corpus as posed: {reason}",
        "timings": timings,
    }


def _format_chunks(chunks: list[dict]) -> str:
    lines: list[str] = []
    for c in chunks:
        section = c.get("section_heading") or "?"
        text = (c.get("chunk_text") or "").strip()[:MAX_CHARS_PER_CHUNK]
        lines.append(
            f"[accession={c.get('accession_number')} | section={section}]\n{text}\n"
        )
    return "\n".join(lines)
