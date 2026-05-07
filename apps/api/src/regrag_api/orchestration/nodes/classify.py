"""Classifier node — determines whether a query is a single-doc lookup or a
multi-doc synthesis task. Per docs/implementation-plan.md §2.5: few-shot Haiku,
returns JSON {intent, confidence}."""

from __future__ import annotations

import logging
import time

from .._anthropic import CLASSIFIER_MODEL, get_client, parse_json_response
from ..state import GraphState

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You classify FERC regulatory queries as either single-document lookups or multi-document synthesis tasks.

single_doc: the answer comes from one specific FERC order, ruling, or filing. Examples:
- "What does Order 2222 require for DER aggregation reporting?"
- "Summarize the dissent in Order 745"
- "What's the deadline for compliance with Order 841?"
- "Who chaired the Commission when Order 2222 was issued?"

multi_doc: the answer requires comparing, synthesizing, or evolving across multiple FERC documents. Examples:
- "How has FERC's treatment of capacity market participation evolved across recent rulings?"
- "Compare DER treatment across Orders 2222, 841, and 745"
- "What are the differences between the storage and DER aggregation rules?"
- "Trace FERC's stance on demand response from Order 745 through Order 2222"

Return ONLY valid JSON, no commentary, no markdown fences:
{"intent": "single_doc", "confidence": 0.95}
"""


def classify(state: GraphState) -> dict:
    t0 = time.perf_counter()
    client = get_client()
    response = client.messages.create(
        model=CLASSIFIER_MODEL,
        max_tokens=64,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": state["query"]}],
    )
    text = response.content[0].text
    parsed = parse_json_response(text)
    intent = parsed.get("intent")
    if intent not in ("single_doc", "multi_doc"):
        log.warning("classifier returned unexpected intent %r — defaulting to single_doc", intent)
        intent = "single_doc"
    confidence = float(parsed.get("confidence", 0.5))

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    timings = dict(state.get("timings", {}))
    timings["classify"] = elapsed_ms
    model_ids = dict(state.get("model_ids_used", {}))
    model_ids["classify"] = CLASSIFIER_MODEL
    token_counts = dict(state.get("token_counts", {}))
    token_counts["classify"] = {
        "in": response.usage.input_tokens,
        "out": response.usage.output_tokens,
    }

    log.info("classified %r as %s (%.2f) in %dms", state["query"], intent, confidence, elapsed_ms)
    return {
        "classification": intent,
        "classification_confidence": confidence,
        "timings": timings,
        "model_ids_used": model_ids,
        "token_counts": token_counts,
    }
