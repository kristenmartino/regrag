"""Decomposer node — splits a multi-doc query into per-document sub-queries.

Per docs/implementation-plan.md §2.6: Sonnet call with the static corpus
summary in the system prompt. Returns JSON {sub_queries: [...]}.
"""

from __future__ import annotations

import logging
import time

from .._anthropic import DECOMPOSER_MODEL, get_client, parse_json_response
from ..corpus_summary import load_corpus_summary
from ..state import GraphState

log = logging.getLogger(__name__)

MAX_SUB_QUERIES = 6  # cap to bound parallel-retrieval cost


def _build_system_prompt() -> str:
    return f"""\
You decompose multi-document FERC regulatory queries into 2-{MAX_SUB_QUERIES} focused sub-queries.

{load_corpus_summary()}

Each sub-query should be:
- Self-contained (a future retrieval system can answer it without seeing the original question)
- Scoped to one document, sub-topic, or comparison axis
- Phrased as a question the corpus could plausibly answer

Examples:

Query: "Compare DER treatment across Orders 2222, 841, and 745"
Output: {{"sub_queries": [
  "What does Order 2222 say about distributed energy resource aggregation?",
  "What does Order 841 say about distributed storage participation in wholesale markets?",
  "What does Order 745 say about demand response compensation?"
]}}

Query: "How has FERC's treatment of capacity market participation evolved?"
Output: {{"sub_queries": [
  "How does Order 745 address demand response in capacity markets?",
  "How does Order 841 address electric storage capacity market participation?",
  "How does Order 2222 address DER aggregation in capacity markets?"
]}}

Return ONLY valid JSON, no commentary, no markdown fences.
"""


def decompose(state: GraphState) -> dict:
    t0 = time.perf_counter()
    client = get_client()
    response = client.messages.create(
        model=DECOMPOSER_MODEL,
        max_tokens=1024,
        system=_build_system_prompt(),
        messages=[{"role": "user", "content": state["query"]}],
    )
    text = response.content[0].text
    parsed = parse_json_response(text)
    sub_queries = parsed.get("sub_queries") or []
    if not isinstance(sub_queries, list) or not sub_queries:
        log.warning("decomposer returned empty sub_queries — falling back to original query")
        sub_queries = [state["query"]]
    sub_queries = [str(s).strip() for s in sub_queries if s][:MAX_SUB_QUERIES]

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    timings = dict(state.get("timings", {}))
    timings["decompose"] = elapsed_ms
    model_ids = dict(state.get("model_ids_used", {}))
    model_ids["decompose"] = DECOMPOSER_MODEL
    token_counts = dict(state.get("token_counts", {}))
    token_counts["decompose"] = {
        "in": response.usage.input_tokens,
        "out": response.usage.output_tokens,
    }

    log.info("decomposed into %d sub-queries in %dms: %s",
             len(sub_queries), elapsed_ms, sub_queries)
    return {
        "sub_queries": sub_queries,
        "timings": timings,
        "model_ids_used": model_ids,
        "token_counts": token_counts,
    }
