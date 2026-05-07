"""StateGraph wiring: classify → (decompose → retrieve_parallel | retrieve_single)
→ synthesize → verify → (regenerate? → synthesize | end).

Compiled lazily on first use.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Iterator

from langgraph.graph import END, START, StateGraph

from ..audit.log import write_query_log
from .nodes.classify import classify
from .nodes.decompose import decompose
from .nodes.retrieve import retrieve_parallel, retrieve_single
from .nodes.synthesize import synthesize
from .nodes.verify import regenerate_or_finalize, verify
from .state import GraphState, initial_state

log = logging.getLogger(__name__)


def _route_by_intent(state: GraphState) -> str:
    return "decompose" if state.get("classification") == "multi_doc" else "retrieve_single"


def _check_pre_gen_refusal(state: GraphState) -> str:
    """If retrieval set refusal_emitted, skip synthesize and end immediately."""
    if state.get("refusal_emitted"):
        return "end"
    return "synthesize"


@lru_cache(maxsize=1)
def build_graph():
    """Build and compile the LangGraph workflow. Cached so repeated CLI calls
    in the same Python process don't pay the construction cost twice."""
    g = StateGraph(GraphState)

    g.add_node("classify", classify)
    g.add_node("decompose", decompose)
    g.add_node("retrieve_single", retrieve_single)
    g.add_node("retrieve_parallel", retrieve_parallel)
    g.add_node("synthesize", synthesize)
    g.add_node("verify", verify)

    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", _route_by_intent, {
        "decompose": "decompose",
        "retrieve_single": "retrieve_single",
    })
    g.add_edge("decompose", "retrieve_parallel")

    # After retrieval, either refuse (skip synthesis) or synthesize
    g.add_conditional_edges("retrieve_single", _check_pre_gen_refusal, {
        "end": END, "synthesize": "synthesize",
    })
    g.add_conditional_edges("retrieve_parallel", _check_pre_gen_refusal, {
        "end": END, "synthesize": "synthesize",
    })

    g.add_edge("synthesize", "verify")
    g.add_conditional_edges("verify", regenerate_or_finalize, {
        "regenerate": "synthesize",
        "finalize": END,
    })

    return g.compile()


def run(query: str, *, user_id: str | None = None, audit: bool = True) -> GraphState:
    """Run the graph end-to-end on a query. Returns the final state.

    audit=True (default) writes one row to query_log per invocation. Failures
    are logged but never raised — the chat answer is more important than the
    audit row.
    """
    graph = build_graph()
    state = graph.invoke(initial_state(query, user_id=user_id))
    if audit:
        write_query_log(state)
    return state


def run_streaming(query: str, *, user_id: str | None = None, audit: bool = True):
    """Generator version of run(): yields one event per node completion plus a
    final 'done' event. Each event is a dict with at least {type, ...}; the
    server wraps each in SSE framing.

    Event shapes:
      {"type": "started", "query": str}
      {"type": "stage_complete", "stage": str, "delta": dict, "elapsed_ms": int}
      {"type": "done", "state": dict}    # full final state, serialized
    """
    import time

    graph = build_graph()
    state = initial_state(query, user_id=user_id)

    yield {"type": "started", "query": query}

    last_timings = dict(state.get("timings") or {})
    for chunk in graph.stream(state, stream_mode="updates"):
        # chunk is {node_name: state_delta}
        for node_name, delta in chunk.items():
            # Merge delta into our running state copy
            for k, v in delta.items():
                state[k] = v
            # Compute the elapsed time for THIS node from the timings delta
            new_timings = state.get("timings") or {}
            # Find the largest new key not in last_timings, or sum of new entries
            elapsed = 0
            for tkey, tval in new_timings.items():
                prev = last_timings.get(tkey, 0)
                if tval > prev:
                    elapsed = max(elapsed, tval - prev)
            last_timings = dict(new_timings)
            yield {
                "type": "stage_complete",
                "stage": node_name,
                "delta_summary": _summarize_delta(node_name, delta, state),
                "elapsed_ms": elapsed,
            }

    if audit:
        try:
            write_query_log(state)
        except Exception as e:  # pragma: no cover
            log.warning("audit write in streaming run failed: %s", e)

    yield {"type": "done", "state": _serialize_state_for_client(state)}


def _summarize_delta(node_name: str, delta: dict, state: dict) -> dict:
    """Per-stage minimal summary the UI can render without the full state."""
    if node_name == "classify":
        return {
            "classification": state.get("classification"),
            "confidence": state.get("classification_confidence"),
        }
    if node_name == "decompose":
        sq = state.get("sub_queries") or []
        return {"n_sub_queries": len(sq), "sub_queries": sq}
    if node_name in ("retrieve_single", "retrieve_parallel"):
        chunks = state.get("retrieved_chunks") or []
        return {
            "n_chunks": len(chunks),
            "top_cosine": state.get("top_cosine_sim"),
            "refusal_emitted": state.get("refusal_emitted", False),
        }
    if node_name == "synthesize":
        return {
            "draft_length": len(state.get("draft_answer") or ""),
            "regeneration_count": state.get("regeneration_count", 0),
            "refusal_emitted": state.get("refusal_emitted", False),
        }
    if node_name == "verify":
        return {
            "citations_stripped": state.get("citations_stripped", 0),
            "regeneration_count": state.get("regeneration_count", 0),
        }
    return {k: v for k, v in delta.items() if k != "verification_result"}


def _serialize_state_for_client(state: dict) -> dict:
    """Strip non-serializable / oversized fields (verification_result dataclass,
    raw prompts) for the client payload. The audit log already has the full state."""
    return {
        "classification": state.get("classification"),
        "classification_confidence": state.get("classification_confidence"),
        "sub_queries": state.get("sub_queries"),
        "retrieved_chunks": [
            {
                "chunk_id": c.get("chunk_id"),
                "accession_number": c.get("accession_number"),
                "section_heading": c.get("section_heading"),
                "paragraph_range": c.get("paragraph_range"),
                "chunk_text_preview": (c.get("chunk_text") or "")[:300],
                "rrf_score": c.get("rrf_score"),
                "cosine_sim": c.get("cosine_sim"),
            }
            for c in (state.get("retrieved_chunks") or [])
        ],
        "final_answer": state.get("final_answer") or "",
        "refusal_emitted": bool(state.get("refusal_emitted")),
        "refusal_reason": state.get("refusal_reason"),
        "citations_stripped": int(state.get("citations_stripped") or 0),
        "regeneration_count": int(state.get("regeneration_count") or 0),
        "timings_ms": state.get("timings") or {},
    }
