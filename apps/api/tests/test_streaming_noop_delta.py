"""Regression test for the /chat/stream crash on a no-op node delta.

Bug: run_streaming iterated `for k, v in delta.items()` over every node update
from `graph.stream(..., stream_mode="updates")`. langgraph surfaces a node that
returns {} or None as a *None* delta in that mode — and the answerability gate
(in every query's path) returns {} when its flag is off (the default). So every
streamed query raised `AttributeError: 'NoneType' object has no attribute
'items'`. graph.invoke() — the non-streaming /chat path — tolerates the same
returns, which is why only /chat/stream broke.

These tests pin the langgraph behavior (so a future upgrade that changes it is
caught) and assert run_streaming survives a no-op delta end-to-end.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from regrag_api.orchestration import graph as graph_mod
from regrag_api.orchestration.state import GraphState


def test_langgraph_updates_mode_yields_none_for_empty_return():
    """Pin the upstream behavior this fix defends against: in stream_mode=
    'updates', a node returning {} (or None) is reported as a None delta."""
    g = StateGraph(GraphState)
    g.add_node("data", lambda s: {"final_answer": "x"})
    g.add_node("noop", lambda s: {})  # like the answerability gate when off
    g.add_edge(START, "data")
    g.add_edge("data", "noop")
    g.add_edge("noop", END)
    app = g.compile()

    deltas = {node: delta for chunk in app.stream({"query": "q"}, stream_mode="updates")
              for node, delta in chunk.items()}
    assert deltas["data"] == {"final_answer": "x"}
    assert deltas["noop"] is None  # <-- the shape that crashed run_streaming


def _tiny_graph_with_noop_gate():
    """A compiled graph shaped like the real one: a no-op middle node (the
    disabled answerability gate) between two nodes that do return state."""
    g = StateGraph(GraphState)
    g.add_node("classify", lambda s: {"classification": "single_doc", "timings": {"classify": 5}})
    g.add_node("answerability_gate", lambda s: {})  # flag off → {} → None delta when streamed
    g.add_node("synthesize", lambda s: {"final_answer": "ok", "timings": {"classify": 5, "synthesize": 7}})
    g.add_edge(START, "classify")
    g.add_edge("classify", "answerability_gate")
    g.add_edge("answerability_gate", "synthesize")
    g.add_edge("synthesize", END)
    return g.compile()


def test_run_streaming_survives_noop_delta(monkeypatch):
    monkeypatch.setattr(graph_mod, "build_graph", _tiny_graph_with_noop_gate)

    # Before the fix this raised AttributeError: 'NoneType' object has no attribute 'items'.
    events = list(graph_mod.run_streaming("What does Order 2222 require?", audit=False))

    types = [e["type"] for e in events]
    assert types[0] == "started"
    assert types[-1] == "done"  # reached the end instead of crashing mid-stream

    # The no-op gate has no update to report and must not surface as a stage event;
    # the real, state-bearing nodes must.
    stages = [e["stage"] for e in events if e["type"] == "stage_complete"]
    assert "answerability_gate" not in stages
    assert {"classify", "synthesize"} <= set(stages)

    # State produced after the None delta still propagated into the final payload.
    assert events[-1]["state"]["final_answer"] == "ok"
