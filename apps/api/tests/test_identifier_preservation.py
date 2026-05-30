"""Tests for original-query identifier preservation (issue #13).

`initial_state()` captures the user's identifiers deterministically (model-free).
On the multi-doc path, `retrieve_parallel` re-adds the original query to the
retrieval set when decomposition dropped any of them, so anchoring/floor still
see them. No external calls — `hybrid_retrieve` is patched.
"""

from __future__ import annotations

from regrag_api.orchestration.nodes import retrieve as retrieve_mod
from regrag_api.orchestration.nodes.retrieve import retrieve_parallel, retrieve_single
from regrag_api.orchestration.state import initial_state
from regrag_api.retrieval.hybrid import RetrievedChunk


def _chunk(chunk_id: str, accession: str, *, anchored: bool, cosine: float = 0.6) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, accession_number=accession, section_heading=None,
        paragraph_range=None, chunk_text="text", parent_chunk_id=None,
        vector_rank=1, keyword_rank=None, anchored_rank=1 if anchored else None,
        cosine_sim=cosine, ts_rank=None, floor_match=False, anchored_match=anchored,
        rrf_score=0.5,
    )


def _patch_hybrid(monkeypatch, mapping=None, *, default=None):
    """Patch hybrid_retrieve to return canned chunks keyed by exact query string.
    Records the queries it was actually called with in the returned `calls` list."""
    mapping = mapping or {}
    calls: list[str] = []

    def fake(query, k=10, **kw):
        calls.append(query)
        return mapping.get(query, default or [])

    monkeypatch.setattr(retrieve_mod, "hybrid_retrieve", fake)
    return calls


# ─── 1. initial_state captures full original identifiers ─────────────


def test_initial_state_captures_orders_and_docket():
    s = initial_state("What was the Order 841 compliance deadline in docket RM16-23-000?")
    ids = s["original_identifiers"]
    assert ids["orders"] == ["841"]
    assert ids["dockets"] == ["RM16-23-000"]
    assert set(ids.keys()) == {"orders", "dockets", "ferc_cites", "usc_cites", "cfr_cites"}


def test_initial_state_no_identifiers():
    s = initial_state("How do capacity markets work?")
    assert s["original_identifiers"] == {
        "orders": [], "dockets": [], "ferc_cites": [], "usc_cites": [], "cfr_cites": [],
    }


# ─── 2/3. retrieve_parallel adds the original query when an identifier is dropped ─


def test_adds_original_query_when_order_dropped(monkeypatch):
    s = initial_state("What was the effective date of Order 841?")
    s["sub_queries"] = ["When did the wholesale storage rule take effect?"]  # drops 841
    orig = s["query"]
    calls = _patch_hybrid(monkeypatch, {
        orig: [_chunk("20180228-3066:c1", "20180228-3066", anchored=True)],
        s["sub_queries"][0]: [_chunk("other:c1", "other", anchored=False)],
    })
    out = retrieve_parallel(s)
    assert orig in calls                                  # original query was retrieved
    assert "841" in out["named_orders"]                   # order preserved into metadata
    # invariant: the anchored chunk from the original-query retrieval populates anchored_accessions
    assert "20180228-3066" in out["anchored_accessions"]


def test_adds_original_query_when_docket_dropped(monkeypatch):
    s = initial_state("What was the compliance deadline in docket RM16-23-000?")
    s["sub_queries"] = ["What was the compliance deadline?"]  # drops the docket
    orig = s["query"]
    calls = _patch_hybrid(monkeypatch, default=[_chunk("c:1", "acc", anchored=False)])
    retrieve_parallel(s)
    assert orig in calls


# ─── 4. does NOT add the original query when subqueries preserve identifiers ──


def test_does_not_add_original_when_preserved(monkeypatch):
    s = initial_state("Compare Orders 841 and 2222")
    s["sub_queries"] = ["What does Order 841 say?", "What does Order 2222 say?"]  # both preserved
    orig = s["query"]
    calls = _patch_hybrid(monkeypatch, default=[_chunk("c:1", "acc", anchored=False)])
    retrieve_parallel(s)
    assert orig not in calls
    assert calls == s["sub_queries"]


# ─── 5. effective named_orders = original ∪ subquery orders ──────────


def test_named_orders_unions_original_and_subquery(monkeypatch):
    s = initial_state("How did Order 841 evolve?")  # original names 841
    s["sub_queries"] = ["What does Order 841 say?", "What does Order 2222 say?"]  # introduces 2222
    _patch_hybrid(monkeypatch, default=[_chunk("c:1", "acc", anchored=False)])
    out = retrieve_parallel(s)
    assert out["named_orders"] == ["2222", "841"]  # sorted union


# ─── single-doc path is unaffected ───────────────────────────────────


def test_single_doc_does_not_augment(monkeypatch):
    s = initial_state("What does Order 841 require?")
    calls = _patch_hybrid(monkeypatch, default=[_chunk("c:1", "acc", anchored=False)])
    retrieve_single(s)
    assert calls == [s["query"]]  # only the original query; no duplication
