"""Tests for the PII gate on the query path (issue #9): refuse-and-don't-store.

A query with structured PII (email / US phone / SSN / Luhn-valid card) is refused
at the handler BEFORE any raw-query logging, the graph, or the audit write — so it
never reaches Anthropic or Neon. The detector must NOT flag FERC identifiers.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from regrag_api import server
from regrag_api.pii import _luhn_ok, detect_pii

# FERC regulatory identifiers that must never be mistaken for PII.
FERC_IDS = [
    "Order 841", "RM18-9-000", "RM16-23-000", "162 FERC ¶ 61,127",
    "18 CFR § 35.28(g)", "16 U.S.C. § 824", "20180228-3066", "fr-2018-03-06-2018-03708",
]


@pytest.fixture
def client() -> TestClient:
    return TestClient(server.app)


# ─── detector ────────────────────────────────────────────────────────


def test_detects_each_pii_kind():
    assert "email" in detect_pii("reach me at john.doe@example.com")
    assert "phone" in detect_pii("call 555-123-4567")
    assert "phone" in detect_pii("(202) 555-0173 thanks")
    assert "ssn" in detect_pii("ssn 123-45-6789")
    assert "credit_card" in detect_pii("card 4111 1111 1111 1111")  # Visa test, Luhn-valid
    assert "credit_card" in detect_pii("4111111111111111")


def test_ferc_identifiers_are_not_pii():
    for s in FERC_IDS:
        assert detect_pii(s) == [], f"false positive on {s!r}: {detect_pii(s)}"
    q = ("What did Order 841 (docket RM16-23-000, 162 FERC ¶ 61,127), effective per "
         "20180228-3066 and fr-2018-03-06-2018-03708, require under 18 CFR § 35.28(g)?")
    assert detect_pii(q) == []


def test_luhn_and_non_luhn_card():
    assert _luhn_ok("4111111111111111")
    assert not _luhn_ok("4111111111111112")
    # a 16-digit non-Luhn reference number is not flagged
    assert "credit_card" not in detect_pii("reference number 1234567812345678")


def test_empty_query():
    assert detect_pii("") == []


# ─── /chat gate ──────────────────────────────────────────────────────


def test_chat_pii_blocked_does_not_run_graph(monkeypatch, client):
    def _boom(*a, **k):
        raise AssertionError("run_graph must not be called for a PII query")

    monkeypatch.setattr(server, "run_graph", _boom)
    r = client.post("/chat", json={"query": "What does Order 841 require? email me at a@b.com"})
    assert r.status_code == 200
    body = r.json()
    assert body["refusal_emitted"] is True
    assert body["refusal_reason"] == "pii_blocked"
    assert body["retrieved_chunks"] == []
    assert "a@b.com" not in body["final_answer"]  # the PII is not echoed back


def test_chat_clean_query_runs_graph(monkeypatch, client):
    captured = {}

    def _fake_graph(q, **k):
        captured["q"] = q
        return {"final_answer": "Order 841 requires tariff revisions.",
                "retrieved_chunks": [], "classification": "single_doc",
                "refusal_emitted": False, "timings": {}}

    monkeypatch.setattr(server, "run_graph", _fake_graph)
    r = client.post("/chat", json={"query": "What does Order 841 require?"})
    assert r.status_code == 200
    assert r.json()["refusal_reason"] != "pii_blocked"
    assert captured["q"] == "What does Order 841 require?"


# ─── /chat/stream gate ───────────────────────────────────────────────


def test_chat_stream_pii_blocked_does_not_run_graph(monkeypatch, client):
    def _boom(*a, **k):
        raise AssertionError("run_streaming must not be called for a PII query")

    monkeypatch.setattr(server, "run_streaming", _boom)
    r = client.post("/chat/stream", json={"query": "my ssn is 123-45-6789"})
    assert r.status_code == 200
    body = r.text
    assert "pii_blocked" in body
    assert '"done"' in body          # delivered as a completed refusal
    assert '"error"' not in body     # NOT an error event (intentional safety refusal)
    assert "123-45-6789" not in body  # the PII is not echoed in any event


def test_chat_stream_clean_query_runs(monkeypatch, client):
    called = {"n": 0}

    def _fake_stream(q, **k):
        called["n"] += 1
        yield {"type": "started", "query": q}
        yield {"type": "done", "state": {"final_answer": "ok", "refusal_emitted": False}}

    monkeypatch.setattr(server, "run_streaming", _fake_stream)
    r = client.post("/chat/stream", json={"query": "What does Order 2222 require?"})
    assert r.status_code == 200
    assert called["n"] == 1
    assert "pii_blocked" not in r.text
