"""Tests for structured JSON logging + the shared query_id (issue #7).

Covers: the JSON formatter (valid JSON, extra fields, text-mode toggle), the
query_id minted at the handler and threaded into the graph / audit row / client
response, and the single `query_completed` event — asserting throughout that NO
raw query / answer / prompt / chunk text ever reaches a log line (issues #7/#9).
"""

from __future__ import annotations

import json
import logging
import uuid

import pytest
from fastapi.testclient import TestClient

from regrag_api import server
from regrag_api.audit.log import write_query_log
from regrag_api.logging_config import JsonFormatter, configure_logging
from regrag_api.orchestration.graph import _log_query_completed
from regrag_api.orchestration.state import initial_state


@pytest.fixture
def client() -> TestClient:
    return TestClient(server.app)


def _record(msg: str = "hello") -> logging.LogRecord:
    return logging.LogRecord("test.logger", logging.INFO, "f.py", 1, msg, (), None)


# ─── formatter ───────────────────────────────────────────────────────


def test_json_formatter_emits_valid_json():
    parsed = json.loads(JsonFormatter().format(_record("hello")))
    assert parsed["msg"] == "hello"
    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "test.logger"
    assert "ts" in parsed  # ISO-8601 timestamp present


def test_json_formatter_includes_extra_fields():
    rec = _record("chat_request")
    rec.query_id = "abc-123"
    rec.pii_kinds = ["email"]
    rec.query_len = 42
    parsed = json.loads(JsonFormatter().format(rec))
    assert parsed["query_id"] == "abc-123"
    assert parsed["pii_kinds"] == ["email"]
    assert parsed["query_len"] == 42


def test_configure_logging_text_vs_json(monkeypatch):
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    try:
        monkeypatch.setenv("REGRAG_LOG_FORMAT", "text")
        configure_logging()
        fmt = root.handlers[0].formatter
        assert not isinstance(fmt, JsonFormatter)      # human-readable mode
        assert isinstance(fmt, logging.Formatter)

        monkeypatch.setenv("REGRAG_LOG_FORMAT", "json")
        configure_logging()
        assert isinstance(root.handlers[0].formatter, JsonFormatter)  # default mode
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)


# ─── query_id threading ──────────────────────────────────────────────


def test_initial_state_preserves_query_id():
    st = initial_state("What does Order 841 require?", query_id="qid-supplied")
    assert st["query_id"] == "qid-supplied"
    # When not supplied, a valid uuid4 is minted.
    auto = initial_state("What does Order 841 require?")["query_id"]
    assert uuid.UUID(auto)  # parses → valid uuid string


def test_write_query_log_uses_state_query_id():
    captured: dict = {}

    class _FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, sql, params):
            captured["params"] = params

    class _FakeConn:
        def cursor(self):
            return _FakeCursor()

        def commit(self):
            captured["committed"] = True

    qid = "11111111-1111-1111-1111-111111111111"
    state = {"query_id": qid, "query": "Order 841?", "timings": {}}
    returned = write_query_log(state, conn=_FakeConn())  # type: ignore[arg-type]
    assert returned == qid
    assert str(captured["params"]["query_id"]) == qid  # state id, not a fresh uuid


# ─── query_completed event ───────────────────────────────────────────


def _completed_state() -> dict:
    return {
        "query_id": "qid-123",
        "user_id": "eval-runner",
        "classification": "single_doc",
        "refusal_emitted": False,
        "refusal_reason": None,
        "regeneration_count": 0,
        "retrieved_chunks": [{"chunk_id": "x"}],
        "timings": {"classify": 10, "synthesize": 50},
        "token_counts": {"synthesize": {"in": 100, "out": 20}},
        # sensitive text that must NEVER appear in the log line:
        "query": "SENSITIVE_QUERY_TEXT about Order 841",
        "draft_answer": "SENSITIVE_DRAFT_TEXT",
        "final_answer": "SENSITIVE_ANSWER_TEXT",
    }


def test_query_completed_event_has_query_id_and_metadata(caplog):
    caplog.set_level(logging.INFO)
    _log_query_completed(_completed_state())
    rec = next(r for r in caplog.records if r.getMessage() == "query_completed")
    assert rec.query_id == "qid-123"
    assert rec.user_id == "eval-runner"
    assert rec.classification == "single_doc"
    assert rec.refusal_emitted is False
    assert rec.n_chunks == 1
    assert rec.latency_ms_total == 60
    assert rec.token_totals == {"in": 100, "out": 20}


def test_query_completed_event_has_no_raw_text(caplog):
    caplog.set_level(logging.INFO)
    _log_query_completed(_completed_state())
    rec = next(r for r in caplog.records if r.getMessage() == "query_completed")
    blob = JsonFormatter().format(rec)
    assert "SENSITIVE_QUERY_TEXT" not in blob
    assert "SENSITIVE_DRAFT_TEXT" not in blob
    assert "SENSITIVE_ANSWER_TEXT" not in blob


# ─── handler-level logs + client exposure ────────────────────────────


def test_pii_blocked_log_has_query_id_not_text(client, caplog):
    caplog.set_level(logging.INFO)
    r = client.post("/chat", json={"query": "email me at secret@xyz.com re Order 841"})
    assert r.status_code == 200
    rec = next(r for r in caplog.records if r.getMessage() == "pii_blocked")
    assert rec.path == "/chat"
    assert isinstance(rec.query_id, str) and rec.query_id
    assert "email" in rec.pii_kinds
    assert "secret@xyz.com" not in JsonFormatter().format(rec)  # no query text in the log
    assert r.json()["query_id"] == rec.query_id                 # same id surfaced to client


def test_chat_response_includes_query_id(monkeypatch, client):
    def _fake_graph(q, *, user_id=None, query_id=None, **k):
        # The handler minted query_id and passed it in; echo it back via the state.
        return {"query_id": query_id, "final_answer": "ok", "retrieved_chunks": [],
                "classification": "single_doc", "refusal_emitted": False, "timings": {}}

    monkeypatch.setattr(server, "run_graph", _fake_graph)
    r = client.post("/chat", json={"query": "What does Order 841 require?"})
    assert r.status_code == 200
    qid = r.json()["query_id"]
    assert isinstance(qid, str) and uuid.UUID(qid)  # present and a valid uuid


def test_chat_stream_done_state_includes_query_id(monkeypatch, client):
    captured: dict = {}

    def _fake_stream(q, *, user_id=None, query_id=None, **k):
        captured["query_id"] = query_id
        yield {"type": "started", "query": ""}
        yield {"type": "done", "state": {"query_id": query_id, "final_answer": "ok",
                                         "refusal_emitted": False}}

    monkeypatch.setattr(server, "run_streaming", _fake_stream)
    r = client.post("/chat/stream", json={"query": "What does Order 2222 require?"})
    assert r.status_code == 200
    assert captured["query_id"]                 # handler passed a query_id into the graph
    assert captured["query_id"] in r.text       # and it reaches the client's done.state
