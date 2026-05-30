"""Tests for the /audit authentication gate (issue #11).

Three security-relevant outcomes:
  - audit disabled by default (no REGRAG_AUDIT_TOKEN)         → 403
  - token configured but caller presents none/wrong token    → 401
  - correct token                                            → passes the gate

The 401/403 cases never touch the database — the auth dependency runs before the
route body, so no Neon connection is attempted. The valid-token case stubs the DB
accessor so it asserts "auth passed" without a live connection.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from regrag_api import server
from regrag_api.audit_auth import AUDIT_TOKEN_ENV, _presented_token, audit_auth

TOKEN = "test-admin-token-123"
AUDIT_PATHS = ["/audit", "/audit/some-query-id"]


class _CIHeaders(dict):
    """Minimal case-insensitive headers, like Starlette's Headers.get."""

    def get(self, key, default=None):
        return super().get(key.lower(), default)


def _req(headers: dict | None = None) -> SimpleNamespace:
    h = _CIHeaders()
    for k, v in (headers or {}).items():
        h[k.lower()] = v
    return SimpleNamespace(headers=h, url=SimpleNamespace(path="/audit"))


@pytest.fixture
def client() -> TestClient:
    return TestClient(server.app)


# ─── unit tests on the dependency ──────────────────────────────────


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv(AUDIT_TOKEN_ENV, raising=False)
    with pytest.raises(HTTPException) as exc:
        audit_auth(_req())
    assert exc.value.status_code == 403


def test_disabled_ignores_presented_token(monkeypatch):
    # Even a well-formed token can't enable a deployment that configured none.
    monkeypatch.delenv(AUDIT_TOKEN_ENV, raising=False)
    with pytest.raises(HTTPException) as exc:
        audit_auth(_req({"Authorization": "Bearer anything"}))
    assert exc.value.status_code == 403


def test_missing_token_when_enabled(monkeypatch):
    monkeypatch.setenv(AUDIT_TOKEN_ENV, TOKEN)
    with pytest.raises(HTTPException) as exc:
        audit_auth(_req())
    assert exc.value.status_code == 401
    assert exc.value.headers.get("WWW-Authenticate") == "Bearer"


def test_wrong_token(monkeypatch):
    monkeypatch.setenv(AUDIT_TOKEN_ENV, TOKEN)
    with pytest.raises(HTTPException) as exc:
        audit_auth(_req({"Authorization": "Bearer nope"}))
    assert exc.value.status_code == 401


def test_correct_bearer_token_passes(monkeypatch):
    monkeypatch.setenv(AUDIT_TOKEN_ENV, TOKEN)
    assert audit_auth(_req({"Authorization": f"Bearer {TOKEN}"})) is None


def test_x_audit_token_header_accepted(monkeypatch):
    monkeypatch.setenv(AUDIT_TOKEN_ENV, TOKEN)
    assert audit_auth(_req({"X-Audit-Token": TOKEN})) is None


def test_presented_token_parsing():
    assert _presented_token(_req({"Authorization": "Bearer abc"})) == "abc"
    assert _presented_token(_req({"authorization": "bearer abc"})) == "abc"  # case-insensitive
    assert _presented_token(_req({"X-Audit-Token": "xyz"})) == "xyz"
    assert _presented_token(_req({})) is None
    assert _presented_token(_req({"Authorization": "Bearer    "})) is None  # blank after prefix


# ─── integration tests through the real routes ─────────────────────


@pytest.mark.parametrize("path", AUDIT_PATHS)
def test_route_disabled_returns_403(monkeypatch, client, path):
    monkeypatch.delenv(AUDIT_TOKEN_ENV, raising=False)
    assert client.get(path).status_code == 403


@pytest.mark.parametrize("path", AUDIT_PATHS)
def test_route_missing_token_returns_401(monkeypatch, client, path):
    monkeypatch.setenv(AUDIT_TOKEN_ENV, TOKEN)
    assert client.get(path).status_code == 401


@pytest.mark.parametrize("path", AUDIT_PATHS)
def test_route_wrong_token_returns_401(monkeypatch, client, path):
    monkeypatch.setenv(AUDIT_TOKEN_ENV, TOKEN)
    assert client.get(path, headers={"Authorization": "Bearer wrong"}).status_code == 401


@pytest.mark.parametrize("path", AUDIT_PATHS)
def test_route_valid_token_reaches_handler(monkeypatch, client, path):
    """A correct token passes the gate; we stub the DB accessor so reaching the
    handler surfaces a sentinel status (599) instead of a real Neon call. Any
    401/403 here would mean the gate wrongly rejected a valid token."""
    monkeypatch.setenv(AUDIT_TOKEN_ENV, TOKEN)

    def _sentinel():
        raise HTTPException(status_code=599, detail="reached handler")

    monkeypatch.setattr(server, "_get_audit_conn", _sentinel)
    r = client.get(path, headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.status_code == 599
