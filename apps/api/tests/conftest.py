"""Shared pytest fixtures for the API test suite."""

from __future__ import annotations

import pytest

from regrag_api import rate_limit


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Isolate each test from the in-memory sliding-window rate limiter.

    The limiter (rate_limit.py) keys on client IP; under TestClient every request
    shares the 'testclient' identity, so without a reset the per-minute chat budget
    (6/min) bleeds across tests — enough chat POSTs in one 60s window trip a spurious
    429 in whichever test happens to run last. Clearing the deques before each test
    makes order and count irrelevant; within-test accumulation (e.g. a test that
    deliberately exercises the 429 path) is unaffected.
    """
    with rate_limit._lock:
        rate_limit._chat_hits.clear()
        rate_limit._audit_hits.clear()
    yield
