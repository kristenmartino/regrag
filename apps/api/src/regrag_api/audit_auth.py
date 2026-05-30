"""Authentication gate for the /audit endpoints (issue #11).

The audit log exposes, for every /chat call, the raw user query, the prompt sent
to the models, the full model response, and the retrieved corpus chunks. On a
public demo URL that is real exposure: rate limiting (``rate_limit.py``) throttles
*volume* but does nothing to stop a single visitor from reading the whole trail.

This module gates both audit routes behind an admin bearer token:

  - ``REGRAG_AUDIT_TOKEN`` unset/empty  → audit is DISABLED (default-deny). Every
    request gets 403, regardless of credentials, so a fresh deploy that forgets to
    set the token exposes nothing.
  - ``REGRAG_AUDIT_TOKEN`` set          → the caller must present the same token as
    ``Authorization: Bearer <token>`` (or ``X-Audit-Token: <token>``). Missing or
    wrong → 401.

The token is read at call time (not import time) so it can be rotated without a
code change and so tests can set it per-case. Comparison is constant-time to
avoid leaking the token through response timing.
"""

from __future__ import annotations

import hmac
import logging
import os

from fastapi import HTTPException, Request

log = logging.getLogger(__name__)

AUDIT_TOKEN_ENV = "REGRAG_AUDIT_TOKEN"


def _configured_token() -> str:
    """The admin token this deployment requires, or '' if audit is disabled."""
    return os.environ.get(AUDIT_TOKEN_ENV, "").strip()


def _presented_token(request: Request) -> str | None:
    """Pull the caller's token from the Authorization or X-Audit-Token header."""
    auth = request.headers.get("authorization", "")
    if auth[:7].lower() == "bearer ":
        token = auth[7:].strip()
        if token:
            return token
    x_token = request.headers.get("x-audit-token", "").strip()
    return x_token or None


def audit_auth(request: Request) -> None:
    """FastAPI dependency that gates the /audit endpoints.

    Raises 403 if audit is disabled (no token configured) and 401 if the caller
    did not present the correct token. Returns ``None`` (allowing the request)
    only on a constant-time token match.
    """
    configured = _configured_token()
    if not configured:
        # Default-deny: with no token configured the audit log is off entirely.
        raise HTTPException(
            status_code=403,
            detail="Audit endpoints are disabled. Set REGRAG_AUDIT_TOKEN to enable them.",
        )
    presented = _presented_token(request)
    if presented is None or not hmac.compare_digest(presented, configured):
        log.warning("rejected unauthorized /audit request (path=%s)", request.url.path)
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid audit token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
