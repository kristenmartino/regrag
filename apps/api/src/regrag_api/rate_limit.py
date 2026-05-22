"""Lightweight in-memory IP rate limiter for the public chat endpoints.

Why: the demo is public and unauthenticated. Without a limit, anyone with the
URL can hammer /chat and burn through the Anthropic spend cap in minutes.
With a budget cap that's "only" $25/mo, this is real exposure for a free
demo URL shared on LinkedIn or a portfolio site.

This module is intentionally simple:
  - Sliding window (deque of timestamps per IP)
  - In-memory only (one Railway container, no Redis needed)
  - X-Forwarded-For respected when present (Railway sets this)
  - Two tiers: chat endpoints are tighter than audit endpoints

For a production multi-replica deployment, swap this for slowapi/Redis or
an upstream limit at the CDN.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import defaultdict, deque
from typing import Callable

from fastapi import HTTPException, Request

log = logging.getLogger(__name__)

# Per-IP rate limits (requests per window). Can be overridden via env vars
# for testing or to relax/tighten without a deploy.
CHAT_PER_MIN = int(os.environ.get("RATELIMIT_CHAT_PER_MIN", "6"))
CHAT_PER_HOUR = int(os.environ.get("RATELIMIT_CHAT_PER_HOUR", "30"))
AUDIT_PER_MIN = int(os.environ.get("RATELIMIT_AUDIT_PER_MIN", "30"))

# State: ip → deque of recent request timestamps. Trimmed lazily on each
# request. Bounded by max-window length so memory stays small.
_lock = threading.Lock()
_chat_hits: dict[str, deque[float]] = defaultdict(deque)
_audit_hits: dict[str, deque[float]] = defaultdict(deque)


def _client_ip(request: Request) -> str:
    """Best-effort client IP. Railway sets X-Forwarded-For with the real client
    IP first in the comma-separated list. If proxied through Vercel or a
    similar edge, the first entry is the original client."""
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _check(hits: dict[str, deque[float]], ip: str, per_min: int, per_hour: int | None) -> tuple[bool, str]:
    """Return (allowed, reason). Trims old entries as a side effect."""
    now = time.time()
    with _lock:
        q = hits[ip]
        # Trim entries older than the longest window
        cutoff = now - 3600
        while q and q[0] < cutoff:
            q.popleft()
        # Count windows
        last_min = sum(1 for t in q if t >= now - 60)
        last_hour = len(q)
        if last_min >= per_min:
            return False, f"rate limit: {per_min}/min exceeded ({last_min} in last 60s)"
        if per_hour is not None and last_hour >= per_hour:
            return False, f"rate limit: {per_hour}/hr exceeded ({last_hour} in last 3600s)"
        q.append(now)
        return True, "ok"


def chat_limit(request: Request) -> None:
    """FastAPI dependency: raise 429 if the calling IP exceeds chat limits."""
    ip = _client_ip(request)
    allowed, reason = _check(_chat_hits, ip, CHAT_PER_MIN, CHAT_PER_HOUR)
    if not allowed:
        log.warning("rate limit hit on /chat from %s: %s", ip, reason)
        raise HTTPException(
            status_code=429,
            detail=f"Too many requests. {reason}. The demo allows ~{CHAT_PER_MIN}/min, ~{CHAT_PER_HOUR}/hr per IP. Wait a minute or try again later.",
        )


def audit_limit(request: Request) -> None:
    """FastAPI dependency: raise 429 if the calling IP exceeds audit limits."""
    ip = _client_ip(request)
    allowed, reason = _check(_audit_hits, ip, AUDIT_PER_MIN, None)
    if not allowed:
        log.warning("rate limit hit on /audit from %s: %s", ip, reason)
        raise HTTPException(status_code=429, detail=f"Too many requests. {reason}.")
