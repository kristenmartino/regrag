"""Stage 2: fetch PDFs from manifest URLs into corpus/raw/.

Idempotent — skips URLs already on disk with matching content hash.
Polite scraping: 1.5s delay between requests, identifying User-Agent.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from .manifest import ManifestEntry

USER_AGENT = "RegRAG-research/0.1 (krissi889@gmail.com)"
INTER_REQUEST_DELAY_S = 1.5
HTTP_TIMEOUT_S = 60

log = logging.getLogger(__name__)


@dataclass
class FetchResult:
    entry: ManifestEntry
    raw_path: Path
    content_hash: str
    status: str  # "fetched", "cached", "size_mismatch_refetched"
    bytes_fetched: int


def fetch_all(entries: list[ManifestEntry], raw_dir: Path) -> list[FetchResult]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    results: list[FetchResult] = []
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    for i, entry in enumerate(entries):
        if i > 0:
            time.sleep(INTER_REQUEST_DELAY_S)
        results.append(_fetch_one(entry, raw_dir, session))
    return results


def _fetch_one(entry: ManifestEntry, raw_dir: Path, session: requests.Session) -> FetchResult:
    target = raw_dir / f"{entry.slug}.pdf"

    if target.exists():
        existing_hash = _hash_file(target)
        existing_size = target.stat().st_size
        if entry.pdf_size_bytes is None or existing_size == entry.pdf_size_bytes:
            log.info("cached: %s (%d bytes)", entry.slug, existing_size)
            return FetchResult(entry, target, existing_hash, "cached", 0)
        log.warning(
            "%s size mismatch (disk=%d, manifest=%d) — refetching",
            entry.slug, existing_size, entry.pdf_size_bytes,
        )

    log.info("fetching: %s ← %s", entry.slug, entry.pdf_url)
    resp = session.get(entry.pdf_url, timeout=HTTP_TIMEOUT_S, stream=True)
    resp.raise_for_status()
    content_type = resp.headers.get("content-type", "")
    if "pdf" not in content_type.lower():
        raise ValueError(f"{entry.pdf_url}: expected application/pdf, got {content_type}")
    body = resp.content  # tolerable at corpus size; no doc > 5 MB
    target.write_bytes(body)
    h = hashlib.sha256(body).hexdigest()
    status = "size_mismatch_refetched" if target.exists() and entry.pdf_size_bytes else "fetched"
    return FetchResult(entry, target, h, status, len(body))


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
