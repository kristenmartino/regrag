"""Corpus summary injected into the decomposer's system prompt.

Per docs/implementation-plan.md §2.6 / critique.md H5: the decomposer needs
to know what's in the corpus before it can produce per-document sub-queries.

Built by querying the `documents` table directly so the summary always
reflects what's actually loaded in pgvector. No file-system dependency on
manifest.yaml — works inside Docker, Vercel, etc. without bundling corpus
data into the build artifact. Cached for the process lifetime; restart the
service after corpus changes.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

import psycopg

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_corpus_summary() -> str:
    """Build a one-line-per-document summary by SELECT-ing from the documents table."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        log.warning("corpus_summary: DATABASE_URL not set, returning placeholder")
        return "(corpus summary unavailable — database not configured)"

    try:
        with psycopg.connect(url, connect_timeout=10) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT order_number, docket_numbers, document_type, issue_date, title
                    FROM documents
                    ORDER BY issue_date DESC
                    """
                )
                rows = cur.fetchall()
    except Exception as e:
        log.error("corpus_summary: query failed: %s", e)
        return "(corpus summary unavailable — database query failed)"

    lines: list[str] = []
    for order_number, dockets, doc_type, issue_date, title in rows:
        truncated = title[:87] + "..." if title and len(title) > 90 else (title or "(untitled)")
        dockets_str = ", ".join(dockets or [])
        if order_number:
            lines.append(f"- Order {order_number} ({dockets_str}, {issue_date}): {truncated}")
        else:
            lines.append(f"- [{doc_type or 'doc'}] {dockets_str} ({issue_date}): {truncated}")

    if not lines:
        return "(corpus is empty — no documents loaded)"

    return (
        "Corpus contents (each is a single document the system can retrieve):\n"
        + "\n".join(lines)
    )
