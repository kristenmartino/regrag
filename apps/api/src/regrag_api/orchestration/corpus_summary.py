"""Corpus summary injected into the decomposer's system prompt.

Per docs/implementation-plan.md §2.6 / critique.md H5: the decomposer needs
to know what's in the corpus before it can produce per-document sub-queries.
This is a static summary loaded from corpus/manifest.yaml at startup; refresh
when the manifest changes (e.g. as part of the ingest cron).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml


@lru_cache(maxsize=1)
def load_corpus_summary(manifest_path: Path | None = None) -> str:
    """Build a one-line-per-document summary of the corpus from manifest.yaml."""
    if manifest_path is None:
        # default: ~/regrag/corpus/manifest.yaml relative to this file.
        # parents: [0]=orchestration [1]=regrag_api [2]=src [3]=api [4]=apps [5]=repo_root
        repo_root = Path(__file__).resolve().parents[5]
        manifest_path = repo_root / "corpus" / "manifest.yaml"
    raw = yaml.safe_load(manifest_path.read_text())
    docs = raw.get("documents") or []
    lines: list[str] = []
    for d in docs:
        order = d.get("order_number")
        dockets = ", ".join(d.get("docket_numbers") or [])
        date = d.get("issue_date")
        title = d.get("title", "(untitled)")
        # truncate long titles
        if len(title) > 90:
            title = title[:87] + "..."
        if order:
            lines.append(f"- Order {order} ({dockets}, {date}): {title}")
        else:
            doc_type = d.get("document_type", "doc")
            lines.append(f"- [{doc_type}] {dockets} ({date}): {title}")
    return "Corpus contents (each is a single document the system can retrieve):\n" + "\n".join(lines)
