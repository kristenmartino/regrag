"""Load and validate corpus/manifest.yaml entries."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ManifestEntry:
    pdf_url: str
    title: str
    document_type: str
    issue_date: str
    docket_numbers: list[str] = field(default_factory=list)
    order_number: str | None = None
    accession_number: str | None = None
    ferc_cite: str | None = None
    pdf_size_bytes: int | None = None
    notes: str | None = None

    @property
    def slug(self) -> str:
        """Filename-safe identifier for raw/parsed artifacts. Prefers
        accession_number, falls back to order_number, then a URL hash."""
        if self.accession_number:
            return self.accession_number
        if self.order_number:
            return f"order-{self.order_number}"
        from hashlib import sha1
        return f"url-{sha1(self.pdf_url.encode()).hexdigest()[:12]}"


def load_manifest(path: Path) -> list[ManifestEntry]:
    """Read corpus/manifest.yaml and return the verified `documents:` block.
    Ignores `todo_documents:` (entries without verified pdf_urls)."""
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or "documents" not in raw:
        raise ValueError(f"{path} missing top-level `documents:` key")

    entries: list[ManifestEntry] = []
    for i, item in enumerate(raw["documents"]):
        try:
            entries.append(_parse_entry(item))
        except Exception as e:
            raise ValueError(f"manifest entry {i}: {e}") from e
    return entries


def _parse_entry(item: dict[str, Any]) -> ManifestEntry:
    required = ["pdf_url", "title", "document_type", "issue_date"]
    missing = [k for k in required if not item.get(k)]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    return ManifestEntry(
        pdf_url=item["pdf_url"],
        title=item["title"],
        document_type=item["document_type"],
        issue_date=item["issue_date"],
        docket_numbers=item.get("docket_numbers") or [],
        order_number=str(item["order_number"]) if item.get("order_number") else None,
        accession_number=item.get("accession_number"),
        ferc_cite=item.get("ferc_cite"),
        pdf_size_bytes=item.get("pdf_size_bytes"),
        notes=item.get("notes"),
    )
