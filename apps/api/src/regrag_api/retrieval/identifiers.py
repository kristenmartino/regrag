"""Extract FERC regulatory identifiers from a query string.

Used by hybrid retrieval to (a) build identifier-aware tsquery terms and
(b) implement the "recall floor" — chunks that contain an exact identifier
from the query are guaranteed to appear in results regardless of vector or
keyword scores.

Patterns drawn from observed FERC PDF content (see Day 1 spike findings):
  - Order numbers: "Order 2222", "Order No. 2222", "Order No 841", with
    optional A/B/etc suffix for rehearing orders.
  - Docket numbers: "RM18-9-000", "AD16-20", "EL21-103-001", "ER22-962",
    case-insensitive prefix.
  - FERC citations: "162 FERC ¶ 61,127" — Volume FERC ¶ Page format.
  - U.S.C. citations: "16 U.S.C. § 824".
  - CFR citations: "18 CFR Part 35", "18 CFR § 35.28(g)".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Matches "Order 2222", "Orders 2222 and 841", "Orders Nos. 2222, 841, and 745".
# Captures the entire run of numbers after the prefix; ORDER_NUM_RE then splits.
ORDER_RE = re.compile(
    r"\bOrders?\s+(?:Nos?\.?\s+)?"
    r"(\d{2,4}(?:-?[A-Z])?"
    r"(?:[\s,]+(?:and\s+|or\s+)?\d{2,4}(?:-?[A-Z])?)*)",
    re.IGNORECASE,
)
ORDER_NUM_RE = re.compile(r"\d{2,4}(?:-?[A-Z])?")
DOCKET_RE = re.compile(r"\b([A-Z]{2}\d{2}-\d{1,3}(?:-\d{3})?)\b")
FERC_CITE_RE = re.compile(r"\b(\d{1,3})\s+FERC\s+[¶P]\s+(\d{2},\d{3})\b")
USC_CITE_RE = re.compile(r"\b(\d{1,2})\s+U\.S\.C\.?\s+§§?\s*(\d+(?:\([a-z0-9]+\))*)", re.IGNORECASE)
CFR_CITE_RE = re.compile(r"\b(\d{1,2})\s+CFR\s+(?:Part\s+|§§?\s*)(\d+(?:\.\d+)?(?:\([a-z0-9]+\))*)", re.IGNORECASE)


@dataclass
class ExtractedIdentifiers:
    orders: list[str]            # ["2222", "841"]
    dockets: list[str]           # ["RM18-9-000"]
    ferc_cites: list[str]        # ["162 FERC ¶ 61,127"]
    usc_cites: list[str]         # ["16 U.S.C. § 824"]
    cfr_cites: list[str]         # ["18 CFR Part 35"]

    @property
    def all_terms(self) -> list[str]:
        """Flattened list of all identifier strings, suitable for tsquery boost."""
        return self.orders + self.dockets + self.ferc_cites + self.usc_cites + self.cfr_cites

    @property
    def is_empty(self) -> bool:
        return not any([self.orders, self.dockets, self.ferc_cites, self.usc_cites, self.cfr_cites])


def extract_identifiers(text: str) -> ExtractedIdentifiers:
    """Pull all FERC regulatory identifiers from a query or chunk."""
    order_runs = (m.group(1) for m in ORDER_RE.finditer(text))
    orders = _dedupe_preserving_order(
        n for run in order_runs for n in ORDER_NUM_RE.findall(run)
    )
    return ExtractedIdentifiers(
        orders=orders,
        dockets=_dedupe_preserving_order(m.group(1).upper() for m in DOCKET_RE.finditer(text)),
        ferc_cites=_dedupe_preserving_order(
            f"{m.group(1)} FERC ¶ {m.group(2)}" for m in FERC_CITE_RE.finditer(text)
        ),
        usc_cites=_dedupe_preserving_order(
            f"{m.group(1)} U.S.C. § {m.group(2)}" for m in USC_CITE_RE.finditer(text)
        ),
        cfr_cites=_dedupe_preserving_order(
            f"{m.group(1)} CFR {m.group(2)}" for m in CFR_CITE_RE.finditer(text)
        ),
    )


def _dedupe_preserving_order(items) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
