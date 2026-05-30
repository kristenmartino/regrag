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

    def as_dict(self) -> dict[str, list[str]]:
        """Serialize to a plain dict for GraphState (issue #13). Audit-serializable."""
        return {
            "orders": self.orders,
            "dockets": self.dockets,
            "ferc_cites": self.ferc_cites,
            "usc_cites": self.usc_cites,
            "cfr_cites": self.cfr_cites,
        }


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


def identifier_terms(identifiers: dict[str, list[str]] | None) -> set[str]:
    """Flatten a serialized identifiers dict (ExtractedIdentifiers.as_dict()) to the
    set of all identifier terms across every class. None/empty → empty set. Used by
    retrieval to detect whether decomposition dropped any original-query identifier
    (issue #13)."""
    if not identifiers:
        return set()
    return {term for terms in identifiers.values() for term in terms}


# ─── Role semantics for anchored accessions (issue #14) ───
ROLE_PRIMARY = "primary"
ROLE_FEDERAL_REGISTER = "federal_register"
ROLE_REHEARING = "rehearing"

_REHEARING_SUFFIX_RE = re.compile(r"-[A-Za-z]+$")


def is_rehearing_order(order: str) -> bool:
    """True if the order id carries a rehearing suffix (e.g. '841-A', '2222-A')."""
    return bool(_REHEARING_SUFFIX_RE.search(order))


def build_anchored_roles(
    named_orders: list[str],
    role_map: dict[str, list[tuple[str, str]]],
) -> dict[str, dict[str, list[str]]]:
    """Group each named order's accessions by ROLE (issue #14).

    Args:
        named_orders: order numbers in play (e.g. ["841", "841-A"]).
        role_map: order_number → [(accession_number, document_type)] from the
            documents metadata.

    Returns {order: {"primary": [...], "federal_register": [...], "rehearing": [...]}}:
      primary           = final_rule docs with that exact order_number
      federal_register  = federal_register_publication docs with that order_number
      rehearing         = rehearing_order docs that ARE this order, PLUS the rehearings
                          OF this order (order_number startswith N + '-')
    """
    roles: dict[str, dict[str, list[str]]] = {}
    for n in named_orders:
        primary: list[str] = []
        federal_register: list[str] = []
        rehearing: list[str] = []
        for acc, doc_type in role_map.get(n, []):
            if doc_type == "final_rule":
                primary.append(acc)
            elif doc_type == "federal_register_publication":
                federal_register.append(acc)
            elif doc_type == "rehearing_order":
                rehearing.append(acc)
        # Rehearings OF this order (e.g. 841 → 841-A), folded in for context.
        for other, entries in role_map.items():
            if other != n and other.startswith(n + "-"):
                for acc, _doc_type in entries:
                    if acc not in rehearing:
                        rehearing.append(acc)
        roles[n] = {
            ROLE_PRIMARY: primary,
            ROLE_FEDERAL_REGISTER: federal_register,
            ROLE_REHEARING: rehearing,
        }
    return roles


def primary_citation_accessions(
    order: str, anchored_roles: dict[str, dict[str, list[str]]]
) -> list[str]:
    """The accessions allowed as the PRIMARY (first) citation for a sentence whose
    subject is `order` (issue #14): the rehearing accessions if `order` is itself a
    rehearing, otherwise the primary + Federal-Register accessions — NEVER the
    rehearing, which is a separate document."""
    role = anchored_roles.get(order) or {}
    if is_rehearing_order(order):
        return list(role.get(ROLE_REHEARING, []))
    return list(role.get(ROLE_PRIMARY, [])) + list(role.get(ROLE_FEDERAL_REGISTER, []))
