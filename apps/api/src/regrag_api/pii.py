"""Structured-PII detection for the public query path (issue #9).

High-precision detectors for email, US phone, SSN, and Luhn-valid credit-card
numbers. Deliberately narrow and tuned NOT to flag FERC regulatory identifiers
(docket numbers like RM18-9-000, order numbers, FERC/USC/CFR citations, accession
numbers like 20180228-3066 / fr-2018-03-06-2018-03708) — those are the corpus's
bread and butter.

This is NOT a semantic confidential-content classifier: arbitrary confidential
prose (a client name, a deal) stays the user's responsibility (the UI disclaimer).
#9 blocks only the structured patterns above — refuse-and-don't-store.
"""

from __future__ import annotations

import re

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
# US phone: optional +1, a 3-digit area code (parenthesized or not), then 3 + 4
# digits, separated by space / dot / hyphen. Bounded by non-digits so it can't
# latch onto a longer numeric string (an accession or docket number).
PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?1[\s.\-]?)?(?:\(\d{3}\)\s?|\d{3}[\s.\-])\d{3}[\s.\-]\d{4}(?!\d)"
)
# SSN: 3-2-4 with hyphens, not embedded in a longer digit run.
SSN_RE = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
# Credit-card candidate: 13-16 digits, optionally grouped by single spaces/hyphens,
# not adjacent to a word char or hyphen (so 'fr-2018-...' / 'RM18-...' can't start a
# match). The Luhn check below is what makes this high-precision.
_CARD_CANDIDATE_RE = re.compile(r"(?<![\w-])\d(?:[ -]?\d){12,15}(?![\w-])")


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = ord(ch) - 48
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _has_credit_card(text: str) -> bool:
    for m in _CARD_CANDIDATE_RE.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if 13 <= len(digits) <= 16 and _luhn_ok(digits):
            return True
    return False


def detect_pii(text: str) -> list[str]:
    """Return the kinds of structured PII found in `text` (e.g. ['email', 'phone']),
    or [] if none. Used by the /chat handlers to refuse-and-don't-store (#9)."""
    if not text:
        return []
    kinds: list[str] = []
    if EMAIL_RE.search(text):
        kinds.append("email")
    if PHONE_RE.search(text):
        kinds.append("phone")
    if SSN_RE.search(text):
        kinds.append("ssn")
    if _has_credit_card(text):
        kinds.append("credit_card")
    return kinds


def contains_pii(text: str) -> bool:
    return bool(detect_pii(text))
