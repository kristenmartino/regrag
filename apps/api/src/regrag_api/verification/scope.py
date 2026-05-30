"""Accession-scope verifier — enforces per-sentence in-scope citation rule.

When the user names a specific order and the synthesize SCOPE block is firing,
the synthesizer SHOULD already prefer in-scope citations. This verifier is the
safety net: it post-checks the draft sentence by sentence and strips citations
that violate the rule, or rewrites/drops the sentence entirely.

The rule: for every sentence whose subject is one of the named orders, the
FIRST citation in that sentence must be from an accession in the in-scope set
(state.anchored_accessions). A second citation from a different accession as
supporting cross-reference is allowed; an out-of-scope FIRST citation is not.

Subject detection is heuristic — we check whether the sentence contains the
order number ("841", "Order No. 841", etc.). This misses pronoun subjects
("It was issued in 2018...") and implicit subjects, but it covers the common
failure pattern: model writes an explicit claim about Order N and cites the
wrong order. Pronoun cases are usually fine because the prior sentence with
the explicit subject already constrained the citation.

Output: a ScopeResult with stripped citations, an optional regen signal, and
a cleaned_text where out-of-scope FIRST citations have been removed (sentence
preserved with a SECOND citation if present, or dropped if only one citation
existed and it was out-of-scope).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from ..retrieval.identifiers import primary_citation_accessions

log = logging.getLogger(__name__)

# Match a citation marker and capture both the full citation and the accession
# prefix (everything before the `:cNNNN` suffix).
_CITATION_RE = re.compile(r"\[\[(?P<chunk_id>[^\]]+)\]\]")

# Threshold: if >= this fraction of in-subject sentences violate scope, the
# draft is unsalvageable by stripping — request regeneration instead.
REGEN_THRESHOLD_SCOPE_VIOLATION_FRAC = 0.5


@dataclass
class ScopeResult:
    """Outcome of accession-scope check on a draft."""
    cleaned_text: str
    sentences_checked: int = 0           # how many sentences had named-order subject
    sentences_violating: int = 0         # how many of those violated the rule
    citations_stripped: list[str] = field(default_factory=list)
    sentences_dropped: list[str] = field(default_factory=list)
    should_regenerate: bool = False
    notes: str = ""


def check_accession_scope(
    draft: str,
    named_orders: list[str],
    anchored_roles: dict[str, dict[str, list[str]]],
    *,
    regeneration_count: int = 0,
) -> ScopeResult:
    """Walk the draft sentence-by-sentence and enforce the role-aware in-scope rule.

    Args:
        draft: the verified-citations text from the previous verification step.
        named_orders: order numbers extracted from the query (e.g. ["841"]).
        anchored_roles: per-order accessions grouped by role (issue #14):
            {order: {"primary": [...], "federal_register": [...], "rehearing": [...]}}.
            The in-scope set is computed PER SENTENCE-SUBJECT — a primary-order subject's
            first citation must be a primary/Federal-Register accession (NOT a rehearing);
            a rehearing subject's must be the rehearing.
        regeneration_count: current regen attempt; used to decide regen vs strip.

    Returns:
        ScopeResult with cleaned_text and stripping/regen decisions.
    """
    if not named_orders or not anchored_roles:
        # Rule doesn't apply — nothing to check.
        return ScopeResult(cleaned_text=draft, notes="no named order in query")

    order_patterns = _build_order_patterns(named_orders)
    sentences = _split_sentences(draft)

    new_sentences: list[str] = []
    stripped_cites: list[str] = []
    dropped_sentences: list[str] = []
    checked = 0
    violating = 0

    for sent in sentences:
        # Which named order (if any) is this sentence's subject?
        subject = _subject_order(sent, order_patterns)
        if subject is None:
            new_sentences.append(sent)
            continue

        # Role-aware in-scope set for THIS subject (issue #14): a primary-order
        # subject's first citation must be a primary or Federal-Register source, not a
        # rehearing; a rehearing subject's must be the rehearing.
        in_scope = set(primary_citation_accessions(subject, anchored_roles))

        checked += 1
        cites = list(_CITATION_RE.finditer(sent))
        if not cites:
            # In-subject sentence with no citation — not a scope violation per se;
            # other verifiers handle uncited claims. Preserve.
            new_sentences.append(sent)
            continue

        first_cite = cites[0]
        first_acc = _accession_of(first_cite.group("chunk_id"))
        if first_acc in in_scope:
            # Compliant. Preserve as-is.
            new_sentences.append(sent)
            continue

        # FIRST citation is out-of-scope. Look for an in-scope citation later
        # in the same sentence: if one exists, strip the first and let the
        # in-scope one promote to primary. Otherwise the sentence has no
        # in-scope support and must be dropped.
        violating += 1
        in_scope_later = [c for c in cites[1:] if _accession_of(c.group("chunk_id")) in in_scope]
        if in_scope_later:
            stripped_cite_id = first_cite.group("chunk_id")
            stripped_cites.append(stripped_cite_id)
            # Strip the first marker; preserve rest of sentence
            new_sent = (
                sent[:first_cite.start()] + sent[first_cite.end():]
            )
            new_sent = _tidy_punctuation(new_sent)
            new_sentences.append(new_sent)
            log.info(
                "scope: stripped out-of-scope first cite [[%s]] from sentence; "
                "second in-scope cite promotes to primary",
                stripped_cite_id,
            )
        else:
            # No in-scope citation anywhere in the sentence — drop entirely.
            # Record all stripped citations from the dropped sentence.
            for c in cites:
                stripped_cites.append(c.group("chunk_id"))
            dropped_sentences.append(sent.strip()[:200])
            log.info(
                "scope: dropped sentence with no in-scope citation: %r",
                sent.strip()[:120],
            )

    cleaned = " ".join(s.strip() for s in new_sentences if s.strip())
    # Collapse multiple spaces from the join
    cleaned = re.sub(r" {2,}", " ", cleaned)

    violation_frac = violating / checked if checked else 0.0
    should_regen = (
        violation_frac >= REGEN_THRESHOLD_SCOPE_VIOLATION_FRAC
        and regeneration_count < 2
        and checked > 1  # don't regen on single-sentence drafts
    )

    notes = (
        f"checked {checked} in-subject sentences, "
        f"{violating} violated scope, "
        f"stripped {len(stripped_cites)} cites, "
        f"dropped {len(dropped_sentences)} sentences"
    )
    return ScopeResult(
        cleaned_text=cleaned,
        sentences_checked=checked,
        sentences_violating=violating,
        citations_stripped=stripped_cites,
        sentences_dropped=dropped_sentences,
        should_regenerate=should_regen,
        notes=notes,
    )


# ───────────── helpers ─────────────


def _accession_of(chunk_id: str) -> str:
    """`'20180228-3066:c0259'` → `'20180228-3066'`. Tolerates chunk_ids without
    a colon (returns the input)."""
    return chunk_id.split(":", 1)[0].strip()


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[])")


def _split_sentences(text: str) -> list[str]:
    """Crude sentence splitter — splits on `.!?` followed by whitespace +
    capital letter / digit / opening bracket. Good enough for prose with
    citation markers; not a linguistic parser.
    """
    # Preserve the trailing terminator on each sentence.
    return _SENTENCE_SPLIT_RE.split(text)


def _build_order_patterns(named_orders: list[str]) -> list[tuple[str, re.Pattern]]:
    """For each order number, build (order, regex) where the regex matches it as a
    token (not inside a longer number): 'Order 841', 'Order No. 841', '841' as a
    standalone, but not '8410'. The boundary lookarounds block '841' from matching
    inside '841-A', so primary and rehearing orders stay distinguishable.
    """
    patterns = []
    for o in named_orders:
        escaped = re.escape(o)
        patterns.append((o, re.compile(rf"(?<![A-Za-z0-9-]){escaped}(?![A-Za-z0-9-])")))
    return patterns


def _subject_order(sentence: str, order_patterns: list[tuple[str, re.Pattern]]) -> str | None:
    """Heuristic subject detection (issue #14): return the named order whose token
    appears EARLIEST within the first 80 characters (the subject area), or None.
    Returning WHICH order — not just whether one matched — lets the caller pick the
    role-appropriate in-scope set (primary/FR vs rehearing).

    Intentionally lenient: a false positive applies the rule where it wasn't strictly
    needed (soft-fail); a false negative (subject past char 80) skips the check, the
    pre-existing silent gap.
    """
    head = sentence[:80]
    best: str | None = None
    best_pos: int | None = None
    for order, pat in order_patterns:
        m = pat.search(head)
        if m and (best_pos is None or m.start() < best_pos):
            best, best_pos = order, m.start()
    return best


def _tidy_punctuation(text: str) -> str:
    """Strip leftover spaces before punctuation after removing a citation."""
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r" {2,}", " ", text)
    return text
