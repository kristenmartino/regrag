"""Quote verification — checks each claim's supporting_quote actually appears
in the chunk it's attributed to.

CURRENT STATUS: NOT WIRED INTO THE GRAPH (vestigial as of 2026-05-08).

This module was built for the structured-output synthesis experiment (commit
5600664, reverted in d186c99). The hypothesis: forcing the synthesizer to
emit (claim, chunk_id, supporting_quote) triples and then substring-checking
each quote against its chunk would beat the LLM-as-judge approach in
substantive.py — by construction, citation drift toward topical chunks
would become impossible.

The eval said otherwise. See docs/eval-results.md for the full v4 writeup.
Briefly: structured-output + quote verification produced CF=84.1% (vs the
LLM judge's 91.4%) because the substring check is too permissive — it
accepts any quote that appears anywhere in the chunk, even if the claim
paraphrases beyond what the quote supports. The Sonnet judge in
substantive.py understands "the quote is real but the claim overreaches"
and the substring check does not.

Module kept for historical reference and as the foundation for any future
attempt to combine structural + LLM verification (e.g. quote check as a
fast pre-filter before the LLM judge runs).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

log = logging.getLogger(__name__)

# Minimum length for a quote to be meaningful — too-short quotes (a single word)
# can match coincidentally. 20 chars / ~3-4 words is the cutoff.
MIN_QUOTE_LEN = 20

# Fuzzy ratio threshold for paraphrased / OCR-mangled quotes. 0.85 = quote text
# matches at least 85% of characters in the closest chunk substring of equal length.
FUZZY_RATIO_THRESHOLD = 0.85


@dataclass
class StructuredClaim:
    """One (claim, chunk_id, supporting_quote) triple from the synthesizer."""
    claim: str
    chunk_id: str
    supporting_quote: str


@dataclass
class QuoteVerificationResult:
    kept_claims: list[StructuredClaim]
    dropped_for_unknown_chunk: list[StructuredClaim]
    dropped_for_quote_not_found: list[StructuredClaim]
    dropped_for_short_quote: list[StructuredClaim]

    @property
    def total_input(self) -> int:
        return (
            len(self.kept_claims)
            + len(self.dropped_for_unknown_chunk)
            + len(self.dropped_for_quote_not_found)
            + len(self.dropped_for_short_quote)
        )

    @property
    def keep_ratio(self) -> float:
        return len(self.kept_claims) / max(1, self.total_input)


def verify_claim_quotes(
    claims: list[StructuredClaim],
    retrieved_chunks: list[dict],
) -> QuoteVerificationResult:
    """Verify each claim's quote against its cited chunk. Returns kept + dropped lists."""
    chunk_lookup = {c["chunk_id"]: (c.get("chunk_text") or "") for c in retrieved_chunks}

    kept: list[StructuredClaim] = []
    bad_chunk: list[StructuredClaim] = []
    bad_quote: list[StructuredClaim] = []
    short_quote: list[StructuredClaim] = []

    for claim in claims:
        chunk_text = chunk_lookup.get(claim.chunk_id)
        if chunk_text is None:
            bad_chunk.append(claim)
            continue
        if len(claim.supporting_quote.strip()) < MIN_QUOTE_LEN:
            short_quote.append(claim)
            continue
        if _quote_in_chunk(claim.supporting_quote, chunk_text):
            kept.append(claim)
        else:
            bad_quote.append(claim)

    log.info(
        "quote verification: kept=%d, bad_chunk=%d, bad_quote=%d, short_quote=%d (input=%d)",
        len(kept), len(bad_chunk), len(bad_quote), len(short_quote), len(claims),
    )
    return QuoteVerificationResult(
        kept_claims=kept,
        dropped_for_unknown_chunk=bad_chunk,
        dropped_for_quote_not_found=bad_quote,
        dropped_for_short_quote=short_quote,
    )


def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace, replace smart quotes with straight,
    strip leading/trailing punctuation. Matches the kind of variation that
    occurs when a model paraphrases a quote slightly without changing meaning.
    """
    if not text:
        return ""
    # Replace common smart quotes / typographical variants
    out = (text
           .replace("‘", "'").replace("’", "'")
           .replace("“", '"').replace("”", '"')
           .replace("–", "-").replace("—", "-"))
    # Lowercase
    out = out.lower()
    # Collapse all whitespace runs to single space
    out = re.sub(r"\s+", " ", out)
    return out.strip()


def _quote_in_chunk(quote: str, chunk_text: str) -> bool:
    """Substring check with normalization. Falls back to fuzzy ratio if the
    exact normalized substring isn't found (handles minor paraphrase / OCR
    artifacts in the chunk)."""
    nq = _normalize(quote)
    nc = _normalize(chunk_text)

    if not nq:
        return False

    if nq in nc:
        return True

    # Fuzzy fallback: try difflib's SequenceMatcher to find best alignment
    try:
        from difflib import SequenceMatcher
        # Find the best matching substring of length len(nq) in nc
        # Use SequenceMatcher to compute a ratio against the closest window
        # of equal length
        if len(nq) > len(nc):
            return False
        best_ratio = 0.0
        # Walk the chunk in steps, computing ratio at each position
        # Step size = ~1/8 of quote length; bounded between 4 and 32
        step = max(4, min(32, len(nq) // 8))
        for i in range(0, len(nc) - len(nq) + 1, step):
            window = nc[i:i + len(nq)]
            ratio = SequenceMatcher(None, nq, window).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                if best_ratio >= FUZZY_RATIO_THRESHOLD:
                    return True
        return best_ratio >= FUZZY_RATIO_THRESHOLD
    except Exception as e:
        log.warning("fuzzy quote match failed: %s", e)
        return False
