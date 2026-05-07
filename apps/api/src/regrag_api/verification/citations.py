"""Citation verifier — extract `[[chunk_id]]` markers from a draft answer,
check each against the retrieved chunk_ids, and decide whether to regenerate
or strip.

Per docs/implementation-plan.md §2.7:
  - regex: `\\[\\[(?P<chunk_id>[^\\]]+)\\]\\]`
  - if no invalid citations → finalize as-is
  - if invalid > 30% AND regen_count < 2 → regenerate
  - else → strip invalid citations and finalize
"""

from __future__ import annotations

import logging
import re

from ..orchestration.state import VerificationResult

CITATION_RE = re.compile(r"\[\[(?P<chunk_id>[^\]]+)\]\]")
REGEN_THRESHOLD_INVALID_FRAC = 0.3
MAX_REGENERATIONS = 2

log = logging.getLogger(__name__)


def _normalize_citation(raw: str) -> str:
    """Strip common LLM mistakes: leading 'ACC:', leading 'chunk_id=', whitespace."""
    s = raw.strip()
    for prefix in ("ACC:", "acc:", "chunk_id=", "ACCESSION:"):
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s.strip()


def verify_citations(
    draft: str,
    retrieved_chunk_ids: set[str],
    *,
    regeneration_count: int,
) -> VerificationResult:
    """Inspect the draft answer's citation markers and decide on action."""
    cited_in_text = [_normalize_citation(m.group("chunk_id")) for m in CITATION_RE.finditer(draft)]
    cited_unique = list(dict.fromkeys(cited_in_text))  # dedupe preserve order

    if not cited_unique:
        # No citations at all — finalize as-is. The eval will catch this if
        # the model is failing to cite when it should.
        return VerificationResult(
            valid_citations=[], invalid_citations=[],
            cleaned_text=draft, action="finalize",
            notes="no citations in draft",
        )

    valid = [c for c in cited_unique if c in retrieved_chunk_ids]
    invalid = [c for c in cited_unique if c not in retrieved_chunk_ids]
    invalid_frac = len(invalid) / len(cited_unique)

    if not invalid:
        return VerificationResult(
            valid_citations=valid, invalid_citations=[],
            cleaned_text=draft, action="finalize",
            notes=f"all {len(valid)} citations valid",
        )

    if invalid_frac > REGEN_THRESHOLD_INVALID_FRAC and regeneration_count < MAX_REGENERATIONS:
        log.info(
            "verifier: %d/%d (%.0f%%) invalid citations — regenerate (attempt %d/%d)",
            len(invalid), len(cited_unique), invalid_frac * 100,
            regeneration_count + 1, MAX_REGENERATIONS,
        )
        return VerificationResult(
            valid_citations=valid, invalid_citations=invalid,
            cleaned_text="", action="regenerate",
            notes=f"{invalid_frac:.0%} invalid citations — regenerate",
        )

    # Strip invalid citations from text. Use the normalized form for lookup
    # but rewrite each marker with the normalized chunk_id (so any ACC: prefix
    # the model added gets cleaned up in the final output).
    invalid_set = set(invalid)
    def _rewrite(m: re.Match) -> str:
        normalized = _normalize_citation(m.group("chunk_id"))
        return f"[[{normalized}]]" if normalized not in invalid_set else ""
    cleaned = CITATION_RE.sub(_rewrite, draft)
    # Tidy up double spaces and orphaned punctuation that strip leaves behind
    cleaned = re.sub(r"\s+([.,;])", r"\1", cleaned)
    cleaned = re.sub(r"  +", " ", cleaned)
    log.info(
        "verifier: stripped %d invalid citations (kept %d valid) — finalize",
        len(invalid), len(valid),
    )
    return VerificationResult(
        valid_citations=valid, invalid_citations=invalid,
        cleaned_text=cleaned, action="finalize",
        notes=f"stripped {len(invalid)} invalid",
    )
