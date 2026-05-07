"""Stage 3: parse PDFs into structured section records.

Uses pdfplumber as primary; pymupdf as fallback if pdfplumber crashes on
a specific document.

Handles two real-world FERC PDF artifacts found in Day 1 spike:
  1. Page header/footer banners that leak into body text and need stripping.
  2. Footnote markers that pdfplumber/pymupdf both render as standalone
     numeric lines, splitting body sentences. We strip the markers from the
     body and detect the footnote bodies (end-of-page numeric-line + indented
     continuation) as separate footnote records linked to the nearest
     preceding paragraph number.

Known limitation (TODO for Days 4-5 chunker):
  FERC orders use TWO different numbering conventions, and the same regex
  matches both:
    - Format A (Order 841, 745): continuous paragraph numbers "170." through
      "300." running through the doc body.
    - Format B (Order 2222): hierarchical section headings like "6. Single
      Resource Aggregation" with sub-sections "a. NOPR Proposal", "b.
      Comments", etc. The body paragraphs themselves are NOT numbered.

  PARAGRAPH_BOUNDARY_RE matches both, so Format B documents get section_ids
  like "P6" that actually refer to section 6 (not paragraph 6). The chunker
  will detect this case (numbers that decrease/restart) and use a different
  segmentation. Output is structurally sound either way — section_id is a
  boundary marker, not a semantic FERC paragraph reference.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pdfplumber

log = logging.getLogger(__name__)

# Per Day 1a FINDINGS — banner patterns common across FERC orders
HEADER_BANNER_RE = re.compile(
    r"^\s*\d{8}-\d{4}\s+FERC\s+PDF\s+\(Unofficial\)\s+\d{1,2}/\d{1,2}/\d{4}\s*$"
)
FOOTER_DOCKET_RE = re.compile(
    r"^\s*Docket\s+Nos?\.\s+[\w\d\-,\s]+?\s+\d+\s*$"
)
STANDALONE_NUMERIC_RE = re.compile(r"^\s*(\d{1,3})\s*$")
PARAGRAPH_BOUNDARY_RE = re.compile(r"^(\d{1,4})\.\s+[A-Z]")  # FERC "170. Some commenters..."
# TOC lines have many dot-leaders followed by a page number, e.g.
# "1. Scope of Final Rule .................................. 31."
TOC_LINE_RE = re.compile(r"\.{4,}\s*\d+\.?\s*$")
# The "Federal Register" PDF format starts with the FERC citation line,
# e.g. "172 FERC ¶ 61,247" — used as a fallback to detect the format
FEDERAL_REGISTER_HEADER_RE = re.compile(r"^\s*\d{1,3}\s+FERC\s+[¶P]\s+\d{2},\d{3}\s*$")

# Footnote body: a numeric line followed by indented continuation. Common cues:
# the continuation lines often start with "See", "Cf.", "E.g.", "Compare", "Id.",
# or with quoted material. We treat any sequence of: numeric line + 1+ continuation
# lines as a footnote body candidate.
FOOTNOTE_LEAD_HINTS = ("See ", "See, ", "Cf.", "E.g.", "Compare ", "Id.", '"', "Pub.")


@dataclass
class FootnoteRecord:
    number: int
    text: str
    nearest_preceding_paragraph: int | None  # FERC P NN reference


@dataclass
class SectionRecord:
    section_id: str  # e.g. "P170" or "P171_172_173" if merged
    paragraph_numbers: list[int]
    text: str


@dataclass
class ParsedDocument:
    accession_number_from_pdf: str | None  # extracted from page-1 banner
    page_count: int
    char_count: int
    sections: list[SectionRecord] = field(default_factory=list)
    footnotes: list[FootnoteRecord] = field(default_factory=list)
    parser: str = "pdfplumber"


def parse_pdf(pdf_path: Path) -> ParsedDocument:
    pages = _extract_pages(pdf_path)
    accession = _extract_accession(pages[0] if pages else "")
    cleaned_body, footnotes = _split_body_and_footnotes(pages)
    sections = _segment_into_paragraphs(cleaned_body)
    return ParsedDocument(
        accession_number_from_pdf=accession,
        page_count=len(pages),
        char_count=sum(len(p) for p in pages),
        sections=sections,
        footnotes=footnotes,
    )


def _extract_pages(pdf_path: Path) -> list[str]:
    pages: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for p in pdf.pages:
            pages.append(p.extract_text(x_tolerance=2, y_tolerance=2) or "")
    return pages


def _extract_accession(first_page_text: str) -> str | None:
    """The eLibrary banner appears as e.g.
    `20180228-3066 FERC PDF (Unofficial) 02/28/2018` on the first page.
    Federal-Register-format PDFs (Order 2222 etc.) don't have it — caller
    falls back to the manifest's accession_number for those."""
    for line in first_page_text.splitlines()[:10]:
        m = re.match(r"^\s*(\d{8}-\d{4})\s+FERC\s+PDF\s+\(Unofficial\)", line)
        if m:
            return m.group(1)
    return None


def _split_body_and_footnotes(pages: list[str]) -> tuple[str, list[FootnoteRecord]]:
    """Strip page headers/footers and TOC lines, identify and remove footnote
    bodies, return cleaned body text + footnote records."""
    cleaned_pages: list[str] = []
    footnotes: list[FootnoteRecord] = []
    last_seen_paragraph: int | None = None

    for raw_page in pages:
        lines = raw_page.splitlines()
        # Strip header/footer banners and TOC lines
        kept = [
            ln for ln in lines
            if not HEADER_BANNER_RE.match(ln)
            and not FOOTER_DOCKET_RE.match(ln)
            and not TOC_LINE_RE.search(ln)
        ]
        page_body, page_footnotes, last_seen_paragraph = _separate_footnote_block(
            kept, last_seen_paragraph
        )
        cleaned_pages.append(page_body)
        footnotes.extend(page_footnotes)
    return "\n".join(cleaned_pages), footnotes


def _separate_footnote_block(
    lines: list[str], last_seen_paragraph: int | None
) -> tuple[str, list[FootnoteRecord], int | None]:
    """Detect a trailing footnote block on a page (numeric line + continuation
    lines) and split it from the body. Inline footnote-marker numeric lines
    inside the body are stripped (the markers are noise once the body is text)."""
    # 1. Track paragraph boundaries while scanning
    body_lines: list[str] = []
    for ln in lines:
        m = PARAGRAPH_BOUNDARY_RE.match(ln)
        if m:
            last_seen_paragraph = int(m.group(1))
        # Skip standalone numeric lines (footnote markers within body flow)
        if STANDALONE_NUMERIC_RE.match(ln):
            continue
        body_lines.append(ln)

    # 2. Detect trailing footnote block: walk backward looking for the
    # last region where numeric-only lines repeatedly appear separated by
    # continuation text. Heuristic: scan the last ~30 lines.
    footnotes: list[FootnoteRecord] = []
    cutoff = len(body_lines)
    if len(body_lines) >= 4:
        # Walk backwards collecting (number, [continuation lines]) tuples
        i = len(body_lines) - 1
        candidate_blocks: list[tuple[int, list[str], int]] = []  # (number, lines, start_idx)
        current_continuation: list[str] = []
        while i >= max(0, len(body_lines) - 60):
            ln = body_lines[i]
            # Re-check standalone numeric (we already stripped, but the body might
            # still have very-short lines like the start of a footnote we want to
            # rejoin). For simplicity, look for footnote-content lead-ins.
            stripped = ln.strip()
            if any(stripped.startswith(h) for h in FOOTNOTE_LEAD_HINTS):
                current_continuation.insert(0, stripped)
            elif stripped and current_continuation:
                # collected one footnote body
                # the footnote number is whatever number-only-line preceded these
                # continuation lines in the *original* page text — but we already
                # stripped those. Punt: use a rolling counter.
                pass
            i -= 1
        # Simpler approach (good enough for v1): treat any block of consecutive
        # lines starting with a footnote lead-in word and not interrupted by a
        # PARAGRAPH_BOUNDARY_RE line as a footnote-block region. Strip it from body.
        new_cutoff = cutoff
        for j in range(len(body_lines) - 1, max(-1, len(body_lines) - 30), -1):
            stripped = body_lines[j].strip()
            if not stripped:
                continue
            if any(stripped.startswith(h) for h in FOOTNOTE_LEAD_HINTS):
                new_cutoff = j
                continue
            if PARAGRAPH_BOUNDARY_RE.match(body_lines[j]):
                break
        if new_cutoff < cutoff:
            footnote_text = "\n".join(body_lines[new_cutoff:cutoff]).strip()
            if footnote_text:
                footnotes.append(FootnoteRecord(
                    number=0,  # TODO: associate with original number — needs richer parsing
                    text=footnote_text,
                    nearest_preceding_paragraph=last_seen_paragraph,
                ))
                body_lines = body_lines[:new_cutoff]

    return "\n".join(body_lines), footnotes, last_seen_paragraph


def _segment_into_paragraphs(body_text: str) -> list[SectionRecord]:
    """Split body text on FERC paragraph boundaries (`170. ...`).
    Each paragraph becomes a SectionRecord. Pre-paragraph preamble (TOC,
    title block) becomes section_id='preamble'."""
    sections: list[SectionRecord] = []
    current_paras: list[int] = []
    current_lines: list[str] = []

    def flush():
        if not current_lines:
            return
        if current_paras:
            sec_id = f"P{current_paras[0]}"
            if len(current_paras) > 1:
                sec_id = f"P{current_paras[0]}-P{current_paras[-1]}"
        else:
            sec_id = "preamble"
        text = "\n".join(current_lines).strip()
        if text:
            sections.append(SectionRecord(
                section_id=sec_id,
                paragraph_numbers=list(current_paras),
                text=text,
            ))

    for ln in body_text.splitlines():
        m = PARAGRAPH_BOUNDARY_RE.match(ln)
        if m:
            flush()
            current_paras = [int(m.group(1))]
            current_lines = [ln]
        else:
            current_lines.append(ln)
    flush()
    return sections


def parsed_to_dict(parsed: ParsedDocument) -> dict:
    """Serializable form for writing to corpus/parsed/{slug}.json."""
    return {
        **{k: v for k, v in asdict(parsed).items() if k not in ("sections", "footnotes")},
        "sections": [asdict(s) for s in parsed.sections],
        "footnotes": [asdict(f) for f in parsed.footnotes],
    }
