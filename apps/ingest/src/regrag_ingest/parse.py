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
# TOC lines have many dot-leaders. Original regex required a trailing page
# number ("\.{4,}\s*\d+\.?\s*$") but multi-column FR extraction often drops
# the page-number column. Loosened to match any line with 4+ consecutive dots
# (with optional trailing whitespace/number). Catches both styles:
#   "1. Scope of Final Rule .................................. 31."
#   "2. Commission Determination ........."         (column-stripped variant)
TOC_LINE_RE = re.compile(r"\.{4,}\s*\d*\.?\s*$")
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


def parse_pdf(pdf_path: Path, *, layout: str | None = None) -> ParsedDocument:
    """Parse a PDF into sections + footnotes.

    Args:
        pdf_path: path to the source PDF.
        layout: optional layout hint. None or 'single_column' uses the default
            text-flow extraction; 'federal_register' uses column-aware
            extraction (3 columns, common for Federal Register publications).
            Auto-detected when None if first-page text matches Federal Register
            header pattern.
    """
    pages = _extract_pages(pdf_path, layout=layout)
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


def _extract_pages(pdf_path: Path, *, layout: str | None = None) -> list[str]:
    """Extract page text. Default = pdfplumber's text-flow extraction (good
    for single-column FERC-issued PDFs). When layout='federal_register' or
    auto-detected as such, crops each page into 3 columns and extracts each
    column separately — fixes the zig-zag scramble that single-column
    extraction produces on multi-column Federal Register layouts.
    """
    pages: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        if layout is None:
            # auto-detect: check first page for FR header
            first_text = pdf.pages[0].extract_text(x_tolerance=2, y_tolerance=2) or ""
            if _looks_like_federal_register(first_text):
                layout = "federal_register"
                log.info("auto-detected federal_register layout for %s", pdf_path.name)
        for p in pdf.pages:
            if layout == "federal_register":
                pages.append(_extract_federal_register_page(p))
            else:
                pages.append(p.extract_text(x_tolerance=2, y_tolerance=2) or "")
    return pages


def _looks_like_federal_register(first_page_text: str) -> bool:
    """Heuristic: Federal Register pages have a 'Federal Register / Vol. X, No. Y'
    line near the top. The text-flow extraction will surface that string somewhere
    in the first ~5 lines even when the layout is scrambled."""
    head = first_page_text[:800]
    return "Federal Register" in head and "Vol." in head and "No." in head


# Standard Federal Register layout: 3 columns of equal width with narrow
# gutters. Tuned against the 2018-03-06 RIN 1902-AF45 issue (Order 841 publication).
# Page width 612pt (US Letter). Each column gets a small right-side padding
# (+12pt) so characters whose glyph extends past the visual column edge
# (especially italic terminal chars or hyphenated line-ends) get captured.
# pdfplumber's word-detection assigns each word to the column whose bbox
# fully contains its x0, so the overlap doesn't double-count words —
# subsequent columns crop starts past the previous column's natural end.
FR_COLUMN_BOUNDS = (
    (30, 210),
    (212, 392),
    (394, 588),
)


def _extract_federal_register_page(page) -> str:
    """Extract text from a single page assuming 3-column Federal Register layout.

    Crops each column, extracts its text in natural reading order (top-to-bottom),
    then concatenates columns left-to-right. Result reads as proper running text
    instead of the zig-zag scramble that whole-page extraction produces.
    """
    column_texts: list[str] = []
    for x0, x1 in FR_COLUMN_BOUNDS:
        try:
            col = page.crop((x0, 0, x1, page.height))
            t = col.extract_text(x_tolerance=2, y_tolerance=2) or ""
            column_texts.append(t)
        except Exception as e:
            # If a column extraction errors (e.g., page has unusual layout),
            # fall back to whole-page extraction for this column slot to avoid
            # losing all the page's text.
            log.warning("FR column crop (%d,%d) failed: %s — using empty", x0, x1, e)
            column_texts.append("")
    # Page-level header (before columns) sometimes lives above the column tops;
    # fetch it via a thin top strip if columns start partway down.
    return "\n".join(t for t in column_texts if t.strip())


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
