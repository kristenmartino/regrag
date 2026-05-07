"""Stage 4: chunk parsed documents into retrieval-sized pieces.

Section-aware variable-size chunking with bounded size:
  - Use the parser's section boundaries as primary chunk boundaries.
  - Split sections that exceed TARGET_MAX_TOKENS at paragraph (\\n\\n)
    then sentence boundaries.
  - Merge adjacent candidates < TARGET_MIN_TOKENS while staying < MAX.
  - Footnotes attach to the chunk containing their nearest preceding
    paragraph; if appending overflows MAX, the footnote becomes a separate
    chunk with parent_chunk_id pointing back to the body chunk.

Token estimation is char-count / 4 — a ~5% rough English-text approximation.
TODO: swap in voyage-3.5-lite's tokenizer for accuracy once we observe
oversized chunks at retrieval time.

Chunk IDs are globally indexed within a document: f"{accession}:c{idx:04d}".
The semantic context (section heading, paragraph numbers) lives in dedicated
columns rather than the ID, so collisions like Order 2222's repeated "P5"
section labels can't happen.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from hashlib import sha256

CHUNKER_VERSION = "v1.0.0"
TARGET_MAX_TOKENS = 800
TARGET_MIN_TOKENS = 200


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass
class Chunk:
    chunk_id: str
    accession_number: str
    section_heading: str | None
    paragraph_range: str | None
    chunk_text: str
    chunk_content_hash: str
    chunk_index: int
    parent_chunk_id: str | None = None


def chunk_document(
    parsed: dict,
    accession: str,
    *,
    max_tokens: int = TARGET_MAX_TOKENS,
    min_tokens: int = TARGET_MIN_TOKENS,
) -> list[Chunk]:
    sections = parsed.get("sections", [])
    footnotes = parsed.get("footnotes", [])

    # Step 1: split oversize sections into max-bounded candidates
    candidates: list[tuple[str, str, list[int]]] = []  # (label, text, paragraph_numbers)
    for section in sections:
        sec_id = section["section_id"]
        text = (section["text"] or "").strip()
        paras = section.get("paragraph_numbers") or []
        if not text:
            continue
        if estimate_tokens(text) <= max_tokens:
            candidates.append((sec_id, text, paras))
        else:
            for i, sub in enumerate(_split_oversize(text, max_tokens)):
                candidates.append((f"{sec_id}#{i}", sub, paras))

    # Step 2: merge consecutive small candidates while staying under max
    merged = _merge_undersize(candidates, min_tokens, max_tokens)

    # Step 3: convert to Chunk records with global indexing
    chunks: list[Chunk] = []
    paragraph_to_chunk: dict[int, Chunk] = {}
    for label, text, paras in merged:
        idx = len(chunks)
        c = Chunk(
            chunk_id=_chunk_id(accession, idx),
            accession_number=accession,
            section_heading=label,
            paragraph_range=_format_paragraph_range(paras),
            chunk_text=text,
            chunk_content_hash=_hash(text),
            chunk_index=idx,
        )
        chunks.append(c)
        for p in paras:
            paragraph_to_chunk.setdefault(p, c)

    # Step 4: attach footnotes
    for fn in footnotes:
        fn_text = (fn.get("text") or "").strip()
        if not fn_text:
            continue
        target = paragraph_to_chunk.get(fn.get("nearest_preceding_paragraph"))
        appended = (
            f"\n\n[Footnote {fn.get('number') or '?'}] {fn_text}"
        )
        if target is None or estimate_tokens(target.chunk_text + appended) > max_tokens:
            # standalone footnote chunk
            idx = len(chunks)
            chunks.append(Chunk(
                chunk_id=_chunk_id(accession, idx),
                accession_number=accession,
                section_heading=(
                    f"footnote (parent: {target.section_heading})" if target
                    else f"footnote (orphan, near P{fn.get('nearest_preceding_paragraph')})"
                ),
                paragraph_range=target.paragraph_range if target else None,
                chunk_text=fn_text,
                chunk_content_hash=_hash(fn_text),
                chunk_index=idx,
                parent_chunk_id=target.chunk_id if target else None,
            ))
        else:
            target.chunk_text += appended
            target.chunk_content_hash = _hash(target.chunk_text)

    return chunks


def chunk_to_dict(chunk: Chunk) -> dict:
    return asdict(chunk)


# ---- internal helpers ----


def _chunk_id(accession: str, idx: int) -> str:
    return f"{accession}:c{idx:04d}"


def _hash(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()[:16]


def _format_paragraph_range(paras: list[int]) -> str | None:
    if not paras:
        return None
    if len(paras) == 1:
        return f"P{paras[0]}"
    return f"P{paras[0]}-P{paras[-1]}"


def _split_oversize(text: str, max_tokens: int) -> list[str]:
    """Split text at paragraph then sentence boundaries until each part <= max_tokens."""
    parts: list[str] = []
    current = ""
    paragraphs = text.split("\n\n") if "\n\n" in text else [text]
    for p in paragraphs:
        candidate = (current + "\n\n" + p) if current else p
        if estimate_tokens(candidate) <= max_tokens:
            current = candidate
        else:
            if current:
                parts.append(current)
                current = ""
            if estimate_tokens(p) <= max_tokens:
                current = p
            else:
                # paragraph itself too big — split at sentence boundaries
                for s in _split_sentences(p):
                    candidate = (current + " " + s) if current else s
                    if estimate_tokens(candidate) <= max_tokens:
                        current = candidate
                    else:
                        if current:
                            parts.append(current)
                        current = s
    if current:
        parts.append(current)
    return parts


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\d])")


def _split_sentences(text: str) -> list[str]:
    return _SENTENCE_RE.split(text) or [text]


def _merge_undersize(
    candidates: list[tuple[str, str, list[int]]],
    min_tokens: int,
    max_tokens: int,
) -> list[tuple[str, str, list[int]]]:
    """Greedy merge of consecutive small candidates while staying under max."""
    merged: list[tuple[str, str, list[int]]] = []
    i = 0
    while i < len(candidates):
        label, text, paras = candidates[i]
        while estimate_tokens(text) < min_tokens and i + 1 < len(candidates):
            n_label, n_text, n_paras = candidates[i + 1]
            combined = text + "\n\n" + n_text
            if estimate_tokens(combined) > max_tokens:
                break
            text = combined
            paras = list(paras) + list(n_paras)
            i += 1
        merged.append((label, text, paras))
        i += 1
    return merged
