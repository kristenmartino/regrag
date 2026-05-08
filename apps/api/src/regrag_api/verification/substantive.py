"""Inline substantive-support check — runtime version of the eval's
LLM-as-judge.

The existing citation verifier (citations.py) only checks that cited
`chunk_id`s exist in the retrieved set. It does not check whether each
cited chunk actually contains the assertion the model is making. The
eval surfaced this as the dominant failure mode: ~30% of cited claims
are attached to chunks that are *topical* but not *specifically supportive*
(e.g. citing a chunk discussing commenter views to support a claim about
the Commission's ruling).

This module runs a lightweight Haiku judge over each (claim, chunk) pair
inline as part of verification:

  - Walk the draft, extract (sentence, [chunk_ids]) pairs (reuses the eval's
    extractor)
  - One Haiku call with all pairs in a single prompt → JSON list of 0/1 scores
    with one-sentence reasons
  - Sentences whose ALL citations score 0 are stripped from the final answer
  - Sentences with a mix get the unsupported citations stripped (sentence stays)
  - If >50% of sentences are stripped entirely, signal that the verify node
    should regenerate (bounded by the existing regen cap)

Cost: ~6K input + 500 output Haiku tokens per response, ~$0.005, 2–3s.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from ..orchestration._anthropic import CLASSIFIER_MODEL, get_client, parse_json_response

log = logging.getLogger(__name__)

CITATION_RE = re.compile(r"\[\[(?P<chunk_id>[^\]]+)\]\]")
# Same sentence splitter the eval judge uses
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\d#*\-])")

JUDGE_PROMPT_VERSION = "inline-v1.0"

SYSTEM_PROMPT = f"""\
You are a citation faithfulness checker for a regulatory-corpus RAG system.

For each (claim, cited_chunk_text) pair, decide whether the chunk substantively
supports the specific claim being made. "Substantively" means the chunk contains
the specific fact, rule, or interpretation the claim asserts — not merely the
same topic.

Score 1 if the chunk supports the claim. Score 0 if it does not, or only
tangentially mentions the topic without supporting the specific assertion.

Be strict but not pedantic. A claim that paraphrases the chunk's substance
should score 1. A claim that adds a specific fact (a number, a deadline, a
jurisdictional carve-out) NOT in the chunk should score 0, even if the chunk
is on-topic.

Common cases that should score 0:
- Claim asserts a Commission requirement; chunk only discusses commenter views
- Claim asserts a specific number; chunk discusses the topic but not the number
- Claim asserts a jurisdictional decision; chunk only describes background

Return ONLY valid JSON:
{{"scores": [{{"pair_idx": 0, "score": 1, "reason": "<10 words max>"}}, ...]}}

Judge prompt version: {JUDGE_PROMPT_VERSION}
"""


@dataclass
class SentenceWithCitations:
    sentence: str           # full sentence text including citation markers
    plain_text: str         # sentence with citation markers stripped
    chunk_ids: list[str]    # chunk_ids cited in this sentence


@dataclass
class SubstantiveCheckResult:
    cleaned_text: str
    sentences_stripped: int            # whole-sentence drops
    citations_stripped: int            # individual-citation drops within kept sentences
    judge_notes: list[dict]            # per-pair judge output for audit
    should_regenerate: bool            # signal back to the verify node


def extract_sentences_with_citations(text: str) -> list[SentenceWithCitations]:
    """Walk the draft, return a list of sentences each paired with its cited chunk_ids."""
    parts = SENTENCE_RE.split(text) if text else []
    out: list[SentenceWithCitations] = []
    for part in parts:
        cited = [m.group("chunk_id") for m in CITATION_RE.finditer(part)]
        if not cited:
            continue
        plain = CITATION_RE.sub("", part).strip()
        if plain:
            out.append(SentenceWithCitations(sentence=part, plain_text=plain, chunk_ids=cited))
    return out


def check_substantive_support(
    query: str,
    draft: str,
    retrieved_chunks: list[dict],
    *,
    sentence_strip_regen_threshold: float = 0.5,
) -> SubstantiveCheckResult:
    """Run the inline judge and produce a cleaned answer.

    Sentences with ALL citations scoring 0 are dropped. Sentences with a mix
    keep the supported citations and drop the rest. If more than
    sentence_strip_regen_threshold of sentences are dropped, sets
    should_regenerate=True so the verify node can route back to synthesize.
    """
    sentences = extract_sentences_with_citations(draft)
    if not sentences:
        return SubstantiveCheckResult(
            cleaned_text=draft, sentences_stripped=0, citations_stripped=0,
            judge_notes=[], should_regenerate=False,
        )

    chunk_lookup = {c["chunk_id"]: c for c in retrieved_chunks}

    # Build the (pair_idx, sentence_idx, chunk_id) triples to score.
    pairs: list[tuple[int, int, str, str]] = []  # (pair_idx, sentence_idx, chunk_id, claim_text)
    for s_idx, s in enumerate(sentences):
        for cid in s.chunk_ids:
            chunk = chunk_lookup.get(cid)
            if chunk is None:
                # The chunk_id verifier already removed bad chunk_ids before we got here,
                # so this would be unusual. Skip and treat as unscored.
                continue
            pairs.append((len(pairs), s_idx, cid, s.plain_text))

    if not pairs:
        return SubstantiveCheckResult(
            cleaned_text=draft, sentences_stripped=0, citations_stripped=0,
            judge_notes=[], should_regenerate=False,
        )

    user_msg = _build_user_message(query, pairs, chunk_lookup)
    scores_by_pair = _invoke_judge(user_msg, n_pairs=len(pairs))

    if not scores_by_pair:
        # Judge failed; fall back to the as-is draft rather than regenerating
        log.warning("substantive judge returned no scores; finalizing draft as-is")
        return SubstantiveCheckResult(
            cleaned_text=draft, sentences_stripped=0, citations_stripped=0,
            judge_notes=[{"warning": "judge_unavailable"}], should_regenerate=False,
        )

    # Build per-sentence support map: which chunk_ids were judged supportive
    sentence_supportive_chunk_ids: dict[int, set[str]] = {i: set() for i in range(len(sentences))}
    judge_notes: list[dict] = []
    for pair_idx, s_idx, cid, _claim in pairs:
        score = scores_by_pair.get(pair_idx, {}).get("score")
        reason = scores_by_pair.get(pair_idx, {}).get("reason", "")
        judge_notes.append({
            "sentence_idx": s_idx,
            "chunk_id": cid,
            "score": score,
            "reason": reason,
        })
        if score == 1:
            sentence_supportive_chunk_ids[s_idx].add(cid)

    # Reconstruct the cleaned text
    cleaned_parts: list[str] = []
    sentences_stripped = 0
    citations_stripped = 0
    sent_iter = iter(sentences)
    for s_idx, s in enumerate(sent_iter):
        supportive = sentence_supportive_chunk_ids.get(s_idx, set())
        if not supportive:
            # ALL citations failed → drop the whole sentence
            sentences_stripped += 1
            continue
        if len(supportive) == len(s.chunk_ids):
            # All citations supportive → keep sentence as-is
            cleaned_parts.append(s.sentence.strip())
        else:
            # Mixed: strip just the unsupported citations from this sentence
            n_dropped = len(s.chunk_ids) - len(supportive)
            citations_stripped += n_dropped

            def _keep(m: re.Match, supportive=supportive) -> str:
                return f"[[{m.group('chunk_id')}]]" if m.group("chunk_id") in supportive else ""

            rewritten = CITATION_RE.sub(_keep, s.sentence).strip()
            # tidy double spaces / orphan punctuation
            rewritten = re.sub(r"\s+([.,;])", r"\1", rewritten)
            rewritten = re.sub(r"  +", " ", rewritten)
            if rewritten:
                cleaned_parts.append(rewritten)

    cleaned = " ".join(cleaned_parts).strip()
    strip_rate = sentences_stripped / max(1, len(sentences))
    should_regen = strip_rate > sentence_strip_regen_threshold

    log.info(
        "substantive judge: %d/%d sentences stripped (%.0f%%), %d citations stripped from kept sentences",
        sentences_stripped, len(sentences), strip_rate * 100, citations_stripped,
    )
    return SubstantiveCheckResult(
        cleaned_text=cleaned or draft,  # if everything got stripped, leave original for regen decision
        sentences_stripped=sentences_stripped,
        citations_stripped=citations_stripped,
        judge_notes=judge_notes,
        should_regenerate=should_regen,
    )


def _build_user_message(
    query: str,
    pairs: list[tuple[int, int, str, str]],
    chunk_lookup: dict[str, dict],
) -> str:
    user_msg = f"Original query: {query}\n\nClaim/citation pairs:\n\n"
    for pair_idx, _s_idx, cid, claim in pairs:
        chunk_text = (chunk_lookup[cid].get("chunk_text") or "")[:1500]
        user_msg += (
            f"--- pair {pair_idx} ---\n"
            f"claim: {claim}\n"
            f"chunk_id: {cid}\n"
            f"chunk text:\n{chunk_text}\n\n"
        )
    user_msg += (
        f"\nScore each of the {len(pairs)} pairs (indexed by 'pair_idx', 0..{len(pairs)-1}). "
        "Return JSON: {\"scores\": [{\"pair_idx\": <int>, \"score\": <0|1>, \"reason\": \"<short>\"}, ...]}\n"
        "Return one entry per pair, in order."
    )
    return user_msg


def _invoke_judge(user_msg: str, *, n_pairs: int) -> dict[int, dict]:
    """Run Haiku and parse scores into a {pair_idx: {score, reason}} dict.
    Returns {} on parse failure (caller falls back to no-strip)."""
    try:
        client = get_client()
        response = client.messages.create(
            model=CLASSIFIER_MODEL,  # Haiku — fast + cheap
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = response.content[0].text
        parsed = parse_json_response(text)
    except Exception as e:
        log.warning("substantive judge invocation failed: %s", e)
        return {}

    out: dict[int, dict] = {}
    for s in parsed.get("scores") or []:
        try:
            out[int(s["pair_idx"])] = {"score": int(s.get("score", 0)), "reason": s.get("reason", "")}
        except (TypeError, ValueError, KeyError):
            continue
    return out
