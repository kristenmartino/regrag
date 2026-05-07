"""LLM-as-judge for citation faithfulness.

Per docs/implementation-plan.md §2.9 / critique M3:

The judge sees the (query, model_answer, retrieved_chunks) tuple. It identifies
each cited claim in the answer and decides whether the cited chunk substantively
supports it. Returns 0/1 per claim. Question-level score is the mean.

Limitations acknowledged (per critique M3):
  - Judge bias toward verbose answers
  - Sensitivity to prompt wording
  - Lower agreement with human raters on subtle calls

Mitigation: judge prompt is versioned alongside the eval set; periodic spot-check
against 5 samples per quarter to calibrate drift. The judge model (Sonnet) is
the same as the synthesizer, which is a known LLM-as-judge weakness; using
Opus or a different family for judge would be stronger but more expensive.
"""

from __future__ import annotations

import logging
import re

from regrag_api.orchestration._anthropic import get_client, parse_json_response

JUDGE_MODEL = "claude-sonnet-4-6"
JUDGE_PROMPT_VERSION = "v1.0"

CITATION_RE = re.compile(r"\[\[(?P<chunk_id>[^\]]+)\]\]")

log = logging.getLogger(__name__)


def extract_claims_with_citations(answer_text: str) -> list[tuple[str, list[str]]]:
    """Walk the answer and pair each sentence with the chunk_ids cited in it.
    A 'sentence' here is a span ending in . ! or ? plus any inline citations
    that immediately precede the punctuation."""
    # Tokenize by sentence-end punctuation while keeping the punctuation
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\d#*\-])", answer_text)
    out: list[tuple[str, list[str]]] = []
    for part in parts:
        cited = [m.group("chunk_id") for m in CITATION_RE.finditer(part)]
        if cited:
            # Strip the citation markers from the displayed claim text so the
            # judge sees only the prose
            claim_text = CITATION_RE.sub("", part).strip()
            out.append((claim_text, cited))
    return out


SYSTEM_PROMPT = f"""\
You are an evaluator scoring citation faithfulness for a regulatory-corpus RAG system.

For each (claim, cited_chunk_text) pair, decide whether the chunk substantively
supports the claim. "Substantively" means the chunk contains the specific fact,
rule, or interpretation the claim asserts — not merely the same topic.

Score 1 if the chunk supports the claim. Score 0 if it does not, or only
tangentially mentions the topic without supporting the specific assertion.

Be strict. A claim that says "the deadline is 270 days" cited to a chunk that
just says "compliance filings are required" should score 0 — the chunk doesn't
contain the deadline.

Return ONLY valid JSON in this shape:
{{"scores": [{{"claim_idx": 0, "score": 1, "reason": "chunk explicitly states 270 days"}}, ...]}}

Judge prompt version: {JUDGE_PROMPT_VERSION}
"""


def judge_citations(
    query: str,
    answer: str,
    retrieved_chunks: list[dict],
) -> tuple[float | None, list[dict]]:
    """Run the judge over an answer's cited claims. Returns
    (mean_faithfulness_0_to_1, per_claim_notes). Returns (None, []) if the
    answer has no citations to judge."""
    chunk_lookup = {c["chunk_id"]: c for c in retrieved_chunks}
    claim_pairs = extract_claims_with_citations(answer)
    if not claim_pairs:
        return None, []

    # Build the judge prompt: enumerate each (claim, chunk_text) the judge
    # needs to score
    items: list[dict] = []
    for i, (claim, cited_ids) in enumerate(claim_pairs):
        for cid in cited_ids:
            chunk = chunk_lookup.get(cid)
            chunk_text = chunk.get("chunk_text", "") if chunk else "(chunk_id not in retrieved set)"
            items.append({
                "claim_idx": i,
                "claim": claim,
                "chunk_id": cid,
                "chunk_text": chunk_text[:1500],  # cap for prompt size
            })

    if not items:
        return None, []

    user_msg = f"Original query: {query}\n\nClaim/citation pairs to evaluate:\n\n"
    for k, it in enumerate(items):
        user_msg += (
            f"--- pair {k} ---\n"
            f"claim_idx: {it['claim_idx']}\n"
            f"claim: {it['claim']}\n"
            f"cited chunk_id: {it['chunk_id']}\n"
            f"cited chunk text:\n{it['chunk_text']}\n\n"
        )
    user_msg += (
        f"\nFor each pair (indexed by 'claim_idx'), output a score 0 or 1. "
        f"There are {len(items)} pairs total. "
        f"Return JSON: {{\"scores\": [{{\"claim_idx\": <int>, \"score\": <0|1>, \"reason\": \"<short>\"}}, ...]}}\n"
        f"Return one entry per pair, in order."
    )

    client = get_client()
    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    text = response.content[0].text
    try:
        parsed = parse_json_response(text)
    except (ValueError, Exception) as e:
        log.warning("judge JSON parse failed (%s) — returning None", e)
        return None, [{"error": f"judge_parse_failed: {e}"}]

    scores_list = parsed.get("scores") or []
    notes = []
    score_sum = 0
    score_count = 0
    for s in scores_list:
        try:
            sv = int(s.get("score", 0))
            score_sum += sv
            score_count += 1
            notes.append({
                "claim_idx": s.get("claim_idx"),
                "score": sv,
                "reason": s.get("reason", ""),
            })
        except (TypeError, ValueError):
            continue

    if score_count == 0:
        return None, notes
    return score_sum / score_count, notes
