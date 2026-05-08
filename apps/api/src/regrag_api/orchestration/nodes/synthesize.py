"""Synthesize node — generate the final answer from retrieved chunks.

Per docs/implementation-plan.md §2.4 (synthesize == final generation, after the
M4 critique reconciliation). Uses Sonnet. Output is JSON with `refused` flag
and `answer` text containing `[[chunk_id]]` citation markers.
"""

from __future__ import annotations

import logging
import time

from .._anthropic import SYNTHESIS_MODEL, get_client, parse_json_response
from ..state import GraphState

log = logging.getLogger(__name__)

MAX_CHUNKS_IN_PROMPT = 12  # cap to keep prompt size bounded

SYSTEM_PROMPT_TEMPLATE = """\
You are answering questions about FERC regulatory orders. Use ONLY the chunks below; do not draw on outside knowledge or training data. Each chunk has a chunk_id, accession_number, and section heading.

CITATION DISCIPLINE — read this carefully:

Before you write any sentence with a citation, find the specific phrase in the cited chunk that asserts what you're claiming. If the chunk discusses the topic but does not make the specific assertion you want to make — DO NOT MAKE THE CLAIM. Drop it. A shorter answer with grounded claims is much better than a longer answer with paraphrased-from-context claims.

The most common failure to avoid: citing a chunk that DISCUSSES a topic to support a claim about what the COMMISSION RULED. Examples:
  BAD: cite a chunk that says "Some commenters argue X" to support a claim that "FERC requires X"
  BAD: cite a chunk that says "The petitioner sought clarification on Y" to support a claim about "the Commission's clarification on Y"
  BAD: cite a chunk discussing jurisdictional context to support a specific tariff requirement

Discussing a topic ≠ establishing a rule. If only commenter views, petitions, or background context appear in the chunk, the chunk does not support a claim about what FERC requires. Drop the claim.

Cite every substantive claim using the chunk_id in double square brackets, exactly as shown in the chunk's header. Place the citation at the end of the sentence the chunk supports. Multiple citations per sentence are fine.

CITATION FORMAT — copy the chunk_id verbatim, including all colons and digits:
  Correct:   [[20200917-3084:c0111]]
  Correct:   [[20180228-3066:c0124]] [[20180228-3066:c0129]]
  WRONG:     [[ACC:c0111]]                  ← do not abbreviate or substitute
  WRONG:     [[20200917-3084]]              ← must include the cNNNN suffix
  WRONG:     [[Order 2222]]                 ← must use the chunk_id, not a label

If the chunks do NOT address the question, or only address tangential aspects, set "refused" to true with a one-sentence reason. Do NOT invent details to fill the gap. A refusal with a short explanation is the right answer when the corpus genuinely doesn't speak to the question.

If the question is ambiguous about which document or aspect, you may answer based on the most relevant chunks but call out the ambiguity in the answer text.

Return ONLY valid JSON, no commentary, no markdown fences. Two valid shapes:

For an answer:
{{"refused": false, "answer": "Order 2222 requires each RTO/ISO to revise its tariff [[20200917-3084:c0111]]. The compliance deadline is 270 days after Federal Register publication [[20200917-3084:c0166]]."}}

For a refusal:
{{"refused": true, "refusal_reason": "The retrieved chunks address Order 841 storage participation, not the residential rooftop solar permitting that the question asks about."}}

CHUNKS:
{chunks_block}

{regen_note}
"""


def synthesize(state: GraphState) -> dict:
    t0 = time.perf_counter()
    chunks = state.get("retrieved_chunks") or []
    if not chunks:
        log.warning("synthesize called with no chunks — returning refusal")
        return {
            "draft_answer": "",
            "refusal_emitted": True,
            "refusal_reason": "no_relevant_chunks",
        }

    chunks_for_prompt = chunks[:MAX_CHUNKS_IN_PROMPT]
    chunks_block = _format_chunks_for_prompt(chunks_for_prompt)

    regen_count = state.get("regeneration_count", 0)
    regen_note = ""
    if regen_count > 0:
        regen_note = (
            f"\nIMPORTANT: This is regeneration attempt #{regen_count}. The previous "
            "draft cited chunk_ids that don't exist in the chunks above. Use ONLY "
            "chunk_ids that appear verbatim in the CHUNKS section."
        )

    system = SYSTEM_PROMPT_TEMPLATE.format(chunks_block=chunks_block, regen_note=regen_note)
    client = get_client()
    response = client.messages.create(
        model=SYNTHESIS_MODEL,
        max_tokens=2048,
        system=system,
        messages=[{"role": "user", "content": state["query"]}],
    )
    text = response.content[0].text
    try:
        parsed = parse_json_response(text)
    except (ValueError, Exception) as e:
        # Fallback: Sonnet 4.6 sometimes ignores the JSON-output instruction
        # and produces a clean text answer with [[chunk_id]] citations. That's
        # functionally fine — treat it as a non-refusal answer rather than
        # crashing the graph. The verifier downstream still strips bad citations.
        log.warning("synthesize JSON parse failed (%s) — treating raw text as answer", type(e).__name__)
        parsed = {"refused": False, "answer": text.strip()}

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    timings = dict(state.get("timings", {}))
    # accumulate across regen attempts
    timings["synthesize"] = timings.get("synthesize", 0) + elapsed_ms
    model_ids = dict(state.get("model_ids_used", {}))
    model_ids["synthesize"] = SYNTHESIS_MODEL
    token_counts = dict(state.get("token_counts", {}))
    prev = token_counts.get("synthesize", {"in": 0, "out": 0})
    token_counts["synthesize"] = {
        "in": prev["in"] + response.usage.input_tokens,
        "out": prev["out"] + response.usage.output_tokens,
    }

    refused = bool(parsed.get("refused", False))
    if refused:
        log.info("synthesize refused: %s", parsed.get("refusal_reason"))
        return {
            "draft_answer": text,  # preserve raw model output for audit
            "synthesize_prompt": system,
            "refusal_emitted": True,
            "refusal_reason": "llm_refusal",
            "final_answer": parsed.get("refusal_reason") or "Unable to answer from the corpus.",
            "timings": timings,
            "model_ids_used": model_ids,
            "token_counts": token_counts,
        }

    draft = parsed.get("answer", "")
    log.info("synthesized %d-char draft answer in %dms (regen=%d)", len(draft), elapsed_ms, regen_count)
    return {
        "draft_answer": draft,
        "synthesize_prompt": system,  # captured for audit log
        "timings": timings,
        "model_ids_used": model_ids,
        "token_counts": token_counts,
    }


def _format_chunks_for_prompt(chunks: list[dict]) -> str:
    """Format retrieved chunks as a compact block the model can cite from."""
    lines: list[str] = []
    for c in chunks:
        section = c.get("section_heading") or "?"
        lines.append(
            f"[chunk_id={c['chunk_id']} | accession={c['accession_number']} | section={section}]\n"
            f"{c['chunk_text'].strip()}\n"
        )
    return "\n".join(lines)
