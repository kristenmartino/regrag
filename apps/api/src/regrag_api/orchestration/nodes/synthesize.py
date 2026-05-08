"""Synthesize node — generate the final answer from retrieved chunks.

Per docs/implementation-plan.md §2.4 (synthesize == final generation).

Uses Anthropic tool use to force the model to emit structured
(claim, chunk_id, supporting_quote) triples. Each claim must include a
direct quote from the cited chunk that asserts what the claim is making.
The verify node then substring-checks each quote against the chunk; any
claim with a bogus quote is dropped before rendering.

This is the structural counterpart to the runtime substantive judge in
verification/substantive.py: instead of asking a judge whether the chunk
supports the claim, the model is forced to identify the supporting phrase
up front, and we check that the phrase actually exists. By construction,
citation drift toward topical chunks becomes impossible (modulo
paraphrasing the quote too aggressively, which the prompt discourages and
the verifier's fuzzy fallback tolerates).
"""

from __future__ import annotations

import logging
import time

from .._anthropic import SYNTHESIS_MODEL, get_client
from ..state import GraphState

log = logging.getLogger(__name__)

MAX_CHUNKS_IN_PROMPT = 12  # cap to keep prompt size bounded

SYNTHESIZE_TOOL = {
    "name": "submit_answer",
    "description": (
        "Submit the final answer to the user's question. Each claim must be "
        "supported by a direct quote from a specific chunk."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "refused": {
                "type": "boolean",
                "description": (
                    "True if the chunks don't address the question and you can't answer "
                    "without making up details. False if you can produce at least one grounded claim."
                ),
            },
            "refusal_reason": {
                "type": "string",
                "description": (
                    "If refused=true, a one-sentence explanation of why. Use empty string otherwise."
                ),
            },
            "claims": {
                "type": "array",
                "description": (
                    "Ordered list of claims that together form the answer. Each claim is a single "
                    "assertion grounded in a specific chunk. If refused=true, this should be empty."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {
                            "type": "string",
                            "description": (
                                "The claim, written as a complete sentence in plain prose. Do NOT "
                                "include [[chunk_id]] citation markers — the renderer adds those."
                            ),
                        },
                        "chunk_id": {
                            "type": "string",
                            "description": (
                                "The chunk_id from the chunks above that supports this claim. Copy "
                                "verbatim, e.g. '20200917-3084:c0111'."
                            ),
                        },
                        "supporting_quote": {
                            "type": "string",
                            "description": (
                                "A direct quote from the cited chunk that supports the claim. The "
                                "quote MUST appear verbatim (or near-verbatim) in the chunk text. "
                                "20+ characters; long enough to be a meaningful match, short enough "
                                "to be a focused phrase. Do not include the chunk header text."
                            ),
                        },
                    },
                    "required": ["claim", "chunk_id", "supporting_quote"],
                },
            },
        },
        "required": ["refused", "claims"],
    },
}

SYSTEM_PROMPT_TEMPLATE = """\
You are answering questions about FERC regulatory orders. Use ONLY the chunks below; do not draw on outside knowledge or training data.

You will respond by calling the `submit_answer` tool. Read its parameter descriptions carefully.

CRITICAL DISCIPLINE — for each claim you want to make:
  1. Find a specific phrase or sentence in one of the chunks that ASSERTS what your claim says.
  2. Quote that phrase verbatim in `supporting_quote`. The quote MUST appear in the chunk text.
  3. Write your `claim` as a paraphrase or restatement of what the quote says.
  4. If you cannot find a supporting quote — if the chunk only DISCUSSES the topic, mentions commenter views, or provides background context — DROP THE CLAIM. Do not invent a supporting_quote.

The most common failure pattern this discipline is meant to prevent: citing a chunk that DISCUSSES a topic to support a claim about what the COMMISSION RULED. Examples:
  BAD: a chunk that says "Some commenters argue X" cannot support a claim that "FERC requires X"
  BAD: a chunk that says "The petitioner sought clarification on Y" cannot support a claim about "the Commission's clarification on Y"
  BAD: a chunk discussing jurisdictional context cannot support a specific tariff requirement

A short answer with 3 well-grounded claims is much better than a long answer with 8 paraphrased-from-context claims. Quality over coverage.

If the chunks don't address the question at all — or only tangentially — set `refused: true` with a one-sentence `refusal_reason` and an empty `claims` array. A refusal is the right answer when the corpus genuinely doesn't speak to the question.

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
            f"\nIMPORTANT: This is regeneration attempt #{regen_count}. The previous draft "
            "had unsupported claims (claims whose supporting_quote was not actually in the cited "
            "chunk, or whose cited chunk only discussed the topic without establishing the claim). "
            "Be stricter this time: only include claims where the supporting_quote appears verbatim "
            "in the cited chunk and directly establishes what the claim says."
        )

    system = SYSTEM_PROMPT_TEMPLATE.format(chunks_block=chunks_block, regen_note=regen_note)
    client = get_client()
    response = client.messages.create(
        model=SYNTHESIS_MODEL,
        max_tokens=4096,
        system=system,
        tools=[SYNTHESIZE_TOOL],
        tool_choice={"type": "tool", "name": "submit_answer"},
        messages=[{"role": "user", "content": state["query"]}],
    )

    # Pull the tool-use block from the response
    tool_use = next((b for b in response.content if getattr(b, "type", None) == "tool_use"), None)
    if tool_use is None:
        # Defensive fallback: tool_choice forced should make this impossible, but if it
        # ever happens, treat as a refusal rather than crashing.
        log.warning("synthesize: model did not call submit_answer tool — refusing")
        parsed = {"refused": True, "refusal_reason": "model failed to call the answer tool", "claims": []}
    else:
        parsed = dict(tool_use.input)

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
        reason = parsed.get("refusal_reason") or "Unable to answer from the corpus."
        log.info("synthesize refused: %s", reason)
        return {
            "draft_answer": "",
            "structured_claims": [],
            "synthesize_prompt": system,
            "refusal_emitted": True,
            "refusal_reason": "llm_refusal",
            "final_answer": reason,
            "timings": timings,
            "model_ids_used": model_ids,
            "token_counts": token_counts,
        }

    # Structured claims from the tool call
    raw_claims = parsed.get("claims") or []
    structured_claims = [
        {
            "claim": (c.get("claim") or "").strip(),
            "chunk_id": (c.get("chunk_id") or "").strip(),
            "supporting_quote": (c.get("supporting_quote") or "").strip(),
        }
        for c in raw_claims
        if isinstance(c, dict) and c.get("claim") and c.get("chunk_id")
    ]

    if not structured_claims:
        # Model returned no claims and didn't refuse — treat as a soft refusal
        log.info("synthesize: no claims emitted and not a refusal — soft refusal")
        return {
            "draft_answer": "",
            "structured_claims": [],
            "synthesize_prompt": system,
            "refusal_emitted": True,
            "refusal_reason": "llm_refusal",
            "final_answer": "The retrieved chunks do not provide grounded support for an answer to this question.",
            "timings": timings,
            "model_ids_used": model_ids,
            "token_counts": token_counts,
        }

    # Render to natural text with [[chunk_id]] markers — this is what the
    # existing chunk-id verifier and substantive judge operate on, so the
    # rest of the verify pipeline keeps working unchanged. Quote verification
    # in verify.py operates on `structured_claims` directly.
    draft = _render_claims_to_text(structured_claims)
    log.info(
        "synthesized %d claims (%d chars) in %dms (regen=%d)",
        len(structured_claims), len(draft), elapsed_ms, regen_count,
    )
    return {
        "draft_answer": draft,
        "structured_claims": structured_claims,
        "synthesize_prompt": system,  # captured for audit log
        "timings": timings,
        "model_ids_used": model_ids,
        "token_counts": token_counts,
    }


def _render_claims_to_text(claims: list[dict]) -> str:
    """Concatenate claims into a natural-text answer with [[chunk_id]] markers.
    Each claim becomes one sentence; the citation is appended after the period."""
    parts: list[str] = []
    for c in claims:
        claim = c["claim"].rstrip(" .")
        parts.append(f"{claim}. [[{c['chunk_id']}]]")
    return " ".join(parts)


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
