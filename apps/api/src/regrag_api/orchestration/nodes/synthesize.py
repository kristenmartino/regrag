"""Synthesize node — generate the final answer from retrieved chunks.

Per docs/implementation-plan.md §2.4 (synthesize == final generation, after the
M4 critique reconciliation). Uses Sonnet. Output is JSON with `refused` flag
and `answer` text containing `[[chunk_id]]` citation markers.
"""

from __future__ import annotations

import logging
import time

from .._anthropic import SYNTHESIS_MODEL, get_client, parse_json_response
from ...retrieval.identifiers import primary_citation_accessions
from ..state import GraphState

log = logging.getLogger(__name__)

MAX_CHUNKS_IN_PROMPT = 12  # cap to keep prompt size bounded

SYSTEM_PROMPT_TEMPLATE = """\
You are answering questions about FERC regulatory orders. Use ONLY the chunks below; do not draw on outside knowledge or training data. Each chunk has a chunk_id, accession_number, and section heading.
{anchor_note}
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

    anchor_note = _anchor_note(
        state.get("named_orders") or [],
        state.get("anchored_roles") or {},
        chunks_for_prompt,
    )

    system = SYSTEM_PROMPT_TEMPLATE.format(
        chunks_block=chunks_block, regen_note=regen_note, anchor_note=anchor_note
    )
    client = get_client()
    try:
        response = client.messages.create(
            model=SYNTHESIS_MODEL,
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": state["query"]}],
        )
    except Exception as e:
        # Anthropic outage, rate-limit, network error, etc. Don't crash the graph;
        # surface a soft refusal so the user sees a graceful message and the
        # frontend can render its standard refusal UI. The exception type and
        # message are logged for ops triage.
        log.warning(
            "synthesize: Anthropic call failed (%s: %s) — emitting soft refusal",
            type(e).__name__, e,
        )
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        timings = dict(state.get("timings", {}))
        timings["synthesize"] = timings.get("synthesize", 0) + elapsed_ms
        return {
            "draft_answer": "",
            "synthesize_prompt": system,
            "refusal_emitted": True,
            "refusal_reason": "llm_unavailable",
            "final_answer": (
                "The synthesis model is temporarily unavailable. "
                "Please try again in a moment."
            ),
            "timings": timings,
        }
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


def _anchor_note(
    named_orders: list[str], anchored_roles: dict[str, dict[str, list[str]]], chunks: list[dict]
) -> str:
    """Build a prompt fragment telling the model which accession(s) count as the PRIMARY
    citation for each named order's subject (issue #14): a primary order's primary citation
    is its final-rule / Federal-Register source, NOT its rehearing; a rehearing's is the
    rehearing. Empty string when nothing applies.

    Only emits when at least one named order has a primary-citation accession that is also
    present among the retrieved chunks (otherwise the instruction is pointless).
    """
    if not named_orders or not anchored_roles:
        return ""
    chunk_accessions = {c.get("accession_number") for c in chunks}
    per_order: list[tuple[str, list[str]]] = []
    for o in named_orders:
        accs = [a for a in primary_citation_accessions(o, anchored_roles) if a in chunk_accessions]
        if accs:
            per_order.append((o, accs))
    if not per_order:
        return ""
    orders_str = ", ".join(f"Order {o}" for o in named_orders)
    table = "\n".join(f"  - Order {o}: [{', '.join(accs)}]" for o, accs in per_order)
    return (
        f"\nSCOPE — read this BEFORE the citation discipline section below. This rule is "
        f"applied PER-SENTENCE, not per-answer:\n"
        f"The user asked specifically about {orders_str}. Primary-citation accessions by "
        f"subject — a rehearing order (e.g. Order 841-A) is a SEPARATE document and does NOT "
        f"count as the primary source for the original order:\n"
        f"{table}\n"
        f"\nFor each sentence whose SUBJECT is one of the named order(s):\n"
        f"  1. The FIRST citation in that sentence MUST be a chunk_id whose accession_number "
        f"is one of the accessions listed for THAT order's subject above. This is non-negotiable. "
        f"You may add a second citation from a different accession (including a rehearing) as "
        f"supporting cross-reference, but the primary citation must come from that order's listed set.\n"
        f"  2. If no listed chunk supports the sentence's primary assertion: DO NOT write that "
        f"sentence. Either drop it, rewrite it to assert only what a listed chunk says, or set "
        f"refused=true.\n"
        f"  3. Sprinkling one in-scope citation elsewhere in the answer does NOT exempt other "
        f"sentences from this rule. The rule is per-sentence about the named order's subject.\n"
        f"\nWORKED EXAMPLE — the user asks 'What was the effective date of Order 841?':\n"
        f"  • BAD: 'Order 841's effective date was within 90 days of publication [[20190221-3057:c0127]]. "
        f"The errata notice confirms a Feb 15, 2018 issue date [[20180228-3066:c0000]].'\n"
        f"    (First citation is to Order 845-A, not Order 841. Adding the 841 cite second doesn't fix it.)\n"
        f"  • GOOD: 'Order 841 was issued February 15, 2018 [[20180228-3066:c0000]] and took effect "
        f"June 4, 2018, as stated in its Federal Register publication [[fr-2018-03-06-2018-03708:c0000]].'\n"
        f"  • ALSO GOOD: 'The FERC-issued Order 841 text states its effective date relative to Federal "
        f"Register publication; the literal date — June 4, 2018 — comes from the Federal Register "
        f"companion [[fr-2018-03-06-2018-03708:c0000]].' (Cite the FR companion for the literal date.)\n"
        f"\nReminder: a chunk from a DIFFERENT accession that merely mentions the named order — "
        f"including the order's own rehearing — does NOT support a claim about what the original order "
        f"itself says or requires, even if its text is topically relevant or more quotable. If no listed "
        f"chunk supports the claim, do not substitute another accession — instead, REFUSE or scope narrowly "
        f"to what the listed chunks actually say. Set refused=true and explain that the corpus chunks for "
        f"the named order don't address the specific question.\n"
        f"\nYou may still cite chunks from other accessions for claims about OTHER orders or for "
        f"genuine cross-references (e.g. 'Order 2222 builds on Order 841 [[chunk from 2222]]'). The "
        f"rule above applies only to claims whose subject is one of the named order(s) above.\n"
    )
