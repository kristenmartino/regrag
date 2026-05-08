"""GraphState — typed state schema for the LangGraph orchestration.

Field semantics (per docs/implementation-plan.md §2.4):
  - query: the original user input
  - user_id: opaque identifier for audit logging; None for anonymous demo use
  - classification: 'single_doc' or 'multi_doc' from the classifier node
  - sub_queries: populated only on the multi_doc path by the decomposer
  - retrieved_chunks: union of all chunks retrieved (deduped by chunk_id)
  - draft_answer: raw model output before citation verification
  - final_answer: post-verification text (with invalid citations stripped)
  - refusal_emitted: True if the system declined to answer
  - refusal_reason: short tag indicating why ('no_relevant_chunks' | 'llm_refusal' | etc.)
  - regeneration_count: bounded at 2 attempts before falling back to strip
  - timings: ms per stage, accumulated as the graph runs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict

Classification = Literal["single_doc", "multi_doc"]
RefusalReason = Literal[
    "no_relevant_chunks",      # pre-generation: top vector hit below threshold
    "llm_refusal",             # generation model returned refused=true
    "verification_unrecoverable",  # too many bad citations after max regens
]


@dataclass
class VerificationResult:
    valid_citations: list[str]
    invalid_citations: list[str]
    cleaned_text: str
    action: Literal["finalize", "regenerate"]
    notes: str = ""


class GraphState(TypedDict, total=False):
    # Input
    query: str
    user_id: str | None

    # Stage outputs
    classification: Classification | None
    classification_confidence: float | None
    sub_queries: list[str] | None

    # Retrieval result — list of dicts (serializable, snapshot-friendly for audit log)
    retrieved_chunks: list[dict] | None
    top_cosine_sim: float | None       # used for pre-generation refusal trigger

    # Generation
    draft_answer: str | None             # raw model output (last attempt) before verification
    cited_chunk_ids: list[str] | None
    verification_result: VerificationResult | None
    final_answer: str | None
    citations_stripped: int               # count from chunk-id verifier (existence check)
    # Inline LLM-judge stripping (Haiku per (claim, chunk) pair):
    sentences_stripped: int               # count of sentences fully removed for unsupported citations
    substantive_citations_stripped: int   # citations stripped from kept sentences
    judge_notes: list[dict] | None        # per-pair judge output, persisted for audit
    # Audit-log capture: the synthesis system prompt as actually rendered (with chunks
    # interpolated). Truncated to ~4KB at audit-write time to keep storage bounded.
    synthesize_prompt: str | None

    # Refusal state
    refusal_emitted: bool
    refusal_reason: RefusalReason | None

    # Bookkeeping
    regeneration_count: int
    timings: dict[str, int]
    model_ids_used: dict[str, str]   # e.g. {"classify": "claude-haiku-4-5", "synthesize": "claude-sonnet-4-7"}
    token_counts: dict[str, dict[str, int]]  # {"classify": {"in": 14, "out": 4}, ...}


def initial_state(query: str, user_id: str | None = None) -> GraphState:
    return GraphState(
        query=query,
        user_id=user_id,
        classification=None,
        classification_confidence=None,
        sub_queries=None,
        retrieved_chunks=None,
        top_cosine_sim=None,
        draft_answer=None,
        cited_chunk_ids=None,
        verification_result=None,
        final_answer=None,
        citations_stripped=0,
        sentences_stripped=0,
        substantive_citations_stripped=0,
        judge_notes=None,
        synthesize_prompt=None,
        refusal_emitted=False,
        refusal_reason=None,
        regeneration_count=0,
        timings={},
        model_ids_used={},
        token_counts={},
    )
