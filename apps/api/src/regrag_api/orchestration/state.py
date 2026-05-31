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

import uuid
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

Classification = Literal["single_doc", "multi_doc"]
RefusalReason = Literal[
    "no_relevant_chunks",      # pre-generation: top vector hit below threshold
    "unanswerable_from_corpus",  # answerability gate: chunks don't support the question as asked
    "llm_refusal",             # generation model returned refused=true
    "llm_unavailable",         # upstream LLM call failed (outage, rate limit, network)
    "verification_unrecoverable",  # unsupported/uncited content after the regen budget is exhausted
    "verification_unavailable",    # substantive judge outage / parse failure — verifier could not complete
    "pii_blocked",                 # query contained structured PII; refused at the handler before processing/logging/storage (#9)
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
    query_id: str   # stable id minted at the handler/graph entry (issue #7); links logs ↔ audit row
    # Identifiers extracted from the ORIGINAL query at state construction (issue #13):
    # deterministic, model-free metadata preserved so decomposition can't drop a
    # user-named order/docket/citation before retrieval/anchoring sees it. Serialized
    # ExtractedIdentifiers — {"orders": [...], "dockets": [...], "ferc_cites": [...], ...}.
    original_identifiers: dict[str, list[str]] | None

    # Stage outputs
    classification: Classification | None
    classification_confidence: float | None
    sub_queries: list[str] | None

    # Retrieval result — list of dicts (serializable, snapshot-friendly for audit log)
    retrieved_chunks: list[dict] | None
    top_cosine_sim: float | None       # used for pre-generation refusal trigger
    # Named-order metadata for downstream synthesis (review finding #9):
    # when the query names a specific order, surface the canonical accessions
    # so the synthesizer can prefer same-accession citations over chunks that
    # merely *reference* the named order.
    named_orders: list[str] | None
    anchored_accessions: list[str] | None
    # Role-split anchored accessions (issue #14): per named order, accessions grouped by
    # role so the scope verifier and SCOPE prompt don't treat a rehearing order as a
    # primary-order source. {order: {"primary": [...], "federal_register": [...], "rehearing": [...]}}.
    anchored_roles: dict[str, dict[str, list[str]]] | None

    # Generation
    draft_answer: str | None             # raw model output (last attempt) before verification
    cited_chunk_ids: list[str] | None
    verification_result: VerificationResult | None
    final_answer: str | None
    citations_stripped: int               # count from chunk-id verifier (existence check)
    # Accession-scope verifier (added 2026-05-27 per review finding #9 v3 follow-up):
    # tracks how many out-of-scope citations were stripped from sentences whose
    # subject is one of state.named_orders, and how many sentences were dropped
    # entirely because they had no in-scope citation.
    scope_citations_stripped: int
    scope_sentences_dropped: int
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
    # Answerability gate (flagged via REGRAG_ANSWERABILITY_GATE): when enabled,
    # a Haiku call between retrieve and synthesize judges whether the retrieved
    # chunks actually support the question as asked. Captured for audit/eval.
    answerability_checked: bool          # True if the gate ran (flag on + not already refused)
    answerability_verdict: bool | None   # the gate's answerable yes/no; None if it didn't run
    answerability_reason: str | None     # the gate's one-line rationale

    # Bookkeeping
    regeneration_count: int
    timings: dict[str, int]
    model_ids_used: dict[str, str]   # e.g. {"classify": "claude-haiku-4-5", "synthesize": "claude-sonnet-4-7"}
    token_counts: dict[str, dict[str, int]]  # {"classify": {"in": 14, "out": 4}, ...}


def initial_state(query: str, user_id: str | None = None, query_id: str | None = None) -> GraphState:
    # Local import keeps the state module free of a retrieval-package import at load time.
    from ..retrieval.identifiers import extract_identifiers

    return GraphState(
        query=query,
        user_id=user_id,
        query_id=query_id or str(uuid.uuid4()),
        original_identifiers=extract_identifiers(query).as_dict(),
        classification=None,
        classification_confidence=None,
        sub_queries=None,
        retrieved_chunks=None,
        top_cosine_sim=None,
        named_orders=None,
        anchored_accessions=None,
        anchored_roles=None,
        draft_answer=None,
        cited_chunk_ids=None,
        verification_result=None,
        final_answer=None,
        citations_stripped=0,
        scope_citations_stripped=0,
        scope_sentences_dropped=0,
        sentences_stripped=0,
        substantive_citations_stripped=0,
        judge_notes=None,
        synthesize_prompt=None,
        refusal_emitted=False,
        refusal_reason=None,
        answerability_checked=False,
        answerability_verdict=None,
        answerability_reason=None,
        regeneration_count=0,
        timings={},
        model_ids_used={},
        token_counts={},
    )
