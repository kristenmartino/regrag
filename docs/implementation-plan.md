# RegRAG Implementation Plan

A concrete build plan for the system described in `regrag-case-study.md`. Supersets `ferc-ingestion-plan.md` (which already covers the corpus pipeline well) by filling in the retrieval, orchestration, generation, eval, frontend, and deployment pieces the existing docs hand-wave or omit. Reflects the corrections from `critique.md` and the verified facts in `feasibility-notes.md`.

---

## 0. Context

The case study describes a system that doesn't exist yet. The ingestion plan is detailed for the corpus pipeline but stops at "the corpus is in Postgres, indices are built." Several design decisions the case study makes claims about — the classifier, the hybrid fusion algorithm, the citation verifier, the refusal trigger, the eval harness — are not specified anywhere.

This plan picks the unspecified decisions, calls them as decisions (not as ambiguities), and lays out a 3-week build that produces a working demo plus the case study artifacts (live URL, GitHub repo, eval results) that section 9 of the case study currently has as empty placeholders.

It is built for a solo developer who has shipped Sift (the prior production RAG system referenced throughout the case study) — assumes familiarity with Postgres+pgvector, OpenAI/Anthropic SDKs, and Vercel/Railway/Neon deployment. Does not assume prior LangGraph experience.

---

## 1. Stack and repo decisions

### 1.1 Languages and frameworks

| Layer | Choice | Why |
|-------|--------|-----|
| Backend / pipeline / orchestration | **Python 3.12** | LangGraph, pdfplumber, pyvoyage, openai/anthropic SDKs are all Python-native. Sift uses Python — operational pattern reuse. |
| API layer | **FastAPI** | Lightweight, async-native, fits LangGraph's async invocation model. Auto-generates OpenAPI for the audit log query interface. |
| Frontend | **Next.js 15 (App Router) + Vercel AI SDK** | Streaming responses out of the box, built-in chat UI primitives via `useChat`, deploys to Vercel cleanly. |
| Database | **Neon Postgres + pgvector** | As specified in the case study. Free tier covers the seed corpus easily (per `feasibility-notes.md` §B4). |
| Embedding | **voyage-3.5-lite at 512 dims** | Per `feasibility-notes.md` §A4 — newer than 3-lite (May 2025), supports flexible dimensions, same effective cost. Forward-compat over matching Sift exactly. |
| Generation | **Claude Sonnet 4.7 + Haiku 4.5** (per current model IDs `claude-sonnet-4-7`, `claude-haiku-4-5`) | Routing per case study §7. |

### 1.2 Repo structure

Single monorepo, two top-level apps + shared packages:

```
regrag/
├── apps/
│   ├── api/                       # FastAPI service
│   │   ├── pyproject.toml
│   │   ├── src/regrag_api/
│   │   │   ├── main.py            # FastAPI app, /chat endpoint, /audit endpoint
│   │   │   ├── orchestration/     # LangGraph workflow
│   │   │   │   ├── graph.py       # state machine wiring
│   │   │   │   ├── state.py       # typed state schema
│   │   │   │   ├── nodes/         # one file per node (classify, decompose, retrieve, synthesize, generate, verify)
│   │   │   │   └── prompts/       # system prompts kept versioned in code
│   │   │   ├── retrieval/
│   │   │   │   ├── hybrid.py      # vector + keyword fusion
│   │   │   │   ├── identifiers.py # FERC identifier regex set (Order #s, dockets, statute cites)
│   │   │   │   └── client.py      # pgvector + tsquery client
│   │   │   ├── verification/
│   │   │   │   └── citations.py   # citation extraction + verification
│   │   │   ├── audit/
│   │   │   │   └── log.py         # append-only writes to query_log
│   │   │   └── models/            # pydantic schemas
│   │   └── tests/
│   ├── web/                       # Next.js chat UI
│   │   ├── package.json
│   │   ├── app/
│   │   │   ├── page.tsx           # main chat surface
│   │   │   ├── api/chat/route.ts  # proxies to api service
│   │   │   └── audit/page.tsx     # read-only audit log viewer (demo-only)
│   │   └── components/
│   └── ingest/                    # corpus pipeline (one-shot + cron)
│       ├── pyproject.toml
│       ├── src/regrag_ingest/
│       │   ├── manifest.py        # load corpus/manifest.yaml
│       │   ├── fetch.py           # download PDFs from www.ferc.gov
│       │   ├── parse.py           # pdfplumber-based extraction
│       │   ├── chunk.py           # section-aware chunker
│       │   ├── embed.py           # voyage batch embedding
│       │   ├── load.py            # write to documents + chunks
│       │   └── cli.py             # `regrag-ingest run --manifest ...`
│       └── tests/
├── packages/
│   └── eval/                      # eval harness, shared between local + CI
│       ├── pyproject.toml
│       ├── src/regrag_eval/
│       │   ├── runner.py          # walks the eval set, calls /chat, scores
│       │   ├── metrics.py         # retrieval recall, citation faithfulness, refusal accuracy
│       │   ├── judge.py           # LLM-as-judge for citation faithfulness
│       │   └── eval_set.yaml      # the 28 questions
│       └── tests/
├── corpus/
│   ├── manifest.yaml              # the seed list (per feasibility §C)
│   ├── raw/                       # PDFs, gitignored, content-addressed
│   └── parsed/                    # JSON records, gitignored
├── infra/
│   ├── neon-schema.sql            # the Postgres schema (IP §6 + chunker_version per critique M6)
│   ├── railway.json               # api + ingest deployment
│   ├── vercel.json                # web deployment
│   └── env.example
├── docs/
│   ├── regrag-case-study.md       # symlinked or copied from this repo
│   ├── ferc-ingestion-plan.md
│   ├── regrag-diagrams.md
│   └── critique.md / feasibility-notes.md / implementation-plan.md
├── .github/workflows/
│   ├── ci.yml                     # lint + tests + eval-on-PR
│   └── ingest-cron.yml            # weekly discovery scrape
└── README.md
```

The `apps/ingest` and `apps/api` could live in one Python package, but separating them keeps the cron job (ingest) deployable independently of the live API.

### 1.3 Decisions called as decisions

These were left ambiguous in the existing docs. Picking now to avoid rework.

| Decision | Pick | Reasoning |
|----------|------|-----------|
| Backend language | Python | LangGraph is Python-native; Sift reuse |
| Frontend framework | Next.js | Streaming chat UX is mature; Vercel deploy |
| LangGraph or hand-rolled state machine | **LangGraph** | The case study sells "LangGraph orchestration" as a feature; commit to it |
| Embedding model | **voyage-3.5-lite @ 512** | Newer than 3-lite, same cost, future-proof |
| Generation models | **claude-sonnet-4-7 (synthesis) / claude-haiku-4-5 (single-doc + classifier)** | Latest stable as of build time |
| PDF parser | **pdfplumber** primary, **pymupdf** fallback for layouts pdfplumber chokes on | pdfplumber is gentler on multi-column; pymupdf is faster and often produces cleaner text — keep both, pick per-doc if needed |
| Chunk store schema additions | Add `chunker_version TEXT NOT NULL`, `chunk_content_hash TEXT NOT NULL` | Per critique M5/M6 |
| Retrieval fusion | **Reciprocal rank fusion (k=60)** over union of top-20 vector + top-10 keyword | Standard RRF; tune `k` against eval set |
| Classifier | **Few-shot Haiku call**, JSON output `{intent, confidence}` | Per critique H4 |
| Decomposer corpus context | **Static corpus summary** (~30 lines) injected into decomp prompt; refreshed on corpus change | Per critique H5 |
| Citation format | **`[[chunk_id]]` markers in model output**, extracted via regex post-generation | Stable, parseable, verifiable |
| Regeneration cap | **2 attempts** (matches DG §2 detail) | Per critique M1 |
| Refusal trigger | Top-1 cosine < 0.55 OR LLM-emitted refusal flag | Per critique M7 |
| Hosting | **Vercel** (web) + **Railway** (api + ingest cron) + **Neon** (db) | As planned in case study; minimizes net-new infra |
| Auth on demo | **Single-tenant, no auth** for the demo; clearly labeled as "demo only" | The case study §8 acknowledges enterprise SSO is out of scope for demo |

---

## 2. Component design specs

### 2.1 Corpus pipeline (mostly reuses `ferc-ingestion-plan.md`)

Use `ferc-ingestion-plan.md` §3 as the source of truth, with these changes from `feasibility-notes.md` §E:

- **Discovery (Stage 1):** for the seed load, read `corpus/manifest.yaml` directly. No eLibrary scraping needed.
- **Fetch (Stage 2):** PDF URLs come from the manifest, not from a derivable accession→URL pattern. Polite scrape (1–2s delay) against `www.ferc.gov/sites/default/files/`.
- **Parse (Stage 3):** pdfplumber primary; budget Day 1 for fidelity check on `Order-841.pdf`.
- **Normalize (Stage 4):** unchanged.

Updated `manifest.yaml` schema:

```yaml
documents:
  - accession_number: "20200917-3084"  # fabricated example, verify before commit
    order_number: "2222"
    docket_numbers: ["RM18-9-000"]
    document_type: "order"
    issue_date: "2020-09-17"
    title: "Participation of Distributed Energy Resource Aggregations in Markets Operated by Regional Transmission Organizations and Independent System Operators"
    pdf_url: "https://www.ferc.gov/sites/default/files/2020-09/E-1_0.pdf"
    notes: "Final order; Docket also includes earlier NOPR"
  - ...
```

### 2.2 Storage schema (revised from IP §6)

Apply critique M5 + M6:

```sql
-- documents: unchanged from IP §6

-- chunks: add chunker_version, chunk_content_hash
CREATE TABLE chunks (
    chunk_id TEXT PRIMARY KEY,                  -- e.g. "20200917-3084:S3:P47-52"
    accession_number TEXT NOT NULL REFERENCES documents(accession_number),
    section_heading TEXT,
    paragraph_range TEXT,
    chunk_text TEXT NOT NULL,
    chunk_content_hash TEXT NOT NULL,           -- NEW: stable identity across re-chunks
    embedding VECTOR(512),
    embedding_model TEXT NOT NULL,
    chunker_version TEXT NOT NULL,              -- NEW: which chunker produced this
    chunk_index INT NOT NULL,
    parent_chunk_id TEXT REFERENCES chunks(chunk_id),  -- NEW: for footnote chunks (critique M9)
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX chunks_accession_idx ON chunks (accession_number);

-- Use 'simple' tokenizer per critique L4 to preserve modal verbs & identifiers
CREATE INDEX chunks_text_search_idx ON chunks USING gin (to_tsvector('simple', chunk_text));

-- query_log: change INT to BIGINT per critique L5; snapshot retrieved chunks per critique M5
CREATE TABLE query_log (
    query_id UUID PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    user_id TEXT,
    raw_query TEXT NOT NULL,
    classification TEXT,                        -- 'single_doc' | 'multi_doc'
    sub_queries JSONB,
    retrieved_chunks JSONB NOT NULL,            -- CHANGED: full snapshot {chunk_id, accession, section, text}
    prompt_sent TEXT NOT NULL,
    model_id TEXT NOT NULL,
    raw_response TEXT NOT NULL,
    verified_response TEXT NOT NULL,
    citations_stripped INT DEFAULT 0,
    refusal_emitted BOOLEAN NOT NULL DEFAULT FALSE,    -- NEW
    refusal_reason TEXT,                              -- NEW: 'no_relevant_chunks' | 'llm_refusal' | etc.
    latency_ms_total BIGINT NOT NULL,                 -- CHANGED to BIGINT
    latency_ms_by_stage JSONB,
    token_counts JSONB
);

CREATE INDEX query_log_timestamp_idx ON query_log (timestamp DESC);
```

### 2.3 Hybrid retrieval (the missing spec from critique H3)

```
def hybrid_retrieve(query: str, k: int = 10) -> list[Chunk]:
    # 1. Identifier extraction — explicit boost path
    identifiers = extract_ferc_identifiers(query)
        # regex set: r"Order\s+(?:No\.?\s+)?(\d{2,4})", r"docket\s+([A-Z]{2}\d{2}-\d{1,3}(?:-\d{3})?)",
        #            r"(\d{1,3})\s+FERC\s+¶\s+(\d{2},\d{3})", r"\d{1,2}\s+U\.S\.C\.\s+§\s+\d+"
    
    # 2. Vector search: top-20 by cosine similarity
    vector_hits = pgvector_topk(embed(query), k=20)
    
    # 3. Keyword search: top-10 by ts_rank, with identifier-match boost
    keyword_query = build_tsquery(query, identifiers)  # boost terms matching identifiers
    keyword_hits = tsquery_topk(keyword_query, k=10)
    
    # 4. RRF fusion: score = sum(1 / (k_const + rank_in_list)) across both lists
    K_CONST = 60
    fused = reciprocal_rank_fusion(vector_hits, keyword_hits, k_const=K_CONST)
    
    # 5. Identifier-match floor: any chunk that contains an exact identifier from the query
    #    is guaranteed to be in the result set, even if it doesn't beat the score threshold
    floor_chunks = chunks_containing_identifiers(identifiers)
    
    return dedupe(fused[:k] + floor_chunks)
```

The "recall floor" framing from CS §7 is implemented as the explicit `floor_chunks` step: queries that name an identifier always get the matching chunks, regardless of vector or keyword scores. Cap floor at 5 chunks to bound result-set size.

### 2.4 LangGraph orchestration

State schema (one TypedDict, all fields nullable until populated):

```python
class GraphState(TypedDict):
    query: str
    user_id: str | None
    classification: Literal["single_doc", "multi_doc"] | None
    classification_confidence: float | None
    sub_queries: list[str] | None        # populated only on multi_doc path
    retrieved_chunks: list[Chunk] | None
    draft_answer: str | None
    cited_chunk_ids: list[str] | None    # parsed out of draft_answer
    verification_result: VerificationResult | None
    final_answer: str | None
    refusal_emitted: bool
    refusal_reason: str | None
    regeneration_count: int              # bounded at 2
    timings: dict[str, int]              # ms per stage, accumulated
```

Nodes (one Python module each in `apps/api/src/regrag_api/orchestration/nodes/`):

- `classify(state) → state` — calls Haiku with few-shot prompt, sets `classification` + `classification_confidence`.
- `decompose(state) → state` — multi-doc path only; calls Sonnet with corpus summary in context, emits `sub_queries`.
- `retrieve_single(state) → state` — runs hybrid retrieval once on `query`.
- `retrieve_parallel(state) → state` — runs hybrid retrieval in parallel for each sub_query, unions results.
- `synthesize(state) → state` — Sonnet writes a draft answer with `[[chunk_id]]` markers.
- `verify(state) → state` — extracts citations, checks each against retrieved chunks, sets verification_result.
- `regenerate_or_finalize(state) → next_node` — routing function: if verification has unresolvable citations and `regeneration_count < 2`, increment and route to `synthesize`; else strip bad citations and finalize.

Edges:

```python
graph.add_edge(START, "classify")
graph.add_conditional_edges("classify", route_by_intent, {
    "single_doc": "retrieve_single",
    "multi_doc": "decompose",
})
graph.add_edge("decompose", "retrieve_parallel")
graph.add_edge("retrieve_single", "synthesize")
graph.add_edge("retrieve_parallel", "synthesize")
graph.add_edge("synthesize", "verify")
graph.add_conditional_edges("verify", regenerate_or_finalize, {
    "regenerate": "synthesize",
    "finalize": END,
})
```

Resolves critique M4 (synthesize-vs-generate split): collapsed to a single `synthesize` node that produces the final draft answer with citation markers. Update DG §2 to match.

### 2.5 Classifier (critique H4 spec)

Few-shot Haiku, returns JSON:

```python
SYSTEM_PROMPT = """
You classify FERC regulatory queries as either single-document lookups or multi-document synthesis tasks.

single_doc: the answer comes from one specific FERC order, ruling, or filing. Examples:
- "What does Order 2222 require for DER aggregation reporting?"
- "Summarize the dissent in Order 745"
- "What's the deadline for compliance with Order 841?"

multi_doc: the answer requires comparing, synthesizing, or evolving across multiple FERC documents. Examples:
- "How has FERC's treatment of capacity market participation evolved across recent rulings?"
- "Compare DER treatment across Orders 2222, 841, and 745"
- "Summarize the public comments on the Order 2222 NOPR and FERC's responses"

Return JSON: {"intent": "single_doc" | "multi_doc", "confidence": 0.0-1.0}
""".strip()

def classify(query: str) -> tuple[Literal["single_doc","multi_doc"], float]:
    response = anthropic.messages.create(
        model="claude-haiku-4-5",
        max_tokens=64,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": query}],
    )
    parsed = json.loads(response.content[0].text)
    return parsed["intent"], parsed["confidence"]
```

Acceptance: 90%+ accuracy on the eval set's 28 labeled questions, measured as part of the eval run.

### 2.6 Decomposer (critique H5 spec)

Sonnet call with a static corpus summary in the system prompt:

```python
CORPUS_SUMMARY = """
The corpus contains the following FERC orders and rules (refresh on corpus change):

- Order 2222 (RM18-9-000, 2020-09-17): DER aggregation participation in wholesale markets
- Order 841 (RM16-23-000, 2018-02-15): Electric storage participation in RTO/ISO markets
- Order 745 (RM10-17-000, 2011-03-15): Demand response compensation in wholesale markets
- [...the rest of the ~30–50 documents...]
""".strip()

DECOMP_SYSTEM_PROMPT = f"""
{CORPUS_SUMMARY}

You decompose multi-document FERC regulatory queries into focused sub-queries — one per document, sub-topic, or comparison axis. Sub-queries should be self-contained and retrievable independently.

Return JSON: {{"sub_queries": ["...", "..."]}}

Examples:
Query: "Compare DER treatment across Orders 2222, 841, and 745"
→ {{"sub_queries": ["What does Order 2222 say about DER aggregation?", "What does Order 841 say about distributed storage participation?", "What does Order 745 say about demand response?"]}}
""".strip()
```

Refresh `CORPUS_SUMMARY` whenever `corpus/manifest.yaml` changes; generate it as part of the ingest cron job and write it to a file the API reads on startup.

### 2.7 Citation verification (critique M1 spec)

```python
CITATION_PATTERN = re.compile(r"\[\[(?P<chunk_id>[^\]]+)\]\]")

def verify(draft_answer: str, retrieved_chunk_ids: set[str]) -> VerificationResult:
    cited = set(m.group("chunk_id") for m in CITATION_PATTERN.finditer(draft_answer))
    valid = cited & retrieved_chunk_ids
    invalid = cited - retrieved_chunk_ids
    
    if not invalid:
        return VerificationResult(action="finalize", citations_stripped=0, valid=valid)
    
    if regeneration_count < 2 and len(invalid) / len(cited) > 0.3:
        # significant drift: regenerate with stronger grounding instruction
        return VerificationResult(action="regenerate", reason="citation_drift")
    
    # final attempt: strip invalid citations from text
    cleaned = CITATION_PATTERN.sub(
        lambda m: f"[[{m.group('chunk_id')}]]" if m.group("chunk_id") in valid else "",
        draft_answer
    )
    return VerificationResult(action="finalize", citations_stripped=len(invalid), text=cleaned)
```

### 2.8 Refusal triggers (critique M7 spec)

Two paths:

1. **Pre-generation refusal:** if `max(cosine_similarity)` of top-20 vector hits is below 0.55, skip to a refusal response without invoking the generation model. The cutoff is calibrated against the eval set's distribution at the 25th percentile of the in-scope answers — so questions clearly outside the corpus's vector neighborhood don't waste tokens.
2. **Post-generation refusal:** the generation model is instructed to emit `{"refused": true, "reason": "..."}` as a structured tag if the retrieved chunks don't address the question. Detected by parser; flips `refusal_emitted` in state.

Both paths populate `refusal_reason` for the audit log.

### 2.9 Eval harness (case study §6 → concrete spec)

`packages/eval/src/regrag_eval/eval_set.yaml`:

```yaml
questions:
  # Compliance analyst persona — single-doc, in-scope (5 questions)
  - id: compliance-001
    persona: compliance_analyst
    expected_behavior: answer
    query: "What reporting requirements does Order 2222 create for DER aggregation programs, and what's the deadline?"
    expected_documents: ["20200917-3084"]
    expected_passages_keywords: ["reporting", "compliance filing", "tariff", "180 days"]
  # ... 4 more compliance, 5 counsel, 5 federal staff, 5 policy researcher
  # Out-of-scope (8 questions, 2 per persona)
  - id: oos-001
    persona: compliance_analyst
    expected_behavior: refuse
    query: "What is FERC's position on residential rooftop solar permitting?"
    expected_refusal_reason: "outside_corpus"
  # ... 7 more OOS
  # Total: 5 × 4 + 2 × 4 = 28 questions
```

Three metrics:

1. **Retrieval recall** — for each `answer`-expected question, fraction of `expected_passages_keywords` present in retrieved chunks. Macro-averaged across questions.
2. **Citation faithfulness** — LLM-as-judge (Sonnet) sees: query, model response, retrieved chunks. Returns 0/1 per cited claim ("does this chunk substantively support this claim?"). Macro-averaged.
3. **Refusal accuracy** — for each `refuse`-expected question, did the system emit a refusal? Binary, averaged.

Judge prompt versioned in `packages/eval/src/regrag_eval/judge.py` (per critique M3). Periodic spot-check by the human author against 5 samples per quarter to calibrate judge drift.

The eval set is checked into git. The eval results are logged into `query_log` (per CS §6 closing line) so eval history is itself audited.

### 2.10 Frontend

Single-page Next.js app with two routes:

- `/` — chat surface. Streams responses from `/api/chat`, which proxies to the FastAPI `/chat` endpoint with SSE. Uses Vercel AI SDK's `useChat` hook for the streaming primitive. Below each assistant turn, render the cited chunks as expandable cards (chunk text + accession + section heading + link to source PDF).
- `/audit` — read-only audit log viewer. Lists recent queries from `query_log` with their classification, retrieval count, citation count, and pass/refusal status. Click into a row to see the full record (query, sub_queries, retrieved_chunks snapshot, prompt, raw response, verified response). Demo-only; production would put this behind RBAC.

### 2.11 Audit log query interface

The `/audit` route is intentionally a thin read-only viewer. For an actual auditor, expose a SQL endpoint via Neon's built-in SQL editor or a separate Metabase/Hex instance pointed at read-only credentials. Document the schema in `infra/neon-schema.sql` and link from the case study §5.

### 2.12 Deployment

- **Vercel** — `apps/web`. Standard Next.js deploy. Env vars: `API_URL` pointing to Railway service.
- **Railway** — `apps/api` (long-running) + `apps/ingest` (one-shot job + weekly cron). Env vars: `NEON_DATABASE_URL`, `VOYAGE_API_KEY`, `ANTHROPIC_API_KEY`. Two services, one Railway project.
- **Neon** — Postgres + pgvector. Free tier OK at corpus size. Branch for staging.
- **Domain** — `regrag.kristenmartino.ai` per the Sift/GridPulse pattern.
- **Secrets** — All in Vercel/Railway env, mirrored in `infra/env.example` for local dev.

---

## 3. Phased build plan

Three weeks. Reframes IP §8's two-week milestone schedule to be honest about LangGraph + frontend learning curves (per critique M8).

### Week 1 — Corpus pipeline + retrieval

- **Day 1 (half day):** PDF parsing fidelity check. Run pdfplumber against `Order-841.pdf` and `Order-2222.pdf`. Inspect: text quality, section heading detection, footnote handling. Decide pdfplumber-only vs. fallback to pymupdf for some docs. *(Per `feasibility-notes.md` §B2 — this replaces IP §8's two-day source-verification block, since source access is already verified.)*
- **Day 1 (half day):** Author `corpus/manifest.yaml` for the three named orders + 10 adjacent recent orders. Pin "recent" to 2018-or-later (per critique L2).
- **Days 2–3:** Build `apps/ingest`. End-of-day-3 acceptance: `regrag-ingest run --manifest corpus/manifest.yaml` fetches all 13 PDFs, parses them, writes JSON records to `corpus/parsed/`.
- **Days 4–5:** Chunker + embedding. End-of-day-5 acceptance: chunks loaded into Postgres with embeddings, vector and keyword indices built, a manual `SELECT chunk_text FROM chunks ORDER BY embedding <=> embed('Order 2222 reporting requirements') LIMIT 5` returns sensible chunks.
- **Days 6–7:** Hybrid retrieval implementation (`apps/api/src/regrag_api/retrieval/`) + a CLI-callable `regrag-retrieve "query"` for sanity-checking. End-of-week-1 acceptance: hybrid retrieval returns relevant chunks for 5 hand-picked queries (3 single-doc, 2 multi-doc).

### Week 2 — Orchestration + verification + eval

- **Days 8–10:** LangGraph workflow (`apps/api/src/regrag_api/orchestration/`). End-of-day-10 acceptance: end-to-end `/chat` endpoint takes a query, runs through classify→decompose(if needed)→retrieve→synthesize→verify, returns an answer with citations.
- **Day 11:** Citation verification + refusal trigger (`apps/api/src/regrag_api/verification/`). End-of-day acceptance: citations that don't match retrieved chunks are stripped or trigger regen; out-of-scope queries return refusals.
- **Days 12–13:** Eval harness + author the 28-question eval_set.yaml. End-of-day-13 acceptance: `regrag-eval run` produces a report with retrieval recall, citation faithfulness, refusal accuracy. Not yet polished.
- **Day 14:** Audit logging end-to-end. Every `/chat` invocation writes a `query_log` row. Every `regrag-eval run` invocation also writes rows (tagged `user_id = "eval-runner"`).

### Week 3 — Frontend, deploy, polish

- **Days 15–16:** Next.js chat UI (`apps/web/`). End-of-day-16 acceptance: chat surface running locally, streaming responses, citation cards rendering.
- **Day 17:** Audit viewer (`/audit` route). End-of-day acceptance: list view + detail view working against query_log.
- **Days 18–19:** Deploy. Vercel + Railway + Neon production setup, env vars wired, domain pointed. End-of-day-19 acceptance: `regrag.kristenmartino.ai` is live, you can ask it a question and get an answer.
- **Days 20–21:** Case study polish. Apply the critique fixes. Replace placeholders in CS §3, §4 with rendered diagrams. Populate CS §9 outcomes with real links. Run final eval, write the numbers into CS §6 / §7. Re-record any claims-formerly-known-as-empirical now that they're real.

---

## 4. What stays out of scope for the demo

Per case study §8 (which is well-framed). Don't sneak these in:

- Authentication / SSO / RBAC. Demo is single-tenant, no auth, clearly labeled as such.
- FedRAMP hosting. The demo is a portfolio piece on Vercel/Railway/Neon.
- Human-in-the-loop review queue. The demo returns answers directly; production would queue them.
- Continuous corpus updates with replay-against-historical-corpus. The demo loads once + weekly cron; replay is a production concern.
- Integration with case management / document management systems. None.
- Agency-specific terminology fine-tuning. FERC-only.

These are listed in case study §8 already. Re-list them in the demo's `/about` page or repo README so the gap between demo and production stays visible.

---

## 5. Open decisions still to make during execution

These can wait until the relevant phase:

| Decision | When to decide | Default if undecided |
|----------|----------------|----------------------|
| Reranker yes/no | After Day 13 eval | No — add only if recall < 0.7 |
| Chunk size tuning (currently 800 max / 200 min target) | After Day 13 eval | Keep defaults |
| Chunker edge cases for footnotes (per critique M9) | Day 4–5 during chunker build | Footnotes appended to ref'ing chunk; if exceeds max, separate chunk with `parent_chunk_id` |
| Eval set growth beyond 28 | After demo ships | 28 stays; expand on first regression |
| Whether to migrate from voyage-3-lite to 3.5-lite mid-project | Day 5 | Start on 3.5-lite from the beginning |

---

## 6. Verification — how to know this plan worked

End-of-Week-3 success criteria, in order of importance:

1. **A user can visit `regrag.kristenmartino.ai`, ask "compare DER treatment across Orders 2222, 841, and 745", and get a multi-paragraph answer with verified citations to the three orders.** That's the canonical demo flow.
2. **Eval results are populated in case study §6 and §7 with real numbers.** No more "60%" without a measurement.
3. **The audit log shows 100+ query rows by ship time** (from eval runs + manual testing). Visible at `/audit`.
4. **All HIGH-severity items in `critique.md` are addressed**, either by the implementation (H3, H4, H5) or by reframing the case study text (H1, H2, H7).
5. **The case study, ingestion plan, and diagrams agree** with the as-built system. Cross-doc consistency check (per critique lens 2) passes.

If 1–4 are true and 5 isn't, the system shipped but the docs lag — fix the docs in a follow-up. If 1 isn't true, ship is delayed; everything else is moot.

---

## 7. After this plan

Once the demo is live and the case study is polished:

- **Add 2 more domain corpora** for cross-domain comparison (e.g., a NERC reliability standards corpus, a state PUC orders corpus). Demonstrates that the system isn't FERC-specific.
- **Build a small "side-by-side" eval** against a baseline RAG (no agentic layer, no verification) to show the agentic + verification work earns its complexity.
- **Write a follow-up post**: "What I'd change for a federal deployment" — turn case study §8 into a separate document with concrete tech picks (FedRAMP-eligible cloud, FedRAMP-eligible LLM provider options, etc.).

These are post-demo polish, not part of this plan.
