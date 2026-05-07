# Critique: RegRAG Case Study Documents

A pre-shipping critique of the three working docs in this directory:

- `regrag-case-study.md` (CS)
- `diagrams and ingestion plan/ferc-ingestion-plan.md` (IP)
- `diagrams and ingestion plan/regrag-diagrams.md` (DG)

Findings are graded **High / Medium / Low**. Each cites file + section + the offending passage in quotes, then proposes a specific fix. The point is to surface what a senior RAG engineer or a careful hiring manager would catch, not to nitpick prose.

---

## High-severity

### H1. The case study describes empirical results from a system that doesn't exist

This is the single most credibility-destroying issue, and it appears in multiple places.

**CS §7:**
> "Haiku handles roughly 60% of the eval set's queries at a fraction of Sonnet's cost and latency, and the quality delta on those queries is negligible."

> "Naive fixed-token chunking was tried first and produced visibly worse retrieval on questions that referenced cross-section structure."

> "This adds infrastructure but materially improves retrieval recall on the identifier-heavy subset of the eval set."

**CS §5:**
> "In practice this catches a small but non-zero rate of citation drift — cases where the model summarizes the substance correctly but attaches the wrong order number or paragraph reference."

There is no system, no eval run, no measured citation drift. A reader who notices this — and any senior reviewer will — will discount the rest of the document.

**Fix options (pick one):**

1. **Reframe the case study as design-intent throughout.** Replace empirical claims with hypotheses: "We expect Haiku to handle the majority of queries — the ratio will be measured against the eval set"; "Naive fixed-token chunking is a known failure mode for cross-section regulatory text, which is why this design uses section-aware splitting." This is honest and still defensible.
2. **Actually build the demo and run the eval, then put real numbers in.** Higher bar but much stronger artifact.
3. **Worst option: leave the numbers in.** Don't.

The closing line — "Built by Kristen Martino. The architecture and patterns in this project draw on a prior production RAG system (Sift…)" — bills RegRAG as built. Either align this with reality (option 1) or make it true (option 2). The current state is the problem.

---

### H2. §9 "Outcomes and Artifacts" is empty placeholders in a section that's supposed to be the proof

**CS §9:**
> "- **Live demo:** [link to deployed FERC RAG instance]
> - **Source code:** [GitHub repository]
> - **Architecture diagram:** [link to high-resolution version]"

A reader following the writeup all the way to §9 expecting the payoff will land on five empty links. This compounds H1 — the document promises a system, then can't show it.

**Fix:** Until there are real artifacts, either (a) cut §9 entirely and end on §8 (which is the strongest section), (b) rename it "Targeted Outcomes" and reword each line as a planned deliverable rather than a live link, or (c) populate it once the demo ships. A reader cannot tell from the current draft which of these you intend.

---

### H3. Hybrid retrieval has no specified fusion algorithm

**CS §3:**
> "Hybrid search — vector similarity over the embedding space combined with keyword matching against docket numbers, order numbers, and statutory citations."

**CS §7:**
> "The keyword matches acting as a recall floor rather than a primary ranking signal."

**IP §6** has an HNSW index for vectors and a GIN index for `to_tsvector('english', chunk_text)`, but no fusion logic.

A senior RAG engineer will ask: reciprocal rank fusion? Weighted score sum? Re-ranking after union? The "recall floor" framing in CS §7 is suggestive but not concrete — does it mean keyword hits are always pulled in even when their vector score is low? With what cap?

This is a real engineering decision and the docs don't make it. It's also the decision most likely to determine whether retrieval works on identifier-heavy queries.

**Fix:** Add a paragraph in CS §3 (or §7) specifying the fusion approach. Recommended default: reciprocal rank fusion with k=60 over the union of top-N vector hits and top-M keyword hits. State the N, M, and the cap on keyword-only inclusions. Or pick a different approach and defend it.

---

### H4. The classifier is invoked everywhere but never specified

**CS §4:**
> "First, an inexpensive classifier model determines whether the query is a single-document lookup or a multi-document synthesis."

**CS §7:**
> "Misclassification toward the agentic path wastes resources, misclassification away from it produces worse answers on synthesis questions."

The classifier is the gating decision for the whole agentic layer. None of the docs say what it is — Haiku with a few-shot prompt? An embedding-based classifier? A fine-tuned small model? Some heuristic on query length and identifier presence?

**Fix:** Pick one and document it. Recommended: few-shot Haiku call with ~6 labeled examples in the system prompt, returning a JSON `{intent: "single" | "multi", confidence: 0–1}` schema. Cheap, fast, easy to test against the eval set, easy to swap if it underperforms.

---

### H5. The decomposition step needs to know what's in the corpus, but no doc says how it gets that

**CS §4:**
> "A decomposition step breaks the question into sub-queries — one per document, sub-topic, or comparison axis."

For the canonical query "compare DER treatment across Orders 2222, 841, and 745", the decomposer needs to know that those three orders exist in the corpus before it can produce per-document sub-queries. For a query like "what's FERC's recent guidance on transmission planning?" the decomposer needs a list of relevant documents (which it doesn't yet have — that's the retrieval step's job).

This is a real architectural question: is decomposition done with a corpus index passed in the prompt? Is it done with a retrieval pass first to identify candidate documents, then a per-document refinement? The docs don't say.

**Fix:** Specify the decomposition input. Recommended: pass a static "corpus summary" (list of order numbers + one-line summaries, ~30–50 lines) into the decomposer's system prompt. Refresh it when the corpus changes. This is cheap and explicit.

---

### H6. ~~Verify whether voyage-3-lite is actually 512-dimensional~~ — RESOLVED

**Status:** Verified during Phase B feasibility check. Voyage AI docs confirm `voyage-3-lite` is natively 512-dim (no Matryoshka truncation supported on this model — that's a 4-series and 3.5-series feature). The ingestion plan's `VECTOR(512)` schema and "at 512 dimensions" claim are both correct. No action needed.

**However**, a forward-looking note worth adding to IP §5: voyage-3-lite has been superseded by `voyage-3.5-lite` (May 2025), which supports flexible dimensions (256/512/1024/2048) at similar cost. For a fresh build today, voyage-3.5-lite at 512 dims gives equivalent index size with the option to upgrade dimensionality later. The choice between the two is essentially future-proofing vs. matching Sift's existing operational pattern. Worth a one-line acknowledgment in the doc.

---

## Medium-severity

### M1. The "small but non-zero rate of citation drift" framing assumes you've measured it

**CS §5:** see H1 quote. Even if you reframe this softer ("the verification step is designed to catch citation drift — cases where..."), you should still:
- Specify the verification mechanism: regex extraction of citations from the model output, lookup against the `chunk_id` set returned by retrieval, behavior on no-match.
- Specify the regeneration cap (DG §2 mentions "two regeneration attempts" but CS doesn't). Cite from DG into CS.

**Fix:** Add a paragraph to CS §5 specifying the citation-extraction regex anchor (e.g. citations are emitted as `[[chunk_id]]` or `(Order 2222 P 47)`) and the regeneration cap. Reference DG §2 inline.

---

### M2. The eval set's size and structure are unjustified

**CS §6:**
> "A small eval set of 28 hand-crafted question/answer pairs drives the measurement."

Why 28? It's an oddly specific number. With 4 personas, that's 7 per persona, which means each persona has limited statistical power for the three metrics. The OOS subset isn't sized.

**Fix:** Either justify the number ("28 = 4 personas × 5 in-scope + 4 personas × 2 out-of-scope, sized to be hand-authorable in 1 day and large enough that a 1-question regression is detectable") or change it to a round 30/40/50 and explain the per-persona / OOS split. State how many of the 28 are refusal-tests.

---

### M3. LLM-as-judge for citation faithfulness is presented uncritically

**CS §6:**
> "This is graded with a separate LLM-as-judge step against the eval set's expected passages."

LLM-as-judge has well-documented limitations: judge bias toward verbose responses, sensitivity to prompt wording, low agreement with human raters on subtle calls. For a regulated-domain pitch, glossing this is a missed opportunity to demonstrate awareness.

**Fix:** Add half a sentence: "...with the judge prompt versioned alongside the eval set, and a periodic spot-check by a human grader to calibrate judge drift." Or stronger: state the judge model (e.g. Sonnet) and that it sees only retrieved chunks, not the model's own response, when scoring substantive support.

---

### M4. The case study and diagram disagree about whether "synthesize" and "generate" are one stage or two

**CS §4:**
> "Fourth, a synthesis step generates the final answer with explicit attribution structure."

**DG §2 state diagram:**
> `Synthesize --> Generate` and `SingleDocRetrieve --> Generate`

The diagram has Generate as a separate node downstream of both synthesis and single-doc retrieval. The case study text reads as if synthesis IS the generation. Either is defensible (you might have one model do everything, or split synthesis-of-evidence from final-answer-formatting), but they should agree.

**Fix:** Decide which it is. If synthesis = generation, simplify DG §2 to remove the separate `Generate` node and route both branches to a single `Generate` node. If they're separate, update CS §4 to say "Fourth, a synthesis step organizes the per-sub-query evidence; fifth, a generation step produces the final answer; sixth, a verification step…" and renumber.

---

### M5. Audit log can't reconstruct historical retrievals if chunks change

**IP §6 schema:**
> `retrieved_chunk_ids TEXT[]` only stores chunk IDs; the chunk text and metadata live in `chunks`.

**CS §5:**
> "the retrieved chunks with their document and section provenance"

If you re-chunk (which IP §7 explicitly anticipates), old `retrieved_chunk_ids` entries point to nothing or to differently-shaped chunks. Audit-time replay breaks.

**Fix:** One of:
1. Snapshot the retrieved chunk text + key metadata into the `query_log` row at write time (denormalized, takes more storage but audit-stable).
2. Add a `chunker_version` column to `chunks` and never delete old chunks — re-chunking creates new versioned rows. Then `retrieved_chunk_ids` resolves correctly forever.
3. Acknowledge the limitation and document an immutability boundary ("chunks are not mutated post-write; chunker changes produce a new namespace").

Recommend option 2 with a `chunker_version` column added to the `chunks` schema in IP §6 and to the document-keying scheme in IP §7.

---

### M6. `chunker_version` is referenced but not in the schema

**IP §7:**
> "every artifact (parsed record, chunk set, embedding set) is keyed by both content hash and pipeline version"

**IP §6 schema** has `embedding_model TEXT NOT NULL` (which is one piece of pipeline version) but no `chunker_version` and no content_hash on chunks.

**Fix:** Add `chunker_version TEXT NOT NULL` and `chunk_content_hash TEXT NOT NULL` to the `chunks` table. Add the same to `documents` if not implied by `content_hash` already there.

---

### M7. Refusal behavior is claimed but the trigger is unspecified

**CS §5:**
> "When retrieval returns nothing relevant, the system says so. When the corpus is silent on the question, the system says so."

What's the threshold for "nothing relevant"? Top-1 vector similarity below a cutoff? Empty keyword hits? Both? "Corpus is silent" is a different condition from "retrieval returned weak hits" — the system needs different thresholds.

**Fix:** Specify in CS §5: the system refuses if (a) the top-K retrieved chunks all fall below a cosine similarity threshold T (recommended: T set per-percentile against the eval-set distribution, not a fixed value), or (b) the LLM, when shown the retrieved chunks, reports that they do not address the question (using a structured refusal flag in the output schema).

---

### M8. Two-week milestone schedule is optimistic for a solo build

**IP §8:** Days 1–2 verify source access; Days 3–4 fetch + parse all seed docs; Days 5–7 chunking + embedding; Days 8–9 hybrid retrieval + LangGraph + generation; Days 10–11 citation verification + audit + eval; Days 12–14 frontend + deploy + writeup polish.

Two days for hybrid retrieval + LangGraph orchestration + generation is very tight if you're learning LangGraph from scratch. Frontend chat UI + deployment in 1–2 days assumes you've shipped one before. PDF parsing in particular tends to absorb time disproportionately.

**Fix:** Either pad to 3 weeks (Days 8–10 retrieval + LangGraph; Days 11–13 verification + audit + eval; Days 14–17 frontend + deploy; Days 18–21 polish) or explicitly state which tasks reuse Sift's existing implementation (which would justify the speed). The plan currently reads like a fresh build at a fresh-build pace.

---

### M9. Footnote handling is hand-waved past a real problem

**IP §4:**
> "Footnotes: attach to the chunk containing the footnote reference rather than chunking separately."

Sounds clean, but FERC orders frequently have substantive footnotes (statutory citations, justifications, dissents). A 700-token chunk with three 200-token footnotes is now 1300 tokens, blowing the 800-token max.

**Fix:** Specify the strategy: footnote text is appended at the end of the chunk that references it, but if the resulting chunk exceeds the max token budget, the footnote becomes a separate chunk that links back to the parent chunk via a `parent_chunk_id` metadata field. This is more complex but matches reality.

---

## Low-severity / polish

### L1. Diagram placeholders are still in the case study

**CS §3:**
> "*[Architecture diagram placeholder: ingestion → transformation → embedding → retrieval → orchestration → generation → monitoring, with the technology choices labeled at each stage.]*"

**CS §4:**
> "*[LangGraph state diagram placeholder: classify → decompose → retrieve (parallel) → synthesize → verify.]*"

The diagrams already exist in DG. Either inline-embed them (if your portfolio renderer supports Mermaid), or replace placeholders with images + caption.

**Fix:** Replace each placeholder with the rendered diagram. If the portfolio site doesn't support Mermaid, render to SVG via mermaid.live (per DG's "Export and embedding notes") and embed as `<img>`.

---

### L2. "Recent" orders should be specified

**IP §1:**
> "**~15–20 adjacent recent orders** on capacity markets, transmission planning, and interconnection."

By 2026 standards, "recent" is ambiguous — post-2020? Post-2023? This matters because the corpus framing affects which questions are answerable.

**Fix:** Pin a date. Recommended: "issued 2018 or later" — captures the relevant doctrinal arc around DER, storage, and capacity reform.

---

### L3. The "polite scraping" detail is good but could note caching

**IP §2:**
> "Single-threaded fetches with a 1–2 second delay between requests, a clear `User-Agent` identifying the tool and a contact email"

Worth adding: cache fetched PDFs locally (under `corpus/raw/` per the layout in IP §3) and skip re-fetch on the same accession number. IP §3 actually mentions this ("Skip if already fetched") so the deduplication is in §3 — but §2's politeness section is a natural place to also mention it.

**Fix:** Add a sentence in IP §2: "Cache aggressively — once a document is fetched, never re-fetch unless its content hash changes."

---

### L4. `chunks_text_search_idx` uses default English stemming

**IP §6 schema:**
> `CREATE INDEX chunks_text_search_idx ON chunks USING gin (to_tsvector('english', chunk_text));`

The English stemmer will stem "Order 2222" oddly (no effect on the number, fine) and may strip stop-word-like tokens that matter in a regulatory context (e.g. "may", "shall" are key modal verbs in regs but Postgres `english` config may treat them as stop words).

**Fix:** Either use `'simple'` config for the GIN index (no stemming, no stop-word removal — closer to literal matching, which is what you want for identifiers), or build a custom text search config that preserves modal verbs. The simple config is the lower-effort choice.

---

### L5. `latency_ms_total INT` is fine but `BIGINT` is safer

**IP §6 schema:**
> `latency_ms_total INT NOT NULL,`

INT is 32-bit, max ~24 days in ms. Plenty for a single query latency. But if you ever sum it for analytics or accidentally use it for a long-running job duration, you'd overflow. BIGINT is one byte more per row and saves a class of bug.

**Fix:** Change `INT` to `BIGINT`. Trivial.

---

### L6. The Mermaid diagram's audit log writes are dotted from G/I/J but not H

**DG §1 architecture diagram:**
```
G -.-> L
I -.-> L
J -.-> L
```

The orchestrator (G), generation (I), and verification (J) write to the audit log. The retrieval node (H) does not — but the case study §5 explicitly says "the retrieved chunks with their document and section provenance" are logged. Retrieval should also dot-line into L, or the orchestrator should be the one that logs retrieval results.

**Fix:** Add `H -.-> L` to the diagram, or update CS §5 to clarify that retrieval results are logged by the orchestrator wrapping the retrieval call.

---

### L7. Closing line markets RegRAG as built

**CS closing line:**
> "Built by Kristen Martino. The architecture and patterns in this project draw on a prior production RAG system (Sift, [siftnews.kristenmartino.ai](https://siftnews.kristenmartino.ai))…"

Until the demo ships, "built" is overclaiming. See H1.

**Fix:** Change to "Designed by Kristen Martino" (if going with the design-intent reframe) or "Built by Kristen Martino" once it's actually built.

---

## Summary of recommended actions

| # | Action | Owner | Effort |
|---|--------|-------|--------|
| H1 | Reframe empirical claims as design intent OR build the demo | author | 1h reframe / weeks build |
| H2 | Fix §9 outcomes section | author | 30 min |
| H3 | Specify hybrid fusion algorithm | author + design call | 1h |
| H4 | Specify the classifier implementation | author + design call | 1h |
| H5 | Specify decomposition's corpus context input | author + design call | 1h |
| H6 | ~~Verify voyage-3-lite dimensionality~~ — resolved (doc is correct); optionally note voyage-3.5-lite as upgrade path | author | 5 min |
| M1 | Specify citation verification mechanism | author + design call | 1h |
| M2 | Justify or restructure the eval set size | author | 30 min |
| M3 | Acknowledge LLM-as-judge limitations | author | 15 min |
| M4 | Reconcile synthesize-vs-generate split between CS and DG | author | 15 min |
| M5 | Decide audit-log immutability strategy | author + design call | 1h |
| M6 | Add `chunker_version` to schema | author | 15 min |
| M7 | Specify refusal triggers | author | 30 min |
| M8 | Pad milestone schedule or call out reused Sift code | author | 30 min |
| M9 | Specify footnote chunking edge case | author | 30 min |
| L1–L7 | Polish | author | 1–2h total |

Total reframe-and-tighten effort, no demo build: **~8 hours**. Most of the high-severity items are short writing fixes once the design questions (H3, H4, H5, M1, M5) have decisions made.
