# Review follow-ups

Findings from the multi-perspective review of the deliverable on 2026-05-22. Items shipped same-day are at the top; deferred items below with severity, scope, and what we'd want to talk through before doing each one.

---

## Shipped same-day (commit batch following the review)

| # | What | Fix |
|---|---|---|
| 1 | README "Status: in development" | Replaced with live-demo + eval link block at the top |
| 2 | README missing live-demo link | Added at top |
| 3 | README's local-setup `uvicorn main:app` was wrong | Fixed to `server:app`; also added `.env` setup, `pnpm dev` for frontend, `regrag-eval run` to quickstart |
| 19 | `/health` returned 405 on HEAD | Changed `@app.get` → `@app.api_route(methods=["GET","HEAD"])` |
| 24 | Chat UI empty-state listed only 5 of 15 corpus docs | Updated to enumerate all 15 by rulemaking arc |
| 32 | No "demo only / not legal advice" disclaimer | Added an amber card to the chat empty state |
| 25 | CS §9 said "28-question harness" | → "40-question harness + baselines" |
| 14 | Δ-vs-v3 column in eval-results.md was apples-to-oranges (different test sets) | Removed Δ column; replaced with prose acknowledging the v3 reference is for shape, not comparison |
| 18 | GitHub Actions cron actually runs every 30-60 min, not 5 | Documented in keep-warm.yml header; explicitly named the frontend pre-warm as the real reliability mechanism |
| 20 | No rate limiting on `/chat`; one attacker can drain Anthropic budget | Added in-memory IP rate limiter (`apps/api/src/regrag_api/rate_limit.py`); 6/min, 30/hr per IP on chat; 30/min on audit. Configurable via env. |
| 26 | UI stage labels were engineer-jargon (`classify`, `decompose`, etc.) | Friendlier: "Understanding your question", "Breaking it into sub-questions", etc. |
| 5 | Eval headline didn't acknowledge author bias | Added a paragraph after the headline table explicitly framing the numbers as "on author-curated questions" |

---

## Deferred — high severity but takes real work

### 9. Live demo cites wrong order for fact-finding queries (the canonical citation-drift failure mode, demonstrated live)

**Observed:** "What was the effective date of Order 841?" → answer cites chunk `20190221-3057:c0127` (which is **Order 845-A**, the rehearing of 845, issued 2019-02-21). The 845-A chunk mentions Order 841's effective date in passing. The user reading the answer sees "[[20190221-3057:c0127]]" and would click expecting Order 841 content; they get Order 845-A.

**Why this is hard to fix:** the same failure mode the substantive judge is *supposed* to catch. The judge approved this citation because Order 845-A *does* substantively support the claim — the date is right, the source mentions it. The fix isn't more judging; it's **document-anchored retrieval**: when the query names a specific order, the retriever should heavily prefer chunks FROM that order over chunks that REFERENCE it.

**Sketch of fix:** in `apps/api/src/regrag_api/retrieval/hybrid.py`, when `extract_identifiers(query)` returns one or more order numbers, look up the corresponding accession_numbers from a manifest map and boost RRF scores for chunks with matching `accession_number`. Probably 2-3 hours of work + an eval re-run to measure impact.

**Severity:** high, because this is the demo's canonical failure mode demonstrated in a routine query. A FERC counsel evaluating the demo would catch this in their first 5 questions.

---

### 10. Corpus is missing Order 841-A (the May 2019 rehearing of 841)

**Observed:** Order 841 is in the corpus but its rehearing order (841-A, issued 2019-05-16) is not. 841-A changed substantive points on state opt-out and storage definitions.

**Why deferred:** same eLibrary-SPA discovery problem as the rest of the corpus expansion. The URL isn't at a derivable `/sites/default/files/` path; Google searches don't turn it up. Would either need Playwright-driven eLibrary scraping or a third-party mirror like the wrightlaw Order 2023.

**Severity:** medium — affects multi-doc questions about the storage-rule arc. The eval includes a question about "the D.C. Circuit's review of Order 841" which is answered from 841 alone, missing the 841-A clarifications that came BEFORE the D.C. Circuit case.

---

### 8. Hybrid retrieval's claim of beating pure vector is unproven on this eval

**Observed:** CS §7 claims hybrid retrieval (vector + keyword + identifier floor) beats pure vector specifically on identifier-heavy queries. But the matched baseline used pure vector and got 95.8% recall on the same 40 questions — essentially identical to RegRAG's 96.9%. The eval doesn't break out an "identifier-heavy queries" subset where the hybrid logic would shine.

**Why deferred:** requires either (a) adding a sub-metric to the eval that tags identifier-heavy questions and reports recall on just that subset, or (b) authoring 5-10 NEW questions specifically designed to fail pure-vector and succeed on identifier-floor. Either is real work + an eval re-run.

**Severity:** medium — the claim in CS §7 is plausibly true (vector embeddings notoriously underweight rare identifier strings) but not measured by this eval. A senior reader will notice.

---

### 13. LLM-as-judge calibration vs human grader not measured

**Observed:** Sonnet is judging Sonnet-generated answers. Self-judging is a known calibration trap. eval-results.md acknowledges this but doesn't quantify it.

**Why deferred:** authoring 10 questions worth of human-grader rationale + comparing to judge scores is half a day of careful work.

**Severity:** medium — for portfolio credibility. A senior ML reader will mentally discount the 95.4% CF number by some amount; an explicit human-rater calibration would either confirm or refute that discount.

---

## Deferred — operational hygiene

### 21. No structured logging

Current: `logging.basicConfig` with human-readable format. Railway logs are searchable but not queryable by field.

What I'd want: JSON-structured logs with `query_id` linking the per-stage LLM calls to the audit row. Half-day of work; pays back the first time a weird production answer needs debugging.

### 22. Audit log write is synchronous

`write_query_log` runs inline in the chat handler. If Neon is slow, the chat response is delayed. There's exception-handling so a Neon outage doesn't break chat, but a slow Neon serializes the response.

Fix: move the audit write to a background task (FastAPI's `BackgroundTasks` or `asyncio.create_task`). ~30 min.

### 23. No graceful degradation if Anthropic API is down

If the synthesize Sonnet call throws, the graph crashes. No fallback to "we're temporarily unable to answer; please try again."

Fix: wrap each LLM call with `try/except` returning a clean refusal with `refusal_reason='upstream_outage'`. ~30 min.

### 33. Audit log captures raw query in plaintext

If a counsel queries with confidential client info, it lands in Neon. For a public regulated-domain demo this is technically the user's responsibility (they shouldn't), but the disclaimer should call it out and the system could PII-scrub before write.

Defer: out of scope for portfolio demo, in scope for regulated-domain production.

### 34. No data retention policy

Audit log grows unbounded.

Defer: same as 33 — out of scope for portfolio, in scope for production.

---

## Deferred — surface polish

### 28. No screenshots in case study

A screenshot of the chat UI mid-response (showing the live pipeline panel firing) would communicate the system's feel more efficiently than the current prose. 30 min if I take screenshots manually; longer if I want them to look polished.

### 27. No favicon

Browser tabs show the default Next/Vercel icon. 5 min fix once I have an icon (a simple SVG of a citation bracket or doc icon would do).

### 11. Order 2222 fact sheet has accession_number=null

The fact sheet is in the corpus but doesn't have a real accession_number (no FERC banner on the PDF). The audit log shows chunks from it as `url-15ba54edbab1`. A counsel wouldn't know how to cite it.

Fix: assign a synthetic accession like `factsheet-RM18-9` and update the manifest. ~10 min.

### 17. Refusal metric is binary, not split into precision/recall

Current metric: `refusal_correct` is True if the system refused-when-expected OR answered-when-expected. A system that refuses everything would score 50% on a balanced set.

Better: split into refusal-precision (of the refusals it made, how many were correct?) and refusal-recall (of the OOS questions, how many did it refuse?). 30 min metric change + eval re-run.

### 15. Audit log conflates production traffic with eval runs

Same table, distinguished only by `user_id` ("eval-runner" vs other). A production /audit query for "how does CF drift over time?" would mix eval and real traffic.

Fix: add a `run_id` or `traffic_class` column. Schema migration + small refactor. Defer to production deployment.

---

## What I'm intentionally not doing

- **#29 cost model section in CS §7/§8**: would strengthen the buyer-conversation story but isn't critical for a portfolio piece. Easy to fill in conversationally if asked.
- **#30 explicit "who would buy this" section**: same — useful for sales, not portfolio.
- **#31 contextualizing 95% CF to buyer trust ladder**: same.
- **#32b verify Sift framing in CS closing line**: requires knowing what Sift's actual production status is. Out of my context.

Total scope: the same-day fixes close ~12 of the 34 findings. The deferred items cluster into 4 categories: (a) real engineering work that needs another half-day each, (b) operational hygiene for production, (c) surface polish, (d) intentional non-goals for a portfolio piece.
