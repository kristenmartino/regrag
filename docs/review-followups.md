# Review follow-ups

Findings from the multi-perspective review of the deliverable on 2026-05-22. Items shipped same-day are at the top; deferred items below with severity, scope, and what we'd want to talk through before doing each one.

---

## Shipped — review batch 1 (2026-05-22)

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

## Shipped — review batch 2 (2026-05-26)

| # | What | Fix |
|---|---|---|
| 9 | Live demo cited Order 845-A for "effective date of Order 841" queries (canonical citation-misattribution failure mode) | **Doc-anchored retrieval**: when the query names a specific order, a separate top-K vector search restricted to the canonical accession(s) joins the RRF fusion pool, and synthesize injects a SCOPE block telling the model to cite from those accessions or refuse — not substitute a different-order chunk that mentions the named order. Smoke-tested: Order 841 chunk (`20180228-3066:c0259`) now cited directly; 845-A used only as corroboration with explicit cross-reference disclosure. |
| 8 | CS §7 claimed hybrid retrieval beats pure vector but matched baseline showed identical recall | Softened the claim: "intended to help on identifier-heavy queries; current eval doesn't isolate that subset cleanly." Noted that a v6 cut would add a 5–10 question adversarial subset. Also updated the stale 98.3% number to current 96.9%. |
| 23 | Anthropic API outage would crash the synthesize node | Wrap `client.messages.create` in try/except → soft refusal with new `refusal_reason="llm_unavailable"`. |
| 22 | Audit log write was synchronous, serialized chat response on Neon latency | Fire `write_query_log` in a daemon thread from `graph.py`. `write_query_log` creates its own connection so it's thread-safe; failures are logged but never raised. |
| 33b | Disclaimer didn't mention plaintext query logging | Added bold "**Do not paste confidential client information** — queries are logged in plaintext for audit." |
| 17 | Refusal accuracy was binary, hid the fact that baselines never refuse anything | Split into refusal_precision and refusal_recall in `AggregateReport`. Recomputed from existing JSON: v5 is 70% P / 87.5% R; both baselines are 0% R (never refused a single OOS question). The split makes the architecture's contribution to refusal capability legible. |
| 11 | Order 2222 fact sheet had `accession_number: null` → fell back to `url-15ba54edbab1` slug in audit log | Manifest now uses synthetic `factsheet-RM18-9`. Documented that existing DB rows keep the old slug; a reingest of this entry would clean them up. |
| 28 | No screenshots in case study | Two PNGs in `docs/images/`: chat empty state (corpus + disclaimer + samples) and pipeline panel mid-response (classify+decompose done, 4 sub-queries visible, retrieve in progress). Embedded in §3 and §4 with prose. Capture script at `scripts/screenshot.mjs`. |
| 27 | No favicon (default Vercel icon) | Added `apps/web/src/app/icon.svg` (citation-bracket motif on slate background). Next.js App Router auto-serves it. |
| 32b | Sift framing implied commercial deployment | Softened to "prior end-to-end RAG system the author designed and deployed (Sift … built for personal and friends-and-family use, not commercial scale)." |

---

## Deferred — high severity but takes real work

### 10. Corpus is missing Order 841-A (the May 2019 rehearing of 841)

**RESOLVED 2026-05-29:** Order 841-A (`20190516-3057`) + the Federal Register Order 841 (`fr-2018-03-06-2018-03708`) were ingested via column-aware parsing; corpus is now 17 docs. See the v6 citation-attribution work.

**Observed:** Order 841 is in the corpus but its rehearing order (841-A, issued 2019-05-16) is not. 841-A changed substantive points on state opt-out and storage definitions.

**Why deferred:** same eLibrary-SPA discovery problem as the rest of the corpus expansion. The URL isn't at a derivable `/sites/default/files/` path; Google searches don't turn it up. Would either need Playwright-driven eLibrary scraping or a third-party mirror like the wrightlaw Order 2023.

**Severity:** medium — affects multi-doc questions about the storage-rule arc. The eval includes a question about "the D.C. Circuit's review of Order 841" which is answered from 841 alone, missing the 841-A clarifications that came BEFORE the D.C. Circuit case.

---

### 13. LLM-as-judge calibration vs human grader

**Tracked as:** [GitHub issue #6](https://github.com/kristenmartino/regrag/issues/6) — BLOCKED on the human-grading step.

**Observed:** Sonnet is judging Sonnet-generated answers. Self-judging is a known calibration trap. eval-results.md acknowledges this but doesn't quantify it.

**Why deferred:** authoring 10 questions worth of human-grader rationale + comparing to judge scores is half a day of careful work.

**Severity:** medium — for portfolio credibility. A senior ML reader will mentally discount the 95.4% CF number by some amount; an explicit human-rater calibration would either confirm or refute that discount.

---

## Deferred — operational hygiene

### 21. No structured logging

**Tracked as:** [GitHub issue #7](https://github.com/kristenmartino/regrag/issues/7).

Current: `logging.basicConfig` with human-readable format. Railway logs are searchable but not queryable by field.

What I'd want: JSON-structured logs with `query_id` linking the per-stage LLM calls to the audit row. Half-day of work; pays back the first time a weird production answer needs debugging.

### 33. Audit log still captures raw query in plaintext — only the disclaimer is in place

**Tracked as:** [GitHub issue #9](https://github.com/kristenmartino/regrag/issues/9).

Disclaimer now warns "**Do not paste confidential client information**" (shipped in batch 2 as #33b). But the underlying system still writes the raw query to Neon — no PII detection or scrubbing.

Defer the PII-scrubbing piece: out of scope for portfolio demo, in scope for regulated-domain production. Likely shape: an inline regex pass for emails, phone numbers, SSN-shaped strings; redact before write.

### 34. No data retention policy

Audit log grows unbounded.

Defer: same as 33 — out of scope for portfolio, in scope for production.

---

## Deferred — surface polish

### 15. Audit log conflates production traffic with eval runs

**Tracked as:** [GitHub issue #8](https://github.com/kristenmartino/regrag/issues/8).

Same table, distinguished only by `user_id` ("eval-runner" vs other). A production /audit query for "how does CF drift over time?" would mix eval and real traffic.

Fix: add a `run_id` or `traffic_class` column. Schema migration + small refactor. Defer to production deployment.

---

## What I'm intentionally not doing

- **#29 cost model section in CS §7/§8**: would strengthen the buyer-conversation story but isn't critical for a portfolio piece. Easy to fill in conversationally if asked.
- **#30 explicit "who would buy this" section**: same — useful for sales, not portfolio.
- **#31 contextualizing 95% CF to buyer trust ladder**: same.
- **#21 structured JSON logging**: production hygiene; matters when there's traffic to debug. Demo doesn't have it.

Total scope: 22 of 34 findings shipped across two batches; remaining 12 cluster into (a) real engineering work that needs another half-day each (Order 841-A corpus expansion, human-rater judge calibration), (b) operational hygiene for production (structured logging, PII scrubbing, eval/prod table split), (c) intentional non-goals for a portfolio piece.
