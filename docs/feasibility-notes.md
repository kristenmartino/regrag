# FERC Corpus Feasibility — Verification Notes

Probing the open questions in `ferc-ingestion-plan.md` §9 and the implicit assumptions throughout the docs, using web fetches and HTTP HEAD checks. All checks performed 2026-05-06.

---

## TL;DR

| Question | Answer | Implication |
|----------|--------|-------------|
| Does eLibrary RSS cover orders? | **No** — the only RSS is `ecollection.ferc.gov/api/rssfeed`, which covers eForms *filings* (incoming), not orders (outgoing). | Discovery for new orders needs a different source. See §B1. |
| Does PDF parsing work on real FERC orders? | **Yes — direct fetch is trivial.** Order PDFs mirror at `www.ferc.gov/sites/default/files/{YYYY-MM}/...pdf`, served via Cloudflare cache, no auth required. | Ingestion plan §3 stage 2 (Fetch) is straightforward. PDF parsing fidelity (text extraction quality) is still TBD until you actually run pdfplumber/pymupdf on a sample — see §B5. |
| Reranker yes/no? | **Defer** — depends on eval results from a corpus that doesn't exist yet. | Plan implementation without reranker; keep the seam to add one later. |
| Does Neon pgvector handle the corpus? | **Defer** — trivially answerable once chunk count is known. | At ~3,000 chunks × 512-dim, this fits in the free tier. Not a constraint. |
| Are the three named orders (2222, 841, 745) accessible? | **Yes** — all three have direct PDFs verified by HTTP HEAD (HTTP 200, sizes 357KB–2MB). | Seed corpus is buildable today. See §C. |
| Does eLibrary permit scraping? | **Eligibility unclear** — robots.txt at `elibrary.ferc.gov/robots.txt` returns 404 (no rules); `www.ferc.gov/robots.txt` is permissive standard Drupal. Cloudflare bot management present. | Polite-scraping approach (1–2s delay, identifying User-Agent) should work. See §B6. |
| Is the "predictable PDF URL pattern" claim true? | **No** — there is no FERC-wide predictable URL based on accession number alone. The eLibrary documents are served via an Angular SPA backed by an undocumented JSON API. PDFs on `www.ferc.gov/sites/default/files/...` are uploaded with arbitrary paths. | Ingestion plan §2 needs revision. See §B2 for the actual URL patterns. |

**Two of the ingestion plan's source-access assumptions are incorrect** (predictable PDF URLs; eLibrary RSS for orders). The good news: the seed corpus is achievable through a simpler path (manual URL identification + direct fetch) that sidesteps eLibrary entirely.

---

## A. Verified facts (with evidence)

### A1. eLibrary is an Angular SPA — direct HTTP scraping returns only the bootstrap shell

`curl https://elibrary.ferc.gov/eLibrary/filelist?accession_number=20200917-3084` returns HTTP 200, Content-Type `text/html`, body 21KB. The body is an Angular bootstrap with `<app-root></app-root>` and `<script src="main.98d5286bf133b6f5.js" type="module"></script>` — no document data. Same shape for `/eLibrary/search` and `/eLibrary/docketsheet`. To actually retrieve eLibrary content, you need either (a) a headless browser to execute the SPA's JS calls, or (b) reverse-engineer the underlying JSON API the SPA hits.

A community wrapper exists: `github.com/4very/ferc-elibrary-api` — likely worth examining if the eLibrary API path is needed. Couldn't fetch the repo's README directly (404), but the project name suggests someone has solved this before.

**Implication for the ingestion plan:** the §3 Discovery stage as currently written ("queries the source(s) identified in §2 and produces a list of candidate documents with metadata") cannot use simple HTTP requests against eLibrary. Either (a) skip eLibrary entirely for the seed corpus and use direct PDF URLs, (b) add Playwright/Puppeteer to the pipeline, or (c) reverse-engineer the JSON API. Option (a) is the lowest-effort path and is sufficient for the demo.

### A2. The eForms RSS feed does NOT cover orders

`https://ecollection.ferc.gov/api/rssfeed` exists and is documented, but per FERC's own docs:
- It started October 1, 2021 (so it wouldn't cover Orders 745 from 2011, 841 from 2018, or 2222 from 2020 even if it covered orders).
- It covers **accepted filings** in the eCollection/eForms system — Forms 1, 2, 6, 60, 714 etc. submitted *by utilities and other entities to FERC*. Not issuances *from FERC*.
- It is capped at 650 most recent filings (a rolling window).

**Implication:** `ferc-ingestion-plan.md` §9 Q1 ("Does eLibrary RSS cover all three document types in scope?") has the answer "no — there is no RSS for orders/NOPRs/final rules." The closest equivalents are:
- `https://www.ferc.gov/news-events/news/decisions-notices` — HTML page that lists recent issuances. Scrapable.
- `https://www.ferc.gov/commission-orders-and-notices-1` — HTML index of orders.
- eSubscription — email-based, useful for tracking specific dockets, kludgy for bulk.

For the ongoing-update story, the realistic path is to scrape `decisions-notices` weekly. For the initial seed load, work from a hand-curated manifest (which is what IP §1 already proposes — good).

### A3. PDF mirrors on www.ferc.gov are accessible directly

HTTP HEAD checks against the three named orders, with `User-Agent: RegRAG-research/0.1 (krissi889@gmail.com)`:

| Order | URL | Status | Size |
|-------|-----|--------|------|
| 2222 (DER aggregation) | https://www.ferc.gov/sites/default/files/2020-09/E-1_0.pdf | 200 OK | 2,085,908 bytes (2.0 MB) |
| 841 (storage participation) | https://ferc.gov/sites/default/files/2020-06/Order-841.pdf | 200 OK | 623,155 bytes (608 KB) |
| 745 (demand response) | https://www.ferc.gov/sites/default/files/2020-06/Order-745.pdf | 200 OK | 357,179 bytes (349 KB) |

All three served via Cloudflare with `cache-control: public, max-age=14400`. No auth, no JS, no rate-limit hits at single requests. The PDF blob fetch step in IP §3 stage 2 is straightforward against these URLs.

### A4. voyage-3-lite IS natively 512-dim

Verified against [docs.voyageai.com/docs/embeddings](https://docs.voyageai.com/docs/embeddings) — voyage-3-lite has dimension **512 only**, no Matryoshka truncation support (truncation is a 3.5-series and 4-series feature). The ingestion plan's `VECTOR(512)` schema and "at 512 dimensions" claim in §5 are correct. *(This resolves H6 in `critique.md`.)*

**Forward-looking note:** voyage-3-lite has been superseded by **voyage-3.5-lite** (released May 2025), which supports flexible dimensions and is at similar price/quality. For a fresh build, voyage-3.5-lite at 512 dims is the obvious choice — same index size, plus the option to upgrade dimensionality later if recall needs it. Decision worth calling out in the implementation plan.

### A5. Robots and crawl etiquette

- `https://www.ferc.gov/robots.txt` → standard Drupal robots, no Crawl-delay specified, no restrictions on `/sites/default/files/` (where the PDF mirrors live). Polite scraping is fine.
- `https://elibrary.ferc.gov/robots.txt` → HTTP 404 (no robots.txt published). Means no explicit rules; not a license to hammer it. Single-threaded with 1–2s delay (as IP §2 already proposes) is the right posture.
- Cloudflare bot-management cookies (`__cf_bm`, `cf-bm`) are set on responses. Behave like a normal client and rotate UA → unlikely to trip blocks at the corpus size in scope.

### A6. Accession number format

Confirmed: `YYYYMMDD-NNNN` where the first 8 digits are the filed date and the last 4 are a sequential within-day series. Example: `20090114-5156`. This means `accession_number` can be validated with a regex (`^\d{8}-\d{4}$`) and the date prefix is parseable directly without needing a separate `filed_date` field.

---

## B. Per-question findings against IP §9

### B1. Q1 — Does eLibrary RSS cover all three in-scope document types?

**Answer: No.** See A2. There is no RSS feed for orders, NOPRs, or final rules issued *by* FERC. The eForms RSS covers only utility-to-FERC filings.

**What to use instead:**
- For initial seed corpus: a hand-curated manifest of accession numbers and direct PDF URLs (IP §1 already proposes this).
- For ongoing updates: weekly scrape of `https://www.ferc.gov/news-events/news/decisions-notices` filtered by document type.
- Optionally: subscribe via eSubscription to specific dockets of interest (RM*, AD*, EL*) for email notifications.

### B2. Q2 — How well does PDF parsing work on a real FERC order layout?

**Partial answer: PDFs are fetchable; parsing fidelity not yet verified.**

Plan-mode can't run pdfplumber against the actual PDFs, so this question is only half-answered. What I can say from PDF metadata:

- All three seed PDFs are real text PDFs (Cloudflare returns `Content-Type: application/pdf`, sizes are consistent with text+formatting, not scanned images).
- File sizes (357KB–2MB for 60–200 page documents) suggest text+light formatting, not scanned/OCR'd image PDFs.
- Last-modified dates are 2021-12-17 across all three files, suggesting FERC re-uploaded them in a batch — they're likely uniformly produced.

**Still unknown without local execution:**
- Multi-column layout prevalence and how it survives text extraction.
- Footnote interleaving — FERC orders use both end-of-page footnotes and end-of-document references.
- Table extraction — orders sometimes include rate schedules, capacity tables, etc. as embedded tables.
- Section numbering preservation — the section-aware chunker depends on detecting "I. Background", "II. Discussion", "P 47", etc.

**Recommendation:** Day 1 of implementation, run pdfplumber and pymupdf both against `Order-841.pdf` and capture (a) text extraction quality, (b) section heading detection, (c) footnote handling. Pick one and budget 1 day for handling whatever edge cases show up.

### B3. Q3 — Reranker yes/no?

**Defer.** The case study and ingestion plan correctly leave this open. There's no point deciding without an eval set to measure against.

### B4. Q4 — Neon pgvector capacity?

**Trivially adequate at corpus size.** Estimated: ~30–50 documents × ~50 chunks/doc avg = 1,500–2,500 chunks × 512 dimensions × 4 bytes/float = ~3–5 MB raw vectors. HNSW index overhead ~3–5x → ~15–25 MB total vector storage. Neon's free tier (3 GB) absorbs this with 100x headroom.

### B5. Implicit Q — "predictable PDF URL pattern" by accession

**The IP §2 claim is wrong.** There is no eLibrary URL where accession number → PDF directly. Instead:

- eLibrary documents are accessed via the SPA's `?accession_Number=...` parameter, which loads via JS-driven JSON API calls. The actual PDF download is handled by SPA logic, not a stable URL.
- PDFs on `www.ferc.gov/sites/default/files/{YYYY-MM}/{filename}.pdf` use arbitrary upload paths and filenames. They're stable once uploaded but not derivable from accession number.

**Implication:** the ingestion plan needs to maintain a `manifest.yaml` mapping each seed document to its actual PDF URL, *not* assume a derivable pattern. (IP §1 already proposes a manifest, so this is more of a clarification than a redesign.)

### B6. Implicit Q — robots.txt and rate limits

See A5. Polite scraping is fine. Cloudflare may push back at volume; not a concern at the seed-corpus scale.

---

## C. Verified seed corpus

The three case-study-named orders, with everything needed to ingest them today:

| Order | Docket | FERC Cite | Issued | PDF URL | Size |
|-------|--------|-----------|--------|---------|------|
| **2222** (Distributed Energy Resources / aggregation) | RM18-9-000 | 172 FERC ¶ 61,247 | 2020-09-17 | https://www.ferc.gov/sites/default/files/2020-09/E-1_0.pdf | 2.0 MB |
| **841** (Electric Storage Participation in RTO/ISO Markets) | RM16-23-000, AD16-20-000 | 162 FERC ¶ 61,127 | 2018-02-15 | https://ferc.gov/sites/default/files/2020-06/Order-841.pdf | 608 KB |
| **745** (Demand Response Compensation in Wholesale Markets) | RM10-17-000 | 134 FERC ¶ 61,187 | 2011-03-15 | https://www.ferc.gov/sites/default/files/2020-06/Order-745.pdf | 349 KB |

These three are sufficient to start the implementation. The ~15–20 adjacent orders called for in IP §1 can be identified via similar searches (`site:ferc.gov "Order No. XXXX"`) and added incrementally.

**Suggested companion documents** for the seed manifest, found during Phase B:
- Order 2222 Fact Sheet: https://www.ferc.gov/sites/default/files/2020-09/E-1-facts.pdf — short summary, useful for testing the chunker on shorter docs and for the eval set's "summarize" questions.
- Compliance filing acceptance for Order 2222 (174 FERC ¶ 61,197): https://www.ferc.gov/sites/default/files/2021-03/E-1.pdf — example of an order *about* a previous order, useful for testing multi-document synthesis questions.

---

## D. Risks and unknowns

Things Phase B couldn't resolve without actually running code:

1. **PDF parsing fidelity.** Need to run pdfplumber/pymupdf on Order-841.pdf and inspect output. Budget 1 day. If parsing is bad, fallback options include (a) pre-existing FERC text extracts, if any exist, or (b) using OCR for the worst-case PDFs.
2. **eLibrary JSON API stability.** If the demo ever needs *more* than the seed corpus + a few manual additions, you'll either need Playwright or to reverse-engineer the SPA's JSON calls. Both are doable but add complexity. Defer until needed.
3. **Re-upload patterns on www.ferc.gov.** All three Order PDFs have a Last-Modified of 2021-12-17. If FERC re-uploads PDFs occasionally (e.g. annual mass migration), the URL might change. Mitigation: capture and store the content hash on first fetch (IP §3 already does this), and re-discover on hash mismatch.
4. **Voyage AI deprecation timeline.** voyage-3-lite is now superseded. Worth checking Voyage's deprecation notices to make sure 3-lite isn't sunsetting in a way that affects a near-term build.

---

## E. Implications for the Phase C implementation plan

Things that need to change from the original ingestion plan:

1. **Discovery stage (IP §3 stage 1)** — drop the eLibrary RSS assumption. For the seed load, work from `manifest.yaml` directly. For ongoing updates, scrape `https://www.ferc.gov/news-events/news/decisions-notices` weekly.
2. **Fetch stage (IP §3 stage 2)** — works as designed, against `www.ferc.gov/sites/default/files/...` URLs. No eLibrary needed for the demo.
3. **`manifest.yaml` schema** — should explicitly include `pdf_url` per document, since URL is not derivable from accession number. Suggested fields: `accession_number`, `order_number`, `docket_numbers[]`, `document_type`, `issue_date`, `title`, `pdf_url`, `notes`.
4. **No change needed** to chunking, embedding, storage schema, or audit log design based on Phase B findings.
5. **Embedding model decision** — implementation plan should explicitly choose between voyage-3-lite (matches Sift) and voyage-3.5-lite (newer, more flexible). Recommend voyage-3.5-lite at 512 dims for forward-compatibility.
6. **Source verification day** (IP §8 days 1–2) — scope can shrink. Not "verify which eLibrary access methods work" (we know none of them work cleanly via plain HTTP), but rather "verify pdfplumber/pymupdf output quality on the seed PDFs." A half-day spike instead of two full days.

---

## Sources

- FERC Order 2222: [https://www.ferc.gov/sites/default/files/2020-09/E-1_0.pdf](https://www.ferc.gov/sites/default/files/2020-09/E-1_0.pdf), context at [https://www.ferc.gov/ferc-order-no-2222-explainer-facilitating-participation-electricity-markets-distributed-energy](https://www.ferc.gov/ferc-order-no-2222-explainer-facilitating-participation-electricity-markets-distributed-energy)
- FERC Order 841: [https://ferc.gov/sites/default/files/2020-06/Order-841.pdf](https://ferc.gov/sites/default/files/2020-06/Order-841.pdf), context at [https://www.ferc.gov/media/order-no-841](https://www.ferc.gov/media/order-no-841)
- FERC Order 745: [https://www.ferc.gov/sites/default/files/2020-06/Order-745.pdf](https://www.ferc.gov/sites/default/files/2020-06/Order-745.pdf)
- FERC eLibrary entry point: [https://elibrary.ferc.gov/](https://elibrary.ferc.gov/)
- FERC eLibrary docket sheet pattern: [https://elibrary.ferc.gov/eLibrary/docketsheet?docket=rm18-9-000](https://elibrary.ferc.gov/eLibrary/docketsheet?docket=rm18-9-000)
- FERC eForms RSS: [https://ecollection.ferc.gov/api/rssfeed](https://ecollection.ferc.gov/api/rssfeed)
- FERC eLibrary FAQs: [https://www.ferc.gov/elibrary-frequently-asked-questions-faqs](https://www.ferc.gov/elibrary-frequently-asked-questions-faqs)
- Community eLibrary wrapper: [https://github.com/4very/ferc-elibrary-api](https://github.com/4very/ferc-elibrary-api)
- Voyage AI text embedding models: [https://docs.voyageai.com/docs/embeddings](https://docs.voyageai.com/docs/embeddings)
- Voyage 3.5-lite announcement: [https://blog.voyageai.com/2025/05/20/voyage-3-5/](https://blog.voyageai.com/2025/05/20/voyage-3-5/)
