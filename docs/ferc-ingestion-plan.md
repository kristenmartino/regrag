# FERC Ingestion Plan

Implementation plan for the corpus pipeline supporting RegRAG. Scoped to produce a working demo and a credible case study within a 1–2 week build window, with explicit notes on what to verify during implementation versus what's locked-in design.

---

## 1. Scope and Initial Corpus

**Document types in scope (v1):** Final orders, Notices of Proposed Rulemaking (NOPRs), and final rules. These are the documents users actually reference in the case study's example questions, and they share enough structural similarity that one parser handles all three.

**Out of scope for v1:** Hearing transcripts, settlement agreements, individual docket filings (FERC eLibrary's "issuance" type that isn't an order), tariff filings. These have different structures and different use cases — including them dilutes the corpus and complicates chunking without adding to the demonstration.

**Initial seed corpus:** ~30–50 documents covering the orders referenced in the case study's example questions plus enough adjacent material to make multi-document synthesis questions answerable. Specifically:

- **Orders explicitly named in the case study:** 2222 (DER aggregation), 841 (storage participation in wholesale markets), 745 (demand response compensation). These three are non-negotiable — the eval set depends on them.
- **NOPRs that preceded those orders:** preserves the rulemaking trajectory and supports the "compare how FERC's treatment evolved" persona question.
- **~15–20 adjacent recent orders** on capacity markets, transmission planning, and interconnection. Enough for the corpus to feel substantive without becoming unmanageable.

Document the seed list in `corpus/manifest.yaml` so the ingestion is reproducible and the eval set has a stable target.

---

## 2. Source Access

**Primary source:** FERC eLibrary at `elibrary.ferc.gov`, which is the public document repository for orders, rulings, filings, and notices.

**Access methods worth investigating (in order of preference):**

1. **eLibrary's RSS feeds** — FERC publishes RSS for issuances filtered by date range and document type. Lowest-friction option for ongoing updates if it covers the document types in scope. *Verify during implementation: which feeds are stable, what document types each covers, whether full text is linked or just metadata.*
2. **eLibrary's search results pages** — scrapeable HTML with structured listings. Reliable but slower; appropriate for the initial bulk seed load if RSS coverage is incomplete.
3. **Direct PDF URLs** — once a document is identified by accession number, the PDF is at a predictable URL pattern. This is the actual fetch step regardless of how documents are discovered.

**Rate limiting and etiquette:** treat FERC as a polite-scraping target. Single-threaded fetches with a 1–2 second delay between requests, a clear `User-Agent` identifying the tool and a contact email, and respect for any `robots.txt` directives. The corpus is small enough that aggressive parallelism isn't needed and would risk getting the IP blocked.

**No authentication required.** All documents in scope are public.

---

## 3. Acquisition Pipeline

The pipeline has four stages and produces a normalized JSON record per document plus the original PDF stored as a content-addressable blob.

**Stage 1: Discovery.** A scheduled job queries the source(s) identified in section 2 and produces a list of candidate documents with metadata (accession number, title, issue date, document type, docket numbers). For the initial bulk load, this stage runs against the seed manifest directly rather than discovering documents.

**Stage 2: Fetch.** For each candidate, fetch the PDF, store it under `corpus/raw/{accession_number}.pdf`, and record the fetch timestamp and source URL. Skip if already fetched (idempotent on accession number). Hash the PDF bytes for content-addressing.

**Stage 3: Parse.** Extract text from the PDF using a library that preserves structure — `pdfplumber` or `pymupdf` rather than the simpler text-only extractors. The output is a structured representation: ordered list of sections with their headings, paragraph numbers, footnotes, and the raw text of each. *FERC orders use consistent section numbering conventions, but PDF parsing is always messier than expected; budget time for handling edge cases (multi-column layouts, footnote interleaving, table extraction).*

**Stage 4: Normalize.** Produce a canonical JSON record per document with: accession number, order number (if applicable), docket numbers, document type, issue date, title, full text, and the structured section list. This is the artifact downstream stages consume.

```
corpus/
├── manifest.yaml          # seed list of documents to fetch
├── raw/                   # original PDFs, keyed by accession number
│   └── {accession}.pdf
├── parsed/                # normalized JSON records
│   └── {accession}.json
└── ingest.log             # append-only log of all fetch + parse activity
```

---

## 4. Chunking Strategy

This is the part of the pipeline that most affects retrieval quality, and the part where naive defaults produce visibly worse results than a small amount of domain-aware engineering.

**Approach: section-aware variable-size chunking with bounded overlap.**

- **Primary chunk boundary: sections.** Each numbered section in a FERC order becomes a chunk candidate. Sections that exceed the maximum chunk size (target: ~800 tokens) are split further at paragraph boundaries; sections smaller than a minimum (target: ~200 tokens) are merged with adjacent sections from the same parent heading.
- **Overlap at boundaries.** A 1–2 sentence overlap between adjacent chunks preserves cross-reference context when a paragraph references "the previous section" or similar.
- **Metadata attached to every chunk:** parent document accession number, order number, docket numbers, section heading, paragraph numbers covered, and a stable chunk identifier. Without this metadata the citation verification step in section 5 of the case study cannot work.
- **Footnotes:** attach to the chunk containing the footnote reference rather than chunking separately. Standalone footnote chunks retrieve poorly and confuse generation.

Implement the chunker as a pure function of the parsed JSON record so it can be re-run without re-fetching when the chunking strategy changes. Re-chunking is going to happen multiple times during the build; design for it.

---

## 5. Embedding Pipeline

**Model:** Voyage AI `voyage-3.5-lite` at 512 dimensions. The 3-lite predecessor was Sift's choice; 3.5-lite (released May 2025) gives equivalent cost with flexible-dimension support that preserves an upgrade path. Operational patterns (batching, rate limit handling, retry on transient failures) carry over from Sift unchanged.

**Batching:** embed in batches of 128 chunks. Voyage's API tolerates larger batches but 128 is a safe default that handles the eval-set-sized corpus in a reasonable wall-clock time without fighting rate limits.

**Idempotency:** key embeddings by chunk content hash and embed model version. If the chunker output changes, re-embed only the chunks whose content actually changed. If the embedding model changes, re-embed the full corpus and store under a versioned namespace so old and new can coexist during evaluation.

**Cost expectation:** for ~30–50 documents at maybe 30–80 chunks each, total embedding volume is ~1,500–4,000 chunks. Voyage's voyage-3.5-lite is in the free tier (200M tokens/month) at this corpus scale; cost is not a constraint.

---

## 6. Storage Schema

Postgres schema designed to support both retrieval (sections 5 of the case study) and audit (also section 5).

```sql
-- Document-level metadata
CREATE TABLE documents (
    accession_number TEXT PRIMARY KEY,
    order_number TEXT,
    docket_numbers TEXT[],
    document_type TEXT NOT NULL,          -- 'order', 'nopr', 'final_rule'
    issue_date DATE NOT NULL,
    title TEXT NOT NULL,
    source_url TEXT NOT NULL,
    fetched_at TIMESTAMPTZ NOT NULL,
    content_hash TEXT NOT NULL
);

-- Chunks with embeddings
CREATE TABLE chunks (
    chunk_id TEXT PRIMARY KEY,
    accession_number TEXT NOT NULL REFERENCES documents(accession_number),
    section_heading TEXT,
    paragraph_range TEXT,                  -- e.g., "P 47-52"
    chunk_text TEXT NOT NULL,
    chunk_content_hash TEXT NOT NULL,      -- stable identity across re-chunks
    embedding VECTOR(512),
    embedding_model TEXT NOT NULL,
    chunker_version TEXT NOT NULL,         -- which chunker produced this row
    chunk_index INT NOT NULL,
    parent_chunk_id TEXT REFERENCES chunks(chunk_id),  -- for footnote chunks split off when parent exceeds max tokens
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX chunks_accession_idx ON chunks (accession_number);
-- 'simple' tokenizer (not 'english') preserves modal verbs (may, shall) and identifiers
-- that the English stemmer would otherwise strip
CREATE INDEX chunks_text_search_idx ON chunks USING gin (to_tsvector('simple', chunk_text));

-- Audit log (append-only)
CREATE TABLE query_log (
    query_id UUID PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    user_id TEXT,
    raw_query TEXT NOT NULL,
    classification TEXT,                   -- 'single_doc' | 'multi_doc'
    sub_queries JSONB,
    retrieved_chunks JSONB NOT NULL,       -- full snapshot {chunk_id, accession, section, text} for audit-time replay even if chunks table is later re-chunked

    prompt_sent TEXT NOT NULL,
    model_id TEXT NOT NULL,
    raw_response TEXT NOT NULL,
    verified_response TEXT NOT NULL,
    citations_stripped INT DEFAULT 0,
    refusal_emitted BOOLEAN NOT NULL DEFAULT FALSE,
    refusal_reason TEXT,                   -- 'no_relevant_chunks' | 'llm_refusal' | etc.
    latency_ms_total BIGINT NOT NULL,
    latency_ms_by_stage JSONB,
    token_counts JSONB
);

CREATE INDEX query_log_timestamp_idx ON query_log (timestamp DESC);
```

The `chunks` table supports both vector search (HNSW index on the embedding column) and keyword search (GIN index on the text), which is what makes the hybrid retrieval in the case study possible. The `query_log` table is the audit substrate referenced throughout case study sections 4, 5, and 6.

---

## 7. Update Strategy

**Initial bulk load:** run the full pipeline against the seed manifest in section 1. This is a one-time event taking a few hours of wall-clock time, mostly bounded by polite scraping delays.

**Ongoing updates:** scheduled job that runs the discovery stage weekly, identifies new documents matching the scope criteria, and runs them through the rest of the pipeline. New documents are additive — existing documents aren't re-fetched unless their content hash changes.

**Re-chunking and re-embedding** are separate concerns from ingestion. When chunking strategy changes, run the chunker against existing parsed JSON records without re-fetching. When the embedding model changes, embed under a new namespace and switch retrieval to the new namespace once eval results validate it.

**Versioning:** every artifact (parsed record, chunk set, embedding set) is keyed by both content hash and pipeline version. This means historical query logs can be replayed against the corpus as it existed at query time, which is the kind of property a real public-sector deployment would require and which the case study's section 8 references.

---

## 8. Implementation Milestones

Two-week build, structured to produce a working end-to-end demo by end of week 1 and add the case-study-required polish in week 2.

**Week 1:**
- *Days 1–2:* Source verification — confirm which eLibrary access methods work, fetch and parse 3–5 documents end-to-end, validate the parsing approach handles FERC's actual PDF structure.
- *Days 3–4:* Build the acquisition pipeline against the full seed manifest. By end of day 4, all seed documents fetched, parsed, and stored.
- *Days 5–7:* Chunking and embedding. By end of week 1, the corpus is in Postgres, vector and keyword indices are built, and a basic retrieval query returns sensible chunks.

**Week 2:**
- *Days 8–9:* Hybrid retrieval implementation, basic LangGraph orchestration, simple generation step. Working end-to-end demo by end of day 9.
- *Days 10–11:* Citation verification, audit logging, eval harness with the 28-question seed set.
- *Days 12–14:* Frontend chat UI, deployment, case study writeup polish, diagram finalization.

The longest pole is almost certainly day 1–2 — verifying that the eLibrary access patterns and PDF parsing work as expected. If those go badly, fall back to a smaller curated set of documents fetched manually rather than letting the ingestion pipeline absorb the full week.

---

## 9. Open Questions to Resolve During Implementation

- **eLibrary RSS coverage.** Does it include all three document types in scope, or only some? If only some, which scraping strategy fills the gap?
- **PDF parsing fidelity.** How well does the chosen library handle FERC's actual layouts? Worth a one-day spike before committing.
- **Reranker yes/no.** A cross-encoder reranker on top of the hybrid retrieval would likely improve recall on the harder eval questions, but adds latency and complexity. Decide based on eval results from week 2 day 10–11.
- **Hosting.** The demo can run on the same Railway/Vercel/Neon stack as Sift, which minimizes net-new infrastructure work. Confirm Neon's pgvector instance has enough capacity for the corpus or upgrade tier as needed.

---

*Companion to `regrag-case-study.md` and `regrag-diagrams.md`.*
