-- RegRAG Postgres schema (Neon + pgvector).
--
-- Apply by pasting into Neon's SQL Editor, or run:
--   psql "$DATABASE_URL" -f infra/neon-schema.sql
--
-- Schema reflects the corrections from docs/critique.md (M5/M6) and
-- the design decisions in docs/implementation-plan.md §2.2.

CREATE EXTENSION IF NOT EXISTS vector;

-- Document-level metadata
CREATE TABLE IF NOT EXISTS documents (
    accession_number TEXT PRIMARY KEY,
    order_number     TEXT,
    docket_numbers   TEXT[],
    document_type    TEXT NOT NULL,        -- 'final_rule' | 'order' | 'nopr' | 'rehearing_order' | 'anopr' | 'fact_sheet'
    issue_date       DATE NOT NULL,
    title            TEXT NOT NULL,
    source_url       TEXT NOT NULL,
    fetched_at       TIMESTAMPTZ NOT NULL,
    content_hash     TEXT NOT NULL
);

-- Chunks with embeddings
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id            TEXT PRIMARY KEY,                    -- e.g. "20180228-3066:P170"
    accession_number    TEXT NOT NULL REFERENCES documents(accession_number),
    section_heading     TEXT,
    paragraph_range     TEXT,                                 -- e.g. "P 170-172"
    chunk_text          TEXT NOT NULL,
    chunk_content_hash  TEXT NOT NULL,                        -- stable identity across re-chunks
    embedding           VECTOR(512),
    embedding_model     TEXT NOT NULL,                        -- e.g. "voyage-3.5-lite"
    chunker_version     TEXT NOT NULL,                        -- e.g. "v1.0"
    chunk_index         INT  NOT NULL,
    parent_chunk_id     TEXT REFERENCES chunks(chunk_id),     -- for footnote chunks split off
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS chunks_accession_idx
    ON chunks (accession_number);

-- 'simple' tokenizer (not 'english') preserves modal verbs (may, shall) and
-- regulatory identifiers that the English stemmer would otherwise strip
CREATE INDEX IF NOT EXISTS chunks_text_search_idx
    ON chunks USING gin (to_tsvector('simple', chunk_text));

-- Append-only audit log
CREATE TABLE IF NOT EXISTS query_log (
    query_id              UUID PRIMARY KEY,
    timestamp             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    user_id               TEXT,
    raw_query             TEXT NOT NULL,
    classification        TEXT,                               -- 'single_doc' | 'multi_doc'
    sub_queries           JSONB,
    retrieved_chunks      JSONB NOT NULL,                     -- snapshot {chunk_id, accession, section, text} for replay
    prompt_sent           TEXT NOT NULL,
    model_id              TEXT NOT NULL,
    raw_response          TEXT NOT NULL,
    verified_response     TEXT NOT NULL,
    citations_stripped    INT  NOT NULL DEFAULT 0,
    refusal_emitted       BOOLEAN NOT NULL DEFAULT FALSE,
    refusal_reason        TEXT,                               -- 'no_relevant_chunks' | 'llm_refusal' | etc.
    latency_ms_total      BIGINT NOT NULL,
    latency_ms_by_stage   JSONB,
    token_counts          JSONB
);

CREATE INDEX IF NOT EXISTS query_log_timestamp_idx
    ON query_log (timestamp DESC);
