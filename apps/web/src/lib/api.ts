// API types and client for the RegRAG FastAPI backend.
//
// Day 15: simple POST /chat returning JSON. Day 16 will swap this for SSE
// streaming so intermediate stage events can drive the pipeline panel.

export type Classification = "single_doc" | "multi_doc";

export type ChunkSummary = {
  chunk_id: string;
  accession_number: string;
  section_heading: string | null;
  paragraph_range: string | null;
  chunk_text_preview: string;
  rrf_score: number | null;
  cosine_sim: number | null;
};

export type ChatResponse = {
  classification: Classification | null;
  classification_confidence: number | null;
  sub_queries: string[] | null;
  retrieved_chunks: ChunkSummary[];
  final_answer: string;
  refusal_emitted: boolean;
  refusal_reason: string | null;
  citations_stripped: number;
  regeneration_count: number;
  timings_ms: Record<string, number>;
};

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function postChat(query: string): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ query }),
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`POST /chat ${res.status}: ${body.slice(0, 200)}`);
  }
  return (await res.json()) as ChatResponse;
}

export function totalLatencyMs(timings: Record<string, number>): number {
  return Object.values(timings).reduce((a, b) => a + b, 0);
}

// ─── Streaming /chat/stream ────────────────────────────────────────

export type StageName =
  | "classify"
  | "decompose"
  | "retrieve_single"
  | "retrieve_parallel"
  | "synthesize"
  | "verify";

export type StreamEvent =
  | { type: "started"; query: string }
  | {
      type: "stage_complete";
      stage: StageName;
      // Free-shape per stage; safe to render as JSON if unknown
      delta_summary: Record<string, unknown>;
      elapsed_ms: number;
    }
  | { type: "done"; state: ChatResponse }
  | { type: "error"; message: string };

/**
 * POST /chat/stream and parse SSE events. Calls onEvent for each event in
 * order. Resolves when the stream completes (done or error event), rejects
 * on a network/transport failure.
 */
export async function streamChat(
  query: string,
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${API_BASE}/chat/stream`, {
    method: "POST",
    headers: { "content-type": "application/json", accept: "text/event-stream" },
    body: JSON.stringify({ query }),
    signal,
  });
  if (!res.ok || !res.body) {
    const body = res.body ? await res.text() : "(no body)";
    throw new Error(`POST /chat/stream ${res.status}: ${body.slice(0, 200)}`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    // SSE events are separated by \n\n; each event has lines like "data: <json>"
    let sep: number;
    while ((sep = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, sep).trim();
      buffer = buffer.slice(sep + 2);
      if (!block) continue;
      const dataLine = block
        .split("\n")
        .find((line) => line.startsWith("data:"));
      if (!dataLine) continue;
      const json = dataLine.slice(5).trim();
      try {
        onEvent(JSON.parse(json) as StreamEvent);
      } catch {
        // Skip malformed events rather than failing the whole stream.
      }
    }
  }
}

// Human-readable stage labels for UI rendering
export const STAGE_LABEL: Record<StageName, string> = {
  classify: "Classify intent",
  decompose: "Decompose into sub-queries",
  retrieve_single: "Retrieve chunks",
  retrieve_parallel: "Retrieve chunks (parallel)",
  synthesize: "Synthesize answer",
  verify: "Verify citations",
};

// ─── /audit endpoints ──────────────────────────────────────────────

export type AuditRowSummary = {
  query_id: string;
  timestamp: string; // ISO
  user_id: string | null;
  raw_query: string;
  classification: string | null;
  refusal_emitted: boolean;
  citations_stripped: number;
  latency_ms_total: number;
  n_chunks: number;
};

export type AuditChunk = {
  chunk_id: string;
  accession_number: string;
  section_heading: string | null;
  paragraph_range: string | null;
  chunk_text: string;
  parent_chunk_id: string | null;
};

export type AuditRowDetail = {
  query_id: string;
  timestamp: string;
  user_id: string | null;
  raw_query: string;
  classification: string | null;
  sub_queries: string[] | null;
  retrieved_chunks: AuditChunk[];
  prompt_sent: string;
  model_id: string;
  raw_response: string;
  verified_response: string;
  citations_stripped: number;
  refusal_emitted: boolean;
  refusal_reason: string | null;
  latency_ms_total: number;
  latency_ms_by_stage: Record<string, number>;
  token_counts: Record<string, { in: number; out: number }>;
};

export async function fetchAuditList(
  opts: { limit?: number; user_id?: string } = {},
): Promise<AuditRowSummary[]> {
  const params = new URLSearchParams();
  if (opts.limit) params.set("limit", String(opts.limit));
  if (opts.user_id) params.set("user_id", opts.user_id);
  const qs = params.toString() ? `?${params}` : "";
  const res = await fetch(`${API_BASE}/audit${qs}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`GET /audit ${res.status}`);
  return (await res.json()) as AuditRowSummary[];
}

export async function fetchAuditDetail(query_id: string): Promise<AuditRowDetail> {
  const res = await fetch(`${API_BASE}/audit/${query_id}`, { cache: "no-store" });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`GET /audit/${query_id} ${res.status}: ${body.slice(0, 200)}`);
  }
  return (await res.json()) as AuditRowDetail;
}
