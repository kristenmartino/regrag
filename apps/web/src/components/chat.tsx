"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Textarea } from "@/components/ui/textarea";
import {
  ChatResponse,
  StreamEvent,
  streamChat,
  totalLatencyMs,
} from "@/lib/api";
import {
  PipelineStage,
  PipelineView,
  SubQueriesPanel,
  stagesFromEvents,
} from "@/components/pipeline";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "";

// Threshold above which we show the warming hint. Anything under this is
// indistinguishable from normal latency, so we don't bother surfacing it.
const COLD_START_HINT_THRESHOLD_MS = 1500;

type WarmStatus =
  | { kind: "warming" }
  | { kind: "warm"; cold_start: boolean; ms: number }
  | { kind: "failed"; reason: string };

type Turn = {
  id: string;
  query: string;
  events: StreamEvent[];
  finalState: ChatResponse | null;
  error: string | null;
  pending: boolean;
};

const SAMPLE_QUERIES = [
  "What does Order 2222 require for DER aggregation reporting?",
  "Compare DER treatment across Orders 2222, 841, and 745",
  "What is the deadline for compliance with Order 841?",
  "What is FERC's position on residential rooftop solar permitting?",
];

export function Chat() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [pending, setPending] = useState(false);
  const [warm, setWarm] = useState<WarmStatus>({ kind: "warming" });
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Pre-warm the Railway backend on page mount so the first query doesn't
  // pay the cold-start tax. If the warm-up itself takes long enough to be
  // noticeable, surface the hint so the user knows what's happening rather
  // than thinking the demo is broken.
  useEffect(() => {
    if (!API_BASE) {
      setWarm({ kind: "failed", reason: "API base URL not configured" });
      return;
    }
    const t0 = performance.now();
    const controller = new AbortController();
    fetch(`${API_BASE}/health`, { cache: "no-store", signal: controller.signal })
      .then((res) => {
        const ms = performance.now() - t0;
        setWarm(
          res.ok
            ? { kind: "warm", cold_start: ms > COLD_START_HINT_THRESHOLD_MS, ms }
            : { kind: "failed", reason: `HTTP ${res.status}` },
        );
      })
      .catch((e) =>
        setWarm({ kind: "failed", reason: e instanceof Error ? e.message : String(e) }),
      );
    return () => controller.abort();
  }, []);

  async function submit(query: string) {
    if (!query.trim() || pending) return;
    const id = crypto.randomUUID();
    setTurns((prev) => [
      ...prev,
      { id, query, events: [], finalState: null, error: null, pending: true },
    ]);
    setInput("");
    setPending(true);

    try {
      await streamChat(query, (event) => {
        setTurns((prev) =>
          prev.map((t) => {
            if (t.id !== id) return t;
            const events = [...t.events, event];
            let finalState = t.finalState;
            let error = t.error;
            let stillPending = t.pending;
            if (event.type === "done") {
              finalState = event.state;
              stillPending = false;
            } else if (event.type === "error") {
              error = event.message;
              stillPending = false;
            }
            return { ...t, events, finalState, error, pending: stillPending };
          }),
        );
      });
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      setTurns((prev) =>
        prev.map((t) =>
          t.id === id ? { ...t, error: message, pending: false } : t,
        ),
      );
    } finally {
      setPending(false);
      inputRef.current?.focus();
    }
  }

  return (
    <div className="flex h-[100dvh] flex-col bg-background">
      <header className="border-b px-6 py-4">
        <div className="mx-auto flex max-w-5xl items-center justify-between">
          <div>
            <h1 className="text-lg font-semibold tracking-tight">RegRAG</h1>
            <p className="text-xs text-muted-foreground">
              Hybrid retrieval over FERC orders with grounded citations
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Link href="/audit">
              <Button variant="outline" size="sm" className="text-xs">
                Audit log
              </Button>
            </Link>
            <Badge variant="outline" className="font-mono text-xs">
              demo
            </Badge>
          </div>
        </div>
      </header>

      <ScrollArea className="flex-1">
        <div className="mx-auto max-w-5xl px-6 py-6">
          {turns.length === 0 && warm.kind === "warming" && (
            <WarmingBanner />
          )}
          {turns.length === 0 && warm.kind === "warm" && warm.cold_start && (
            <WarmedUpBanner ms={warm.ms} />
          )}
          {turns.length === 0 && warm.kind === "failed" && (
            <ApiUnavailableBanner reason={warm.reason} />
          )}
          {turns.length === 0 ? (
            <EmptyState onPick={submit} disabled={pending} />
          ) : (
            <div className="space-y-8">
              {turns.map((turn) => (
                <TurnView key={turn.id} turn={turn} />
              ))}
            </div>
          )}
        </div>
      </ScrollArea>

      <div className="border-t bg-background px-6 py-4">
        <form
          className="mx-auto flex max-w-5xl items-end gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            submit(input);
          }}
        >
          <Textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit(input);
              }
            }}
            placeholder="Ask about a FERC order…  (Enter to send, Shift+Enter for newline)"
            rows={2}
            className="resize-none"
            disabled={pending}
          />
          <Button type="submit" disabled={!input.trim() || pending}>
            {pending ? "…" : "Send"}
          </Button>
        </form>
      </div>
    </div>
  );
}

function WarmingBanner() {
  return (
    <Card className="mb-6 border-amber-300/60 bg-amber-50 px-4 py-3 text-xs text-amber-900 dark:border-amber-500/40 dark:bg-amber-950/30 dark:text-amber-200">
      <div className="flex items-start gap-2">
        <span className="relative mt-1 inline-flex h-2 w-2 flex-shrink-0">
          <span className="absolute inset-0 animate-ping rounded-full bg-amber-500 opacity-60" />
          <span className="relative inline-flex h-2 w-2 rounded-full bg-amber-500" />
        </span>
        <div>
          <div className="font-medium">Waking the demo backend…</div>
          <p className="mt-1 leading-relaxed">
            Railway's free tier sleeps after idle. First request typically takes
            5–30 seconds while the container starts. Subsequent queries are fast.
          </p>
        </div>
      </div>
    </Card>
  );
}

function WarmedUpBanner({ ms }: { ms: number }) {
  return (
    <Card className="mb-6 border-muted bg-muted/30 px-4 py-2 text-xs text-muted-foreground">
      Backend cold-started in {(ms / 1000).toFixed(1)}s. Subsequent queries will be fast.
    </Card>
  );
}

function ApiUnavailableBanner({ reason }: { reason: string }) {
  return (
    <Card className="mb-6 border-destructive bg-destructive/10 px-4 py-3 text-xs text-destructive">
      <div className="font-medium">Backend unreachable</div>
      <p className="mt-1 font-mono">{reason}</p>
    </Card>
  );
}

function EmptyState({
  onPick,
  disabled,
}: {
  onPick: (q: string) => void;
  disabled: boolean;
}) {
  return (
    <div className="mx-auto max-w-2xl text-center">
      <h2 className="text-2xl font-semibold tracking-tight">Ask about FERC orders</h2>
      <p className="mt-2 text-sm text-muted-foreground">
        The corpus contains Orders 2222, 841, 745, 845/845-A, plus the
        RM21-17 transmission planning ANOPR. Single-document lookups go
        through Haiku; multi-document synthesis questions get decomposed and
        answered with Sonnet. Each stage of the pipeline streams live.
      </p>
      <div className="mt-6 grid gap-2 text-left">
        {SAMPLE_QUERIES.map((q) => (
          <Button
            key={q}
            variant="outline"
            className="h-auto justify-start whitespace-normal py-3 text-left text-sm font-normal"
            disabled={disabled}
            onClick={() => onPick(q)}
          >
            {q}
          </Button>
        ))}
      </div>
    </div>
  );
}

function TurnView({ turn }: { turn: Turn }) {
  const stages: PipelineStage[] = stagesFromEvents(turn.events);
  // Pull sub_queries early from the decompose event so they show before the final answer
  const decomposeEvent = turn.events.find(
    (e) => e.type === "stage_complete" && e.stage === "decompose",
  );
  const subQueries =
    decomposeEvent?.type === "stage_complete"
      ? (decomposeEvent.delta_summary.sub_queries as string[] | undefined)
      : undefined;

  return (
    <div className="grid gap-4 md:grid-cols-[1fr_18rem]">
      <div className="space-y-3">
        {/* User message */}
        <div className="flex justify-end">
          <Card className="max-w-[85%] bg-primary px-4 py-3 text-primary-foreground">
            <p className="text-sm leading-relaxed">{turn.query}</p>
          </Card>
        </div>

        {/* Assistant response */}
        {turn.error ? (
          <Card className="border-destructive bg-destructive/10 px-4 py-3 text-sm text-destructive">
            <p className="font-medium">Request failed</p>
            <p className="mt-1 font-mono text-xs">{turn.error}</p>
          </Card>
        ) : turn.finalState ? (
          <ResponseView response={turn.finalState} />
        ) : (
          <Card className="border-dashed bg-muted/30 px-4 py-3 text-sm italic text-muted-foreground">
            Working through the pipeline…
          </Card>
        )}
      </div>

      {/* Sidebar: live pipeline + sub-queries */}
      <aside className="space-y-3">
        <PipelineView stages={stages} pending={turn.pending} />
        {subQueries && subQueries.length > 0 && (
          <SubQueriesPanel queries={subQueries} />
        )}
      </aside>
    </div>
  );
}

function ResponseView({ response }: { response: ChatResponse }) {
  const total = totalLatencyMs(response.timings_ms);
  return (
    <div className="space-y-2">
      <Card className="px-4 py-3">
        <div className="mb-3 flex flex-wrap items-center gap-2 text-xs">
          {response.refusal_emitted && (
            <Badge variant="destructive" className="font-mono">
              refused: {response.refusal_reason ?? "n/a"}
            </Badge>
          )}
          {response.citations_stripped > 0 && (
            <Badge variant="destructive" className="font-mono">
              {response.citations_stripped} stripped
            </Badge>
          )}
          {response.regeneration_count > 0 && (
            <Badge variant="outline" className="font-mono">
              regen×{response.regeneration_count}
            </Badge>
          )}
          <span className="ml-auto font-mono text-xs text-muted-foreground">
            {(total / 1000).toFixed(1)}s total
          </span>
        </div>

        <div className="prose prose-sm dark:prose-invert max-w-none">
          <AnswerText text={response.final_answer} />
        </div>
      </Card>

      {response.retrieved_chunks.length > 0 && (
        <details className="text-xs">
          <summary className="cursor-pointer text-muted-foreground hover:text-foreground">
            {response.retrieved_chunks.length} retrieved chunks
          </summary>
          <div className="mt-2 grid gap-2">
            {response.retrieved_chunks.slice(0, 8).map((c) => (
              <Card key={c.chunk_id} className="px-3 py-2 text-xs">
                <div className="mb-1 flex items-center gap-2 font-mono text-[10px] text-muted-foreground">
                  <span>{c.chunk_id}</span>
                  {c.section_heading && (
                    <>
                      <Separator orientation="vertical" className="h-3" />
                      <span>{c.section_heading}</span>
                    </>
                  )}
                  {c.cosine_sim !== null && (
                    <>
                      <Separator orientation="vertical" className="h-3" />
                      <span>cos={c.cosine_sim.toFixed(2)}</span>
                    </>
                  )}
                </div>
                <p className="text-muted-foreground">{c.chunk_text_preview}…</p>
              </Card>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}

function AnswerText({ text }: { text: string }) {
  const parts: (string | { citation: string })[] = [];
  const re = /\[\[([^\]]+)\]\]/g;
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) parts.push(text.slice(last, m.index));
    parts.push({ citation: m[1] });
    last = m.index + m[0].length;
  }
  if (last < text.length) parts.push(text.slice(last));

  return (
    <p className="whitespace-pre-wrap text-sm leading-relaxed">
      {parts.map((p, i) =>
        typeof p === "string" ? (
          <span key={i}>{p}</span>
        ) : (
          <Badge
            key={i}
            variant="outline"
            className="mx-0.5 inline-block px-1.5 py-0 align-middle font-mono text-[10px]"
          >
            {p.citation}
          </Badge>
        ),
      )}
    </p>
  );
}
