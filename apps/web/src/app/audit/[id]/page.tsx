"use client";

import Link from "next/link";
import { use, useEffect, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { AuditRowDetail, fetchAuditDetail } from "@/lib/api";

export default function AuditDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const [row, setRow] = useState<AuditRowDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchAuditDetail(id)
      .then(setRow)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [id]);

  return (
    <div className="flex h-[100dvh] flex-col bg-background">
      <header className="border-b px-6 py-4">
        <div className="mx-auto flex max-w-5xl items-center justify-between">
          <div>
            <Link href="/audit" className="text-xs text-muted-foreground hover:underline">
              ← Audit log
            </Link>
            <h1 className="text-lg font-semibold tracking-tight">Query record</h1>
            <p className="font-mono text-[11px] text-muted-foreground">{id}</p>
          </div>
          <Link href="/">
            <Button variant="outline" size="sm">Chat</Button>
          </Link>
        </div>
      </header>

      <ScrollArea className="flex-1">
        <div className="mx-auto max-w-5xl space-y-4 px-6 py-6">
          {error ? (
            <Card className="border-destructive bg-destructive/10 px-4 py-3 text-sm text-destructive">
              <p className="font-medium">Failed to load audit row</p>
              <p className="mt-1 font-mono text-xs">{error}</p>
            </Card>
          ) : row === null ? (
            <div className="space-y-3">
              {[1, 2, 3, 4].map((i) => (
                <Skeleton key={i} className="h-32 w-full" />
              ))}
            </div>
          ) : (
            <DetailContent row={row} />
          )}
        </div>
      </ScrollArea>
    </div>
  );
}

function DetailContent({ row }: { row: AuditRowDetail }) {
  return (
    <>
      {/* Top-line summary */}
      <Card className="px-4 py-3">
        <div className="mb-3 flex flex-wrap items-center gap-2 text-xs">
          <Badge variant="secondary" className="font-mono">
            {row.classification ?? "no class"}
          </Badge>
          <Badge variant="outline" className="font-mono">
            user: {row.user_id ?? "(demo)"}
          </Badge>
          <Badge variant="outline" className="font-mono">
            {row.retrieved_chunks.length} chunks
          </Badge>
          {row.refusal_emitted && (
            <Badge variant="destructive" className="font-mono">
              refused: {row.refusal_reason ?? "?"}
            </Badge>
          )}
          {row.citations_stripped > 0 && (
            <Badge variant="destructive" className="font-mono">
              {row.citations_stripped} stripped
            </Badge>
          )}
          <span className="ml-auto font-mono text-xs text-muted-foreground">
            {(row.latency_ms_total / 1000).toFixed(1)}s · {new Date(row.timestamp).toLocaleString()}
          </span>
        </div>
        <p className="text-sm font-medium">Query</p>
        <p className="mt-1 text-sm leading-relaxed">{row.raw_query}</p>
      </Card>

      {/* Sub-queries (multi-doc only) */}
      {row.sub_queries && row.sub_queries.length > 0 && (
        <Card className="px-4 py-3">
          <p className="mb-2 text-sm font-medium">Sub-queries</p>
          <ul className="space-y-1 text-sm">
            {row.sub_queries.map((q, i) => (
              <li key={i} className="flex gap-2">
                <span className="font-mono text-xs text-muted-foreground">{i + 1}.</span>
                <span>{q}</span>
              </li>
            ))}
          </ul>
        </Card>
      )}

      {/* Verified response */}
      <Card className="px-4 py-3">
        <p className="mb-2 text-sm font-medium">Verified response (returned to user)</p>
        <p className="whitespace-pre-wrap text-sm leading-relaxed">
          {row.verified_response || <span className="italic text-muted-foreground">(empty)</span>}
        </p>
      </Card>

      {/* Raw response (collapsed if same as verified) */}
      <Card className="px-4 py-3">
        <details>
          <summary className="cursor-pointer text-sm font-medium">
            Raw model response (before citation verification)
          </summary>
          <p className="mt-2 whitespace-pre-wrap font-mono text-xs leading-relaxed text-muted-foreground">
            {row.raw_response || <span className="italic">(empty)</span>}
          </p>
        </details>
      </Card>

      {/* Retrieved chunks (snapshot at query time) */}
      <Card className="px-4 py-3">
        <details>
          <summary className="cursor-pointer text-sm font-medium">
            Retrieved chunks ({row.retrieved_chunks.length}) — snapshot at query time
          </summary>
          <p className="mt-1 text-xs text-muted-foreground">
            Stored as JSONB so this row remains replayable even if the chunks
            table is later re-chunked.
          </p>
          <div className="mt-3 grid gap-2">
            {row.retrieved_chunks.map((c) => (
              <Card key={c.chunk_id} className="px-3 py-2 text-xs">
                <div className="mb-1 flex flex-wrap items-center gap-2 font-mono text-[10px] text-muted-foreground">
                  <span className="font-semibold">{c.chunk_id}</span>
                  <Separator orientation="vertical" className="h-3" />
                  <span>{c.accession_number}</span>
                  {c.section_heading && (
                    <>
                      <Separator orientation="vertical" className="h-3" />
                      <span>{c.section_heading}</span>
                    </>
                  )}
                  {c.parent_chunk_id && (
                    <>
                      <Separator orientation="vertical" className="h-3" />
                      <span>parent: {c.parent_chunk_id}</span>
                    </>
                  )}
                </div>
                <p className="whitespace-pre-wrap leading-relaxed text-muted-foreground">
                  {c.chunk_text.slice(0, 800)}
                  {c.chunk_text.length > 800 ? "…" : ""}
                </p>
              </Card>
            ))}
          </div>
        </details>
      </Card>

      {/* Prompt sent to synthesizer */}
      <Card className="px-4 py-3">
        <details>
          <summary className="cursor-pointer text-sm font-medium">
            Prompt sent to {row.model_id}
          </summary>
          <p className="mt-1 text-xs text-muted-foreground">
            Truncated at 8 KB (full prompt includes all retrieved chunks; original may be larger).
          </p>
          <pre className="mt-2 max-h-[40rem] overflow-auto rounded bg-muted/50 p-3 text-[11px] leading-relaxed">
            {row.prompt_sent}
          </pre>
        </details>
      </Card>

      {/* Per-stage latency + token counts */}
      <Card className="px-4 py-3">
        <p className="mb-2 text-sm font-medium">Stages</p>
        <div className="overflow-hidden rounded border text-xs">
          <table className="w-full">
            <thead className="bg-muted/50 text-[10px] uppercase tracking-wider text-muted-foreground">
              <tr>
                <th className="px-3 py-1.5 text-left">Stage</th>
                <th className="px-3 py-1.5 text-right">Latency</th>
                <th className="px-3 py-1.5 text-right">Tokens (in)</th>
                <th className="px-3 py-1.5 text-right">Tokens (out)</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(row.latency_ms_by_stage).map(([stage, ms]) => {
                const tc = row.token_counts[stage];
                return (
                  <tr key={stage} className="border-t">
                    <td className="px-3 py-1.5 font-mono">{stage}</td>
                    <td className="px-3 py-1.5 text-right font-mono">
                      {(ms / 1000).toFixed(2)}s
                    </td>
                    <td className="px-3 py-1.5 text-right font-mono text-muted-foreground">
                      {tc?.in?.toLocaleString() ?? "—"}
                    </td>
                    <td className="px-3 py-1.5 text-right font-mono text-muted-foreground">
                      {tc?.out?.toLocaleString() ?? "—"}
                    </td>
                  </tr>
                );
              })}
              <tr className="border-t bg-muted/30 font-medium">
                <td className="px-3 py-1.5 font-mono">total</td>
                <td className="px-3 py-1.5 text-right font-mono">
                  {(row.latency_ms_total / 1000).toFixed(2)}s
                </td>
                <td colSpan={2}></td>
              </tr>
            </tbody>
          </table>
        </div>
      </Card>
    </>
  );
}
