"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { AuditRowSummary, fetchAuditList } from "@/lib/api";

type Filter = "all" | "demo" | "eval-runner";

export default function AuditListPage() {
  const [rows, setRows] = useState<AuditRowSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<Filter>("all");

  useEffect(() => {
    const opts =
      filter === "eval-runner"
        ? { user_id: "eval-runner", limit: 50 }
        : { limit: 50 };
    fetchAuditList(opts)
      .then((r) =>
        filter === "demo"
          ? setRows(r.filter((x) => x.user_id === null))
          : setRows(r),
      )
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [filter]);

  return (
    <div className="flex h-[100dvh] flex-col bg-background">
      <header className="border-b px-6 py-4">
        <div className="mx-auto flex max-w-6xl items-center justify-between">
          <div>
            <h1 className="text-lg font-semibold tracking-tight">Audit log</h1>
            <p className="text-xs text-muted-foreground">
              Append-only record of every <code>/chat</code> invocation —
              query, classification, retrieved chunks, model output, latencies
            </p>
          </div>
          <Link href="/">
            <Button variant="outline" size="sm">
              ← Back to chat
            </Button>
          </Link>
        </div>
      </header>

      <div className="border-b px-6 py-2">
        <div className="mx-auto flex max-w-6xl items-center gap-2">
          <span className="text-xs text-muted-foreground">Filter:</span>
          {(["all", "demo", "eval-runner"] as Filter[]).map((f) => (
            <Button
              key={f}
              variant={filter === f ? "default" : "outline"}
              size="sm"
              className="h-7 text-xs"
              onClick={() => setFilter(f)}
            >
              {f}
            </Button>
          ))}
        </div>
      </div>

      <div className="flex-1 overflow-auto px-6 py-6">
        <div className="mx-auto max-w-6xl">
          {error ? (
            <Card className="border-destructive bg-destructive/10 px-4 py-3 text-sm text-destructive">
              <p className="font-medium">Failed to load audit log</p>
              <p className="mt-1 font-mono text-xs">{error}</p>
            </Card>
          ) : rows === null ? (
            <div className="space-y-2">
              {[1, 2, 3, 4, 5].map((i) => (
                <Skeleton key={i} className="h-14 w-full" />
              ))}
            </div>
          ) : rows.length === 0 ? (
            <Card className="px-6 py-8 text-center text-sm text-muted-foreground">
              No audit rows yet. Ask a question on the chat page and refresh.
            </Card>
          ) : (
            <div className="overflow-hidden rounded-lg border">
              <table className="w-full text-sm">
                <thead className="bg-muted/50 text-xs uppercase tracking-wider text-muted-foreground">
                  <tr>
                    <th className="px-4 py-2 text-left">Time</th>
                    <th className="px-4 py-2 text-left">Query</th>
                    <th className="px-4 py-2 text-left">User</th>
                    <th className="px-4 py-2 text-left">Class</th>
                    <th className="px-4 py-2 text-left">Result</th>
                    <th className="px-4 py-2 text-right">Chunks</th>
                    <th className="px-4 py-2 text-right">Latency</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => (
                    <tr
                      key={r.query_id}
                      className="border-t hover:bg-muted/30"
                    >
                      <td className="whitespace-nowrap px-4 py-2 align-top font-mono text-[11px] text-muted-foreground">
                        {formatTimestamp(r.timestamp)}
                      </td>
                      <td className="px-4 py-2 align-top">
                        <Link
                          href={`/audit/${r.query_id}`}
                          className="text-foreground hover:text-primary hover:underline"
                        >
                          {r.raw_query.length > 110
                            ? r.raw_query.slice(0, 110) + "…"
                            : r.raw_query}
                        </Link>
                      </td>
                      <td className="whitespace-nowrap px-4 py-2 align-top text-xs text-muted-foreground">
                        {r.user_id ?? "(demo)"}
                      </td>
                      <td className="px-4 py-2 align-top">
                        {r.classification && (
                          <Badge variant="secondary" className="font-mono text-[10px]">
                            {r.classification === "multi_doc" ? "multi" : "single"}
                          </Badge>
                        )}
                      </td>
                      <td className="px-4 py-2 align-top">
                        {r.refusal_emitted ? (
                          <Badge variant="destructive" className="font-mono text-[10px]">
                            refused
                          </Badge>
                        ) : r.citations_stripped > 0 ? (
                          <Badge variant="outline" className="font-mono text-[10px]">
                            {r.citations_stripped} stripped
                          </Badge>
                        ) : (
                          <Badge variant="outline" className="font-mono text-[10px]">
                            ok
                          </Badge>
                        )}
                      </td>
                      <td className="whitespace-nowrap px-4 py-2 text-right align-top font-mono text-xs text-muted-foreground">
                        {r.n_chunks}
                      </td>
                      <td className="whitespace-nowrap px-4 py-2 text-right align-top font-mono text-xs text-muted-foreground">
                        {(r.latency_ms_total / 1000).toFixed(1)}s
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function formatTimestamp(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}
