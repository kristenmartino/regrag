"use client";

import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { STAGE_LABEL, StageName, StreamEvent } from "@/lib/api";

export type PipelineStage = {
  name: StageName;
  status: "running" | "complete";
  elapsed_ms?: number;
  detail?: string;
};

/**
 * Build a list of stages from accumulated stream events. Stages appear in
 * the order they fire; the most recent one is shown as 'running' until its
 * stage_complete arrives.
 */
export function stagesFromEvents(events: StreamEvent[]): PipelineStage[] {
  const stages: PipelineStage[] = [];
  for (const e of events) {
    if (e.type !== "stage_complete") continue;
    stages.push({
      name: e.stage,
      status: "complete",
      elapsed_ms: e.elapsed_ms,
      detail: detailForStage(e.stage, e.delta_summary),
    });
  }
  return stages;
}

function detailForStage(
  name: StageName,
  summary: Record<string, unknown>,
): string {
  switch (name) {
    case "classify": {
      const cls = summary.classification as string | undefined;
      const conf = summary.confidence as number | undefined;
      const confStr = conf != null ? ` (${(conf * 100).toFixed(0)}%)` : "";
      return cls ? `${cls.replace("_", "-")}${confStr}` : "";
    }
    case "decompose": {
      const n = summary.n_sub_queries as number | undefined;
      return n != null ? `${n} sub-queries` : "";
    }
    case "retrieve_single":
    case "retrieve_parallel": {
      const n = summary.n_chunks as number | undefined;
      const cos = summary.top_cosine as number | undefined;
      const refused = summary.refusal_emitted as boolean | undefined;
      let s = n != null ? `${n} chunks` : "";
      if (cos != null) s += ` · top cos ${cos.toFixed(2)}`;
      if (refused) s += " · refusal triggered";
      return s;
    }
    case "synthesize": {
      const len = summary.draft_length as number | undefined;
      const regen = summary.regeneration_count as number | undefined;
      const refused = summary.refusal_emitted as boolean | undefined;
      if (refused) return "model refused";
      let s = len != null ? `${len.toLocaleString()} chars` : "";
      if (regen && regen > 0) s += ` · regen×${regen}`;
      return s;
    }
    case "verify": {
      const stripped = summary.citations_stripped as number | undefined;
      return stripped && stripped > 0
        ? `${stripped} citations stripped`
        : "all citations valid";
    }
  }
}

export function PipelineView({
  stages,
  pending,
}: {
  stages: PipelineStage[];
  pending: boolean;
}) {
  if (stages.length === 0 && !pending) return null;

  // If still pending, append a synthetic "running" stage for the next-expected step
  const display: PipelineStage[] = [...stages];
  if (pending) {
    const last = stages[stages.length - 1];
    const nextStage = inferNextStage(last);
    if (nextStage) {
      display.push({ name: nextStage, status: "running" });
    }
  }

  return (
    <Card className="px-3 py-2">
      <div className="mb-2 text-xs font-medium text-muted-foreground">Pipeline</div>
      <ol className="space-y-1.5">
        {display.map((s, i) => (
          <li key={`${s.name}-${i}`} className="flex items-start gap-2 text-xs">
            <StatusDot status={s.status} />
            <div className="flex-1">
              <div className="flex items-baseline gap-2">
                <span className="font-medium">{STAGE_LABEL[s.name]}</span>
                {s.elapsed_ms != null && (
                  <span className="font-mono text-[10px] text-muted-foreground">
                    {(s.elapsed_ms / 1000).toFixed(1)}s
                  </span>
                )}
              </div>
              {s.detail && (
                <div className="text-[10px] text-muted-foreground">{s.detail}</div>
              )}
            </div>
          </li>
        ))}
      </ol>
    </Card>
  );
}

function inferNextStage(last: PipelineStage | undefined): StageName | null {
  if (!last) return "classify";
  const after: Record<StageName, StageName | null> = {
    classify: "retrieve_single", // best guess; will be replaced when actual stage fires
    decompose: "retrieve_parallel",
    retrieve_single: "synthesize",
    retrieve_parallel: "synthesize",
    synthesize: "verify",
    verify: null,
  };
  return after[last.name] ?? null;
}

function StatusDot({ status }: { status: "running" | "complete" }) {
  if (status === "running") {
    return (
      <span className="relative mt-1 inline-flex h-2 w-2 flex-shrink-0">
        <span className="absolute inset-0 animate-ping rounded-full bg-primary opacity-60" />
        <span className="relative inline-flex h-2 w-2 rounded-full bg-primary" />
      </span>
    );
  }
  return (
    <span className="mt-1 inline-flex h-2 w-2 flex-shrink-0 rounded-full bg-green-500" />
  );
}

export function SubQueriesPanel({ queries }: { queries: string[] }) {
  return (
    <Card className="px-3 py-2">
      <div className="mb-2 flex items-center gap-2 text-xs font-medium text-muted-foreground">
        Sub-queries from decomposition
        <Badge variant="secondary" className="font-mono">
          {queries.length}
        </Badge>
      </div>
      <ul className="space-y-1 text-xs">
        {queries.map((q, i) => (
          <li key={i} className="flex gap-2">
            <span className="text-muted-foreground">{i + 1}.</span>
            <span>{q}</span>
          </li>
        ))}
      </ul>
    </Card>
  );
}
