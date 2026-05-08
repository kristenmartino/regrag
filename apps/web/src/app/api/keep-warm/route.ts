// Cron-pinged endpoint to keep the Railway backend container warm.
//
// Vercel hits this every 5 minutes (configured in vercel.json). It pings
// the Railway /health endpoint, which has the side effect of keeping the
// container alive — Railway's free tier sleeps after idle, and the first
// request after sleep takes 30-60s to cold-start, which would tank the
// demo's first-impression UX.
//
// Returns the upstream response status + a short summary so the Vercel
// dashboard's cron logs are scannable.

import { NextResponse } from "next/server";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "";

export async function GET() {
  if (!API_BASE) {
    return NextResponse.json(
      { ok: false, reason: "NEXT_PUBLIC_API_URL not set" },
      { status: 500 },
    );
  }
  const t0 = Date.now();
  try {
    const res = await fetch(`${API_BASE}/health`, {
      // Don't cache — we want a real network round-trip to wake the container
      cache: "no-store",
      // Reasonable timeout — if the cold start takes >25s we still want
      // to know about it
      signal: AbortSignal.timeout(30_000),
    });
    const elapsedMs = Date.now() - t0;
    const body = await res.text();
    return NextResponse.json({
      ok: res.ok,
      upstream_status: res.status,
      upstream_body: body.slice(0, 200),
      elapsed_ms: elapsedMs,
      timestamp: new Date().toISOString(),
    });
  } catch (e) {
    return NextResponse.json(
      {
        ok: false,
        error: e instanceof Error ? e.message : String(e),
        elapsed_ms: Date.now() - t0,
      },
      { status: 502 },
    );
  }
}
