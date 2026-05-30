"use client";

import { FormEvent, useState } from "react";

import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { setAuditToken } from "@/lib/api";

/**
 * Unlock prompt for the admin-only audit pages (issue #11). The API requires a
 * bearer token, and we never bake it into the client bundle. The operator
 * enters it here; it's held in sessionStorage for the tab and sent as
 * `Authorization: Bearer` on each audit request.
 */
export function AuditGate({
  disabled,
  onUnlock,
}: {
  disabled?: boolean;
  onUnlock: () => void;
}) {
  const [token, setToken] = useState("");

  function submit(e: FormEvent) {
    e.preventDefault();
    const trimmed = token.trim();
    if (!trimmed) return;
    setAuditToken(trimmed);
    onUnlock();
  }

  return (
    <div className="flex min-h-[60dvh] items-center justify-center px-6">
      <Card className="w-full max-w-sm px-6 py-6">
        <h2 className="text-base font-semibold tracking-tight">Audit access</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          {disabled
            ? "The audit log is disabled on this deployment. Set REGRAG_AUDIT_TOKEN on the API to enable it."
            : "The audit log is admin-only. Enter the audit token to continue."}
        </p>
        {!disabled && (
          <form onSubmit={submit} className="mt-4 space-y-3">
            <input
              type="password"
              autoFocus
              value={token}
              onChange={(e) => setToken(e.target.value)}
              placeholder="Audit token"
              aria-label="Audit token"
              className="w-full rounded-md border bg-background px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-ring"
            />
            <Button type="submit" className="w-full" disabled={!token.trim()}>
              Unlock
            </Button>
          </form>
        )}
      </Card>
    </div>
  );
}
