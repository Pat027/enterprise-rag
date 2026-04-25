"use client";

import * as React from "react";
import { CheckCircle2, Loader2, XCircle } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { API_KEY, API_URL, fetchHealth, type HealthResponse } from "@/lib/api";

type State =
  | { kind: "loading" }
  | { kind: "ok"; data: HealthResponse }
  | { kind: "error"; message: string };

export default function SettingsPage() {
  const [state, setState] = React.useState<State>({ kind: "loading" });

  const refresh = React.useCallback(async () => {
    setState({ kind: "loading" });
    try {
      const data = await fetchHealth();
      setState({ kind: "ok", data });
    } catch (err) {
      setState({
        kind: "error",
        message: err instanceof Error ? err.message : "Health check failed",
      });
    }
  }, []);

  React.useEffect(() => {
    refresh();
  }, [refresh]);

  return (
    <div className="mx-auto w-full max-w-[860px] px-6 py-10">
      <header className="mb-6">
        <h1 className="text-2xl font-semibold tracking-tight">Settings</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Connection and backend status.
        </p>
      </header>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Backend</CardTitle>
          <CardDescription>
            Configured via <code className="font-mono">NEXT_PUBLIC_API_URL</code>.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          <Row label="API base URL">
            <code className="rounded bg-muted px-2 py-1 font-mono text-xs">{API_URL}</code>
          </Row>
          <Row label="Bearer token">
            {API_KEY ? (
              <Badge variant="muted">configured</Badge>
            ) : (
              <Badge variant="outline">not set</Badge>
            )}
          </Row>
          <Row label="Health">
            {state.kind === "loading" && (
              <Badge variant="muted" className="gap-1.5">
                <Loader2 className="h-3 w-3 animate-spin" /> checking
              </Badge>
            )}
            {state.kind === "ok" && (
              <Badge variant="success" className="gap-1.5">
                <CheckCircle2 className="h-3 w-3" /> {state.data.status}
              </Badge>
            )}
            {state.kind === "error" && (
              <Badge variant="danger" className="gap-1.5">
                <XCircle className="h-3 w-3" /> {state.message}
              </Badge>
            )}
          </Row>
          {state.kind === "ok" && (
            <Row label="Package version">
              <code className="rounded bg-muted px-2 py-1 font-mono text-xs">
                {state.data.package_version}
              </code>
            </Row>
          )}
          <button
            type="button"
            onClick={refresh}
            className="self-start text-xs text-accent hover:underline"
          >
            Re-check
          </button>
        </CardContent>
      </Card>
    </div>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-4 border-b border-border pb-3 last:border-b-0 last:pb-0">
      <span className="text-sm text-muted-foreground">{label}</span>
      <span className="flex items-center gap-2">{children}</span>
    </div>
  );
}
