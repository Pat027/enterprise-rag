export const API_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "http://localhost:8088";

export const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "";

export function authHeaders(extra: Record<string, string> = {}): HeadersInit {
  const headers: Record<string, string> = { ...extra };
  if (API_KEY) headers["Authorization"] = `Bearer ${API_KEY}`;
  return headers;
}

export interface Citation {
  index: number;
  source: string;
  page: number | null;
  section_path: string[];
  rerank_score: number;
}

export interface QueryResponse {
  answer: string;
  citations: Citation[];
  blocked_by: string | null;
  refusal: string | null;
}

export interface HealthResponse {
  status: string;
  package_version: string;
}

export interface IngestResponse {
  source: string;
  chunks_indexed: number;
}

export type StreamEvent =
  | { type: "safety_check"; stage: "input" | "output"; verdict: "passed" | "blocked"; reason?: string }
  | { type: "passages"; citations: Citation[] }
  | { type: "token"; text: string }
  | { type: "done"; answer: string; citations: Citation[] }
  | { type: "error"; message: string };

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_URL}/health`, { headers: authHeaders() });
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}

export async function ingestFile(file: File): Promise<IngestResponse> {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(`${API_URL}/ingest`, {
    method: "POST",
    headers: authHeaders(),
    body: fd,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Ingest failed (${res.status}): ${text || res.statusText}`);
  }
  return res.json();
}

export async function* streamQuery(
  query: string,
  signal?: AbortSignal
): AsyncGenerator<StreamEvent, void, void> {
  const res = await fetch(`${API_URL}/query/stream`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ query }),
    signal,
  });
  if (!res.ok || !res.body) {
    const text = await res.text().catch(() => "");
    throw new Error(`Stream failed (${res.status}): ${text || res.statusText}`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) !== -1) {
      const raw = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      const line = raw
        .split("\n")
        .find((l) => l.startsWith("data:"));
      if (!line) continue;
      const json = line.slice(5).trim();
      if (!json) continue;
      try {
        yield JSON.parse(json) as StreamEvent;
      } catch {
        // ignore malformed event
      }
    }
  }
}
