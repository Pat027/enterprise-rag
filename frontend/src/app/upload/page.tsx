"use client";

import * as React from "react";
import { useDropzone } from "react-dropzone";
import { toast } from "sonner";
import { CheckCircle2, FileText, Loader2, UploadCloud, XCircle } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ingestFile } from "@/lib/api";
import { cn } from "@/lib/utils";

type Status = "uploading" | "ingested" | "error";

interface Item {
  id: string;
  name: string;
  size: number;
  status: Status;
  chunks?: number;
  error?: string;
}

export default function UploadPage() {
  const [items, setItems] = React.useState<Item[]>([]);

  const onDrop = React.useCallback(async (files: File[]) => {
    const newItems: Item[] = files.map((f) => ({
      id: `${f.name}-${f.lastModified}-${Math.random().toString(36).slice(2, 6)}`,
      name: f.name,
      size: f.size,
      status: "uploading",
    }));
    setItems((prev) => [...newItems, ...prev]);

    await Promise.all(
      files.map(async (file, i) => {
        const id = newItems[i].id;
        try {
          const res = await ingestFile(file);
          setItems((prev) =>
            prev.map((it) =>
              it.id === id ? { ...it, status: "ingested", chunks: res.chunks_indexed } : it
            )
          );
          toast.success(`Ingested ${file.name} (${res.chunks_indexed} chunks)`);
        } catch (err) {
          const msg = err instanceof Error ? err.message : "Upload failed";
          setItems((prev) =>
            prev.map((it) => (it.id === id ? { ...it, status: "error", error: msg } : it))
          );
          toast.error(`${file.name}: ${msg}`);
        }
      })
    );
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop });

  return (
    <div className="mx-auto w-full max-w-[860px] px-6 py-10">
      <header className="mb-6">
        <h1 className="text-2xl font-semibold tracking-tight">Upload documents</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Drop PDFs, Markdown, or text files. Each file will be chunked,
          embedded, and indexed.
        </p>
      </header>

      <Card>
        <CardContent className="p-0">
          <div
            {...getRootProps()}
            className={cn(
              "flex cursor-pointer flex-col items-center justify-center gap-3 rounded-lg border-2 border-dashed border-border px-6 py-16 text-center transition-colors",
              isDragActive && "border-accent bg-accent/5"
            )}
          >
            <input {...getInputProps()} />
            <UploadCloud
              className={cn(
                "h-8 w-8 text-muted-foreground transition-colors",
                isDragActive && "text-accent"
              )}
            />
            <div>
              <p className="text-sm font-medium">
                {isDragActive ? "Drop to upload" : "Drag files here or click to browse"}
              </p>
              <p className="mt-1 text-xs text-muted-foreground">
                Multiple files supported · Indexed in this session
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      <section className="mt-8">
        <h2 className="mb-3 text-sm font-semibold tracking-tight text-muted-foreground">
          Ingested in this session
        </h2>
        {items.length === 0 ? (
          <Card>
            <CardHeader>
              <CardTitle className="text-sm font-medium">No files yet</CardTitle>
              <CardDescription>
                Files you upload here will be listed below with their chunk counts.
              </CardDescription>
            </CardHeader>
          </Card>
        ) : (
          <ul className="flex flex-col gap-2">
            {items.map((it) => (
              <li
                key={it.id}
                className="flex items-center gap-3 rounded-md border border-border bg-card px-4 py-3"
              >
                <FileText className="h-4 w-4 shrink-0 text-muted-foreground" />
                <div className="min-w-0 flex-1">
                  <div className="truncate text-sm font-medium">{it.name}</div>
                  <div className="text-xs text-muted-foreground">
                    {formatBytes(it.size)}
                    {it.error && (
                      <span className="ml-2 text-red-600 dark:text-red-400">
                        · {it.error}
                      </span>
                    )}
                  </div>
                </div>
                <StatusBadge item={it} />
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}

function StatusBadge({ item }: { item: Item }) {
  if (item.status === "uploading")
    return (
      <Badge variant="muted" className="gap-1.5">
        <Loader2 className="h-3 w-3 animate-spin" /> uploading
      </Badge>
    );
  if (item.status === "ingested")
    return (
      <Badge variant="success" className="gap-1.5">
        <CheckCircle2 className="h-3 w-3" /> {item.chunks ?? 0} chunks
      </Badge>
    );
  return (
    <Badge variant="danger" className="gap-1.5">
      <XCircle className="h-3 w-3" /> error
    </Badge>
  );
}

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
