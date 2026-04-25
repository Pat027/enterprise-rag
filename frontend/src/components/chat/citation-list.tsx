"use client";

import * as React from "react";
import { ChevronDown, FileText } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Citation } from "@/lib/api";
import { Badge } from "@/components/ui/badge";

interface CitationListProps {
  pairId: string;
  citations: Citation[];
}

export function CitationList({ pairId, citations }: CitationListProps) {
  const [open, setOpen] = React.useState(false);

  if (!citations.length) return null;

  return (
    <div className="mt-3 rounded-md border border-border bg-card">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between gap-2 px-3 py-2 text-left text-xs font-medium text-muted-foreground hover:text-foreground"
      >
        <span className="inline-flex items-center gap-2">
          <FileText className="h-3.5 w-3.5" />
          Sources ({citations.length})
        </span>
        <ChevronDown className={cn("h-4 w-4 transition-transform", open && "rotate-180")} />
      </button>
      {open && (
        <ul className="divide-y divide-border border-t border-border">
          {citations.map((c) => (
            <li
              key={`${pairId}-${c.index}`}
              id={`citation-${pairId}-${c.index}`}
              data-citation-card={`${pairId}-${c.index}`}
              className="flex flex-col gap-1.5 px-3 py-3 text-sm"
            >
              <div className="flex items-baseline gap-2">
                <Badge variant="muted" className="font-mono">
                  [{c.index}]
                </Badge>
                <span className="truncate font-medium">{c.source}</span>
                {c.page != null && (
                  <span className="ml-auto shrink-0 text-xs text-muted-foreground">
                    p.{c.page}
                  </span>
                )}
              </div>
              {c.section_path?.length > 0 && (
                <div className="text-xs text-muted-foreground">
                  {c.section_path.join(" › ")}
                </div>
              )}
              <div className="text-[11px] font-mono text-muted-foreground">
                rerank {c.rerank_score.toFixed(3)}
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
