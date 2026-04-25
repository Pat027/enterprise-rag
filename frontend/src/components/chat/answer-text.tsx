"use client";

import * as React from "react";
import { cn } from "@/lib/utils";

interface AnswerTextProps {
  text: string;
  pairId: string;
  citationCount: number;
  onCitationClick: (index: number) => void;
}

/**
 * Renders the answer with [N] tokens turned into clickable monospace pills
 * that scroll to / flash the matching citation card.
 */
export function AnswerText({ text, pairId, citationCount, onCitationClick }: AnswerTextProps) {
  const parts = React.useMemo(() => {
    const out: Array<{ type: "text"; value: string } | { type: "cite"; index: number }> = [];
    const re = /\[(\d+)\]/g;
    let last = 0;
    let m: RegExpExecArray | null;
    while ((m = re.exec(text)) !== null) {
      if (m.index > last) out.push({ type: "text", value: text.slice(last, m.index) });
      const idx = parseInt(m[1], 10);
      out.push({ type: "cite", index: idx });
      last = m.index + m[0].length;
    }
    if (last < text.length) out.push({ type: "text", value: text.slice(last) });
    return out;
  }, [text]);

  return (
    <div className="whitespace-pre-wrap text-[15px] leading-7 text-foreground">
      {parts.map((p, i) =>
        p.type === "text" ? (
          <React.Fragment key={i}>{p.value}</React.Fragment>
        ) : (
          <button
            key={i}
            type="button"
            onClick={() => onCitationClick(p.index)}
            disabled={p.index < 1 || p.index > citationCount}
            className={cn(
              "mx-0.5 inline-flex h-5 min-w-[20px] -translate-y-px items-center justify-center rounded-md border border-border bg-muted px-1 font-mono text-[11px] font-medium text-foreground shadow-[0_1px_0_rgba(0,0,0,0.04)] hover:bg-accent hover:text-accent-foreground hover:border-accent disabled:cursor-not-allowed disabled:opacity-40"
            )}
            aria-label={`Citation ${p.index}`}
            data-citation-link={`${pairId}-${p.index}`}
          >
            {p.index}
          </button>
        )
      )}
    </div>
  );
}
