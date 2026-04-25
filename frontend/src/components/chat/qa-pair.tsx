"use client";

import * as React from "react";
import { Loader2, ShieldAlert } from "lucide-react";
import type { QAPair } from "@/store/chat";
import { AnswerText } from "./answer-text";
import { CitationList } from "./citation-list";
import { StageBadges } from "./stage-badges";

export function QAPairView({ pair }: { pair: QAPair }) {
  const handleCitationClick = React.useCallback(
    (index: number) => {
      const id = `citation-${pair.id}-${index}`;
      const el = document.getElementById(id);
      if (!el) return;
      el.scrollIntoView({ behavior: "smooth", block: "center" });
      el.classList.remove("citation-flash");
      // Force reflow to restart animation.
      void el.offsetWidth;
      el.classList.add("citation-flash");
    },
    [pair.id]
  );

  const showThinking = pair.streaming && !pair.answer && !pair.blockedBy && !pair.error;

  return (
    <div className="flex flex-col gap-3">
      {/* Question (right-aligned, gray) */}
      <div className="flex justify-end">
        <div className="max-w-[80%] rounded-2xl rounded-br-md bg-muted px-4 py-2.5 text-[15px] leading-6 text-foreground">
          {pair.question}
        </div>
      </div>

      {/* Answer */}
      <div className="flex flex-col gap-2">
        <StageBadges stages={pair.stages} />

        {pair.blockedBy ? (
          <div className="rounded-md border border-red-500/40 bg-red-500/5 p-4 text-sm text-red-700 dark:text-red-300">
            <div className="flex items-center gap-2 font-medium">
              <ShieldAlert className="h-4 w-4" />
              Blocked by {pair.blockedBy} safety layer
            </div>
            {pair.refusal && <p className="mt-1.5 leading-6">{pair.refusal}</p>}
          </div>
        ) : pair.error ? (
          <div className="rounded-md border border-red-500/40 bg-red-500/5 p-3 text-sm text-red-700 dark:text-red-300">
            {pair.error}
          </div>
        ) : showThinking ? (
          <div className="inline-flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            thinking…
          </div>
        ) : (
          <AnswerText
            pairId={pair.id}
            text={pair.answer}
            citationCount={pair.citations.length}
            onCitationClick={handleCitationClick}
          />
        )}

        {pair.citations.length > 0 && !pair.blockedBy && (
          <CitationList pairId={pair.id} citations={pair.citations} />
        )}
      </div>
    </div>
  );
}
