"use client";

import * as React from "react";
import { ArrowUp, Square } from "lucide-react";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";

interface ChatComposerProps {
  onSubmit: (q: string) => void;
  onAbort?: () => void;
  busy: boolean;
}

export function ChatComposer({ onSubmit, onAbort, busy }: ChatComposerProps) {
  const [value, setValue] = React.useState("");
  const ref = React.useRef<HTMLTextAreaElement | null>(null);

  const submit = () => {
    const q = value.trim();
    if (!q || busy) return;
    onSubmit(q);
    setValue("");
    if (ref.current) ref.current.style.height = "auto";
  };

  const onKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  React.useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 240) + "px";
  }, [value]);

  return (
    <div className="rounded-xl border border-border bg-card p-2 shadow-sm focus-within:border-accent/50 focus-within:ring-2 focus-within:ring-accent/20">
      <div className="flex items-end gap-2">
        <Textarea
          ref={ref}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={onKey}
          placeholder="Ask a question about your documents…"
          rows={1}
          className="border-0 bg-transparent shadow-none focus-visible:ring-0 focus-visible:ring-offset-0 px-2"
        />
        {busy && onAbort ? (
          <Button size="icon" variant="outline" aria-label="Stop" onClick={onAbort}>
            <Square className="h-3.5 w-3.5 fill-current" />
          </Button>
        ) : (
          <Button
            size="icon"
            variant="accent"
            aria-label="Send"
            disabled={!value.trim()}
            onClick={submit}
          >
            <ArrowUp className="h-4 w-4" />
          </Button>
        )}
      </div>
      <div className="px-2 pt-1.5 text-[11px] text-muted-foreground">
        Enter to send · Shift+Enter for new line
      </div>
    </div>
  );
}
