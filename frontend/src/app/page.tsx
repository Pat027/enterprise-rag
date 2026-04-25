"use client";

import * as React from "react";
import { toast } from "sonner";
import { ScrollArea } from "@/components/ui/scroll-area";
import { ChatComposer } from "@/components/chat/chat-composer";
import { QAPairView } from "@/components/chat/qa-pair";
import { useChatStore } from "@/store/chat";
import { streamQuery } from "@/lib/api";

export default function HomePage() {
  const { pairs, addPair, updatePair, appendToken, setStage } = useChatStore();
  const [busy, setBusy] = React.useState(false);
  const abortRef = React.useRef<AbortController | null>(null);
  const scrollRef = React.useRef<HTMLDivElement | null>(null);

  React.useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [pairs]);

  const handleSubmit = async (question: string) => {
    const id = addPair(question);
    setBusy(true);
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    try {
      for await (const ev of streamQuery(question, ctrl.signal)) {
        switch (ev.type) {
          case "safety_check":
            if (ev.stage === "input") {
              setStage(id, "input_safety", ev.verdict);
              if (ev.verdict === "passed") setStage(id, "retrieve", "pending");
              if (ev.verdict === "blocked") {
                updatePair(id, {
                  blockedBy: "input",
                  refusal: ev.reason ?? "Request blocked by input safety filter.",
                });
              }
            } else {
              setStage(id, "output_safety", ev.verdict);
              if (ev.verdict === "blocked") {
                updatePair(id, {
                  blockedBy: "output",
                  refusal: ev.reason ?? "Response blocked by output safety filter.",
                });
              }
            }
            break;
          case "passages":
            setStage(id, "retrieve", "done");
            setStage(id, "generate", "pending");
            updatePair(id, { citations: ev.citations });
            break;
          case "token":
            appendToken(id, ev.text);
            break;
          case "done":
            setStage(id, "generate", "done");
            updatePair(id, {
              citations: ev.citations,
              ...(ev.answer ? { answer: ev.answer } : {}),
            });
            break;
          case "error":
            updatePair(id, { error: ev.message });
            toast.error(ev.message);
            break;
        }
      }
    } catch (err) {
      const aborted = (err as Error).name === "AbortError";
      if (!aborted) {
        const message = err instanceof Error ? err.message : "Request failed";
        updatePair(id, { error: message });
        toast.error(message);
      }
    } finally {
      updatePair(id, { streaming: false });
      setBusy(false);
      abortRef.current = null;
    }
  };

  const handleAbort = () => {
    abortRef.current?.abort();
  };

  return (
    <div className="mx-auto flex h-[calc(100vh-3.5rem)] w-full max-w-[860px] flex-col px-6 py-6">
      <ScrollArea ref={scrollRef} className="flex-1 pr-1">
        {pairs.length === 0 ? (
          <EmptyState />
        ) : (
          <div className="flex flex-col gap-10 pb-6">
            {pairs.map((p) => (
              <QAPairView key={p.id} pair={p} />
            ))}
          </div>
        )}
      </ScrollArea>
      <div className="pt-4">
        <ChatComposer onSubmit={handleSubmit} onAbort={handleAbort} busy={busy} />
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-3 py-24 text-center">
      <div className="inline-block h-2 w-2 rounded-sm bg-accent" aria-hidden />
      <h1 className="text-2xl font-semibold tracking-tight">
        Ask your documents
      </h1>
      <p className="max-w-md text-sm text-muted-foreground">
        Upload a PDF or text file from the Upload page, then ask grounded
        questions here. Answers will cite the exact passages they were derived
        from.
      </p>
    </div>
  );
}
