import { Check, Loader2, ShieldAlert } from "lucide-react";
import { cn } from "@/lib/utils";
import type { StageState } from "@/store/chat";

const ORDER: { key: keyof StageState; label: string }[] = [
  { key: "input_safety", label: "input safety" },
  { key: "retrieve", label: "retrieve" },
  { key: "generate", label: "generate" },
  { key: "output_safety", label: "output safety" },
];

export function StageBadges({ stages }: { stages: StageState }) {
  return (
    <div className="flex flex-wrap items-center gap-1.5 text-[11px] font-mono text-muted-foreground">
      {ORDER.map(({ key, label }) => {
        const v = stages[key];
        if (!v) return null;
        const blocked = v === "blocked";
        const passed = v === "passed" || v === "done";
        return (
          <span
            key={key}
            className={cn(
              "inline-flex items-center gap-1 rounded-md border border-border px-2 py-0.5 lowercase",
              blocked && "border-red-500/40 text-red-600 dark:text-red-400",
              passed && "border-accent/30 text-accent"
            )}
          >
            {v === "pending" ? (
              <Loader2 className="h-3 w-3 animate-spin" />
            ) : blocked ? (
              <ShieldAlert className="h-3 w-3" />
            ) : (
              <Check className="h-3 w-3" />
            )}
            {label}
          </span>
        );
      })}
    </div>
  );
}
