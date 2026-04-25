import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center rounded-md border px-2 py-0.5 text-xs font-medium transition-colors",
  {
    variants: {
      variant: {
        default: "border-transparent bg-foreground text-background",
        outline: "border-border text-foreground",
        muted: "border-transparent bg-muted text-muted-foreground",
        accent: "border-transparent bg-accent/15 text-accent",
        success: "border-transparent bg-emerald-500/10 text-emerald-700 dark:text-emerald-400",
        danger: "border-transparent bg-red-500/10 text-red-700 dark:text-red-400",
      },
    },
    defaultVariants: { variant: "default" },
  }
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {}

export function Badge({ className, variant, ...props }: BadgeProps) {
  return <span className={cn(badgeVariants({ variant }), className)} {...props} />;
}
