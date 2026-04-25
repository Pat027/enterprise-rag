import Link from "next/link";
import { ThemeToggle } from "@/components/theme-toggle";

export function SiteHeader() {
  return (
    <header className="sticky top-0 z-30 w-full border-b border-border bg-background/80 backdrop-blur-md">
      <div className="mx-auto flex h-14 w-full max-w-[1200px] items-center justify-between px-6">
        <div className="flex items-center gap-6">
          <Link href="/" className="flex items-center gap-2">
            <span className="inline-block h-2.5 w-2.5 rounded-sm bg-accent" aria-hidden />
            <span className="text-sm font-semibold tracking-tight">Enterprise RAG</span>
          </Link>
          <nav className="hidden items-center gap-1 text-sm text-muted-foreground sm:flex">
            <Link href="/" className="rounded-md px-2 py-1 hover:text-foreground">Chat</Link>
            <Link href="/upload" className="rounded-md px-2 py-1 hover:text-foreground">Upload</Link>
            <Link href="/settings" className="rounded-md px-2 py-1 hover:text-foreground">Settings</Link>
          </nav>
        </div>
        <ThemeToggle />
      </div>
    </header>
  );
}
