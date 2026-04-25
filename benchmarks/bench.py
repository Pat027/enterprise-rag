"""Latency + throughput benchmark for the /query endpoint.

Sends a fixed corpus of queries against http://localhost:8088/query, measures
end-to-end wall-clock per query, reports p50/p95/p99/throughput.

Run from repo root after `docker compose up`:

    python benchmarks/bench.py --warmup 1 --runs 5 --concurrency 1

Concurrency > 1 stresses vLLM's continuous batching. Set N high enough that you
can see the difference between sequential vs batched serving in the throughput
column.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass

import httpx

DEFAULT_URL = "http://localhost:8088/query"

QUERIES = [
    "What is the STAR method and how should I use it in interviews?",
    "What are common interview mistakes to avoid?",
    "How should I answer 'tell me about yourself'?",
    "What questions should I ask the interviewer at the end?",
    "How do I describe a difficult stakeholder situation?",
    "What motivates me in my work — how should I frame this?",
    "How do I explain why I'm leaving my current role tactfully?",
    "What are my biggest strengths and how to present them?",
]


@dataclass
class Result:
    query: str
    latency_s: float
    answer_chars: int
    citations: int
    blocked_by: str | None
    error: str | None = None


async def one_request(client: httpx.AsyncClient, query: str) -> Result:
    t0 = time.perf_counter()
    try:
        r = await client.post(
            DEFAULT_URL, json={"query": query}, timeout=120.0
        )
        elapsed = time.perf_counter() - t0
        if r.status_code != 200:
            return Result(query, elapsed, 0, 0, None, error=f"HTTP {r.status_code}")
        data = r.json()
        return Result(
            query=query,
            latency_s=elapsed,
            answer_chars=len(data.get("answer") or ""),
            citations=len(data.get("citations") or []),
            blocked_by=data.get("blocked_by"),
        )
    except Exception as e:
        return Result(query, time.perf_counter() - t0, 0, 0, None, error=repr(e))


async def run(runs: int, concurrency: int, warmup: int) -> list[Result]:
    async with httpx.AsyncClient() as client:
        # Warmup — warm the BGE model load, prefix cache, etc.
        if warmup:
            print(f"warming up ({warmup} queries)…", flush=True)
            for i in range(warmup):
                await one_request(client, QUERIES[i % len(QUERIES)])

        # Build the request schedule
        plan = [QUERIES[i % len(QUERIES)] for i in range(runs)]
        print(f"running {runs} queries at concurrency={concurrency}…", flush=True)

        sem = asyncio.Semaphore(concurrency)

        async def _bounded(q: str) -> Result:
            async with sem:
                return await one_request(client, q)

        wall_start = time.perf_counter()
        results = await asyncio.gather(*[_bounded(q) for q in plan])
        wall = time.perf_counter() - wall_start
        return results, wall


def report(results: list[Result], wall: float) -> dict:
    successes = [r for r in results if r.error is None and r.blocked_by is None]
    errors = [r for r in results if r.error is not None]
    blocked = [r for r in results if r.blocked_by is not None]
    latencies = sorted(r.latency_s for r in successes)

    def pct(p: float) -> float:
        if not latencies:
            return 0.0
        idx = max(0, min(len(latencies) - 1, int(round(p * (len(latencies) - 1)))))
        return latencies[idx]

    summary = {
        "total_requests": len(results),
        "successes": len(successes),
        "blocked": len(blocked),
        "errors": len(errors),
        "wall_clock_s": round(wall, 2),
        "throughput_qps": round(len(successes) / wall, 3) if wall > 0 else 0.0,
        "latency_s": {
            "min": round(min(latencies), 2) if latencies else None,
            "p50": round(pct(0.50), 2),
            "p95": round(pct(0.95), 2),
            "p99": round(pct(0.99), 2),
            "max": round(max(latencies), 2) if latencies else None,
            "mean": round(statistics.fmean(latencies), 2) if latencies else None,
        },
        "answer_chars_mean": round(
            statistics.fmean(r.answer_chars for r in successes), 1
        )
        if successes
        else 0,
        "citations_mean": round(
            statistics.fmean(r.citations for r in successes), 1
        )
        if successes
        else 0,
    }
    if errors:
        summary["error_samples"] = [r.error for r in errors[:3]]
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=8, help="number of timed queries")
    p.add_argument("--concurrency", type=int, default=1, help="parallel inflight")
    p.add_argument("--warmup", type=int, default=1, help="warmup queries (untimed)")
    p.add_argument("--label", type=str, default="", help="tag for this run")
    args = p.parse_args()

    results, wall = asyncio.run(run(args.runs, args.concurrency, args.warmup))
    summary = report(results, wall)
    if args.label:
        summary["label"] = args.label
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
