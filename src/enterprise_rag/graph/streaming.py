"""Streaming RAG pipeline.

The non-streaming path goes through LangGraph (see ``builder.run_query``).
For SSE we bypass LangGraph and re-implement the same pipeline as an async
generator that yields structured events at each stage.

Event shapes (dicts) emitted:
    {"type": "safety_check", "stage": "input"|"output",
     "verdict": "passed"|"blocked", "reason": str|None}
    {"type": "passages",  "citations": list[dict]}
    {"type": "token",     "text": str}
    {"type": "done",      "answer": str|None,
                          "refusal": str|None,
                          "citations": list[dict],
                          "blocked_by": str|None}
    {"type": "error",     "message": str}
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import nullcontext

import structlog

from .. import generation, retrieval, safety
from ..config import get_settings
from ..observability import (
    RAG_GENERATION_DURATION_SECONDS,
    RAG_RETRIEVAL_DURATION_SECONDS,
    get_tracer,
)


def _span(name: str):
    t = get_tracer()
    if t is None:
        return nullcontext(None)
    return t.start_as_current_span(name)

log = structlog.get_logger()


def _citations_from_passages(passages: list[dict]) -> list[dict]:
    return [
        {
            "index": i + 1,
            "source": p.get("source", "unknown"),
            "page": p.get("page"),
            "section_path": p.get("section_path", []),
            "rerank_score": p.get("rerank_score"),
        }
        for i, p in enumerate(passages)
    ]


async def run_query_stream(query: str) -> AsyncIterator[dict]:
    """Run the full RAG pipeline, yielding SSE-friendly event dicts."""
    s = get_settings()
    try:
        # 1. Input safety (sync → thread)
        with _span("input_safety") as sp:
            if sp is not None:
                sp.set_attribute("query.length", len(query))
                sp.set_attribute("safety.layer", "input")
            in_verdict = await asyncio.to_thread(safety.check_input, query)
            if sp is not None:
                sp.set_attribute("safety.allowed", in_verdict is None)
        if in_verdict is not None:
            yield {
                "type": "safety_check",
                "stage": "input",
                "verdict": "blocked",
                "reason": in_verdict.reason,
            }
            yield {
                "type": "done",
                "answer": None,
                "refusal": (
                    "Your request was blocked by the safety layer "
                    f"({in_verdict.layer}): {in_verdict.reason}"
                ),
                "citations": [],
                "blocked_by": in_verdict.layer,
            }
            return

        yield {
            "type": "safety_check",
            "stage": "input",
            "verdict": "passed",
            "reason": None,
        }

        # 2. Retrieve (sync → thread)
        with _span("retrieve") as sp:
            if sp is not None:
                sp.set_attribute("retrieval.mode", s.retrieval_mode)
            t0 = time.perf_counter()
            passages = await asyncio.to_thread(retrieval.retrieve, query)
            RAG_RETRIEVAL_DURATION_SECONDS.labels(mode=s.retrieval_mode).observe(
                time.perf_counter() - t0
            )
            if sp is not None:
                sp.set_attribute("passages.count", len(passages))
        citations = _citations_from_passages(passages)
        yield {"type": "passages", "citations": citations}

        # 3. Stream generation
        chunks: list[str] = []
        with _span("generate") as sp:
            if sp is not None:
                sp.set_attribute("passages.count", len(passages))
            t0 = time.perf_counter()
            async for token in generation.generate_stream(query, passages):
                chunks.append(token)
                yield {"type": "token", "text": token}
            RAG_GENERATION_DURATION_SECONDS.observe(time.perf_counter() - t0)
            if sp is not None:
                sp.set_attribute("tokens.generated", len(chunks))
        answer = "".join(chunks).strip()

        # 4. Output safety
        context_text = generation.format_context(passages)
        with _span("output_safety") as sp:
            if sp is not None:
                sp.set_attribute("safety.layer", "output")
            out_verdict = await asyncio.to_thread(
                safety.check_output, query, context_text, answer
            )
            if sp is not None:
                sp.set_attribute("safety.allowed", out_verdict is None)
        if out_verdict is not None:
            yield {
                "type": "safety_check",
                "stage": "output",
                "verdict": "blocked",
                "reason": out_verdict.reason,
            }
            yield {
                "type": "done",
                "answer": None,
                "refusal": (
                    "The model's response was withheld by the safety layer "
                    f"({out_verdict.layer}): {out_verdict.reason}"
                ),
                "citations": [],
                "blocked_by": out_verdict.layer,
            }
            return

        yield {
            "type": "safety_check",
            "stage": "output",
            "verdict": "passed",
            "reason": None,
        }

        # 5. Final
        yield {
            "type": "done",
            "answer": answer,
            "refusal": None,
            "citations": citations,
            "blocked_by": None,
        }
    except Exception as exc:  # noqa: BLE001
        log.exception("query_stream_error")
        yield {"type": "error", "message": str(exc)}
