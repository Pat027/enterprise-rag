"""LangGraph node implementations for the RAG pipeline.

Each node is wrapped in a manual OpenTelemetry span so the trace tree mirrors
the LangGraph stage diagram. We also record per-stage Prometheus histograms.
"""

from __future__ import annotations

import time

from .. import generation, retrieval, safety
from ..config import get_settings
from ..observability import (
    RAG_GENERATION_DURATION_SECONDS,
    RAG_RETRIEVAL_DURATION_SECONDS,
    get_tracer,
)
from .state import RAGState


def _span(name: str):
    """Return a context manager for a span (no-op if OTel disabled)."""
    t = get_tracer()
    if t is None:
        # Provide a minimal context manager fallback
        from contextlib import nullcontext

        return nullcontext(None)
    return t.start_as_current_span(name)


def input_safety(state: RAGState) -> RAGState:
    with _span("input_safety") as span:
        if span is not None:
            span.set_attribute("query.length", len(state.get("query", "")))
            span.set_attribute("safety.layer", "input")
        verdict = safety.check_input(state["query"])
        if verdict is not None:
            if span is not None:
                span.set_attribute("safety.allowed", False)
                span.set_attribute("safety.blocked_layer", verdict.layer)
            return {
                "blocking_verdict": verdict,
                "refusal": (
                    "Your request was blocked by the safety layer "
                    f"({verdict.layer}): {verdict.reason}"
                ),
            }
        if span is not None:
            span.set_attribute("safety.allowed", True)
        return {}


def retrieve(state: RAGState) -> RAGState:
    s = get_settings()
    with _span("retrieve") as span:
        if span is not None:
            span.set_attribute("retrieval.mode", s.retrieval_mode)
            span.set_attribute("query.length", len(state.get("query", "")))
        t0 = time.perf_counter()
        passages = retrieval.retrieve(state["query"])
        elapsed = time.perf_counter() - t0
        RAG_RETRIEVAL_DURATION_SECONDS.labels(mode=s.retrieval_mode).observe(elapsed)
        if span is not None:
            span.set_attribute("passages.count", len(passages))
        return {
            "passages": passages,
            "context_text": generation.format_context(passages),
        }


def generate(state: RAGState) -> RAGState:
    with _span("generate") as span:
        passages = state.get("passages", [])
        if span is not None:
            span.set_attribute("passages.count", len(passages))
        t0 = time.perf_counter()
        answer = generation.generate(state["query"], passages)
        elapsed = time.perf_counter() - t0
        RAG_GENERATION_DURATION_SECONDS.observe(elapsed)
        if span is not None:
            span.set_attribute("tokens.generated", len(answer.split()))
        return {"answer": answer}


def output_safety(state: RAGState) -> RAGState:
    with _span("output_safety") as span:
        if span is not None:
            span.set_attribute("safety.layer", "output")
        verdict = safety.check_output(
            state["query"], state.get("context_text", ""), state.get("answer", "")
        )
        if verdict is not None:
            if span is not None:
                span.set_attribute("safety.allowed", False)
                span.set_attribute("safety.blocked_layer", verdict.layer)
            return {
                "blocking_verdict": verdict,
                "refusal": (
                    "The model's response was withheld by the safety layer "
                    f"({verdict.layer}): {verdict.reason}"
                ),
            }
        if span is not None:
            span.set_attribute("safety.allowed", True)
        return {}


def should_continue_after_input(state: RAGState) -> str:
    return "blocked" if state.get("blocking_verdict") else "ok"


def should_continue_after_output(state: RAGState) -> str:
    return "blocked" if state.get("blocking_verdict") else "ok"
