"""OpenTelemetry tracing + Prometheus metrics setup.

The story: docs/vllm-optimization.md explains how we *tuned* vLLM (FP8 weights,
FP8 KV cache, prefix caching, speculative decoding). This module wires up the
runtime *observability* so we can confirm those optimizations behave under load:

    - OTLP traces to an OpenTelemetry Collector (→ Tempo for storage in Grafana)
    - Auto-instrumented FastAPI + httpx (so calls to vLLM are traced end-to-end)
    - A small set of custom Prometheus counters/histograms scoped to RAG stages
    - A module-level tracer + meter exposed for manual span/metric creation

Importing this module is cheap. Calling ``setup_otel(...)`` is the side-effecty
bit and is idempotent.
"""

from __future__ import annotations

import logging
from typing import Any

from prometheus_client import Counter, Histogram

log = logging.getLogger(__name__)


# ── Module-level tracer (no-op until setup_otel runs) ────────────────────────
try:
    from opentelemetry import trace

    tracer = trace.get_tracer("enterprise_rag")
except Exception:  # pragma: no cover — OTel optional at import time
    trace = None  # type: ignore[assignment]
    tracer = None  # type: ignore[assignment]


# ── Prometheus metrics — names match dashboards in deploy/observability ──────
RAG_REQUEST_TOTAL = Counter(
    "rag_request_total",
    "Total RAG requests by endpoint and status.",
    labelnames=("endpoint", "status"),
)

RAG_SAFETY_BLOCK_TOTAL = Counter(
    "rag_safety_block_total",
    "Total safety blocks broken down by which layer fired.",
    labelnames=("layer",),
)

RAG_RETRIEVAL_DURATION_SECONDS = Histogram(
    "rag_retrieval_duration_seconds",
    "Time spent in the retrieval stage, by mode (dense|bm25|hybrid).",
    labelnames=("mode",),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

RAG_GENERATION_DURATION_SECONDS = Histogram(
    "rag_generation_duration_seconds",
    "Time spent in the LLM generation stage.",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

RAG_CRITIC_DURATION_SECONDS = Histogram(
    "rag_critic_duration_seconds",
    "Time spent in the constitutional critic stage.",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)


def setup_otel(
    service_name: str = "enterprise-rag-api",
    otlp_endpoint: str = "http://otel-collector:4317",
    fastapi_app: Any | None = None,
) -> Any:
    """Configure the global TracerProvider and auto-instrument FastAPI+httpx.

    Returns the module-level tracer so callers can grab it in one line:

        tracer = setup_otel(...)
        with tracer.start_as_current_span("custom"): ...

    Safe to call multiple times — OpenTelemetry's set_tracer_provider is
    idempotent in practice (later calls are no-ops with a warning).
    """
    global tracer
    try:
        from opentelemetry import trace as _trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as exc:  # pragma: no cover
        log.warning("otel_import_failed", exc_info=exc)
        return None

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    _trace.set_tracer_provider(provider)

    # httpx instrumentation covers OpenAI SDK → vLLM calls (it uses httpx under
    # the hood). FastAPI instrumentation gives one server span per request.
    HTTPXClientInstrumentor().instrument()
    if fastapi_app is not None:
        FastAPIInstrumentor.instrument_app(fastapi_app)

    tracer = _trace.get_tracer("enterprise_rag")
    log.info(
        "otel_initialized service=%s endpoint=%s",
        service_name,
        otlp_endpoint,
    )
    return tracer


def get_tracer() -> Any:
    """Return the module tracer (no-op tracer until setup_otel has run)."""
    return tracer
