"""Defense-in-depth safety orchestrator.

Layer 1 (input) : OpenAI Moderation — fast, cheap, catches obvious abuse.
Layer 2 (input) : LlamaGuard 3 — policy-aware classification.
Layer 3 (output): LlamaGuard 3 — applied to model output.
Layer 4 (output): Constitutional critic — LLM-as-judge against written principles.

Each layer is independently toggleable via settings. Failing open on transient
errors is intentional — safety layers should never bring down the whole system.
"""

from __future__ import annotations

import time
from contextlib import nullcontext

import structlog

from ..config import get_settings
from ..observability import RAG_CRITIC_DURATION_SECONDS, get_tracer
from . import constitutional, llamaguard, openai_moderation
from .types import SafetyVerdict

log = structlog.get_logger()


def _span(name: str):
    t = get_tracer()
    if t is None:
        return nullcontext(None)
    return t.start_as_current_span(name)


def check_input(text: str) -> SafetyVerdict | None:
    """Return the first blocking verdict, or None if all enabled layers pass."""
    s = get_settings()

    if s.safety_openai_moderation:
        try:
            v = openai_moderation.moderate(text)
            if not v.allowed:
                return v
        except Exception as e:
            log.warning("openai_moderation_failed", error=str(e))

    if s.safety_llamaguard:
        try:
            v = llamaguard.check_user_input(text)
            if not v.allowed:
                return v
        except Exception as e:
            log.warning("llamaguard_input_failed", error=str(e))

    return None


def check_output(query: str, context: str, response: str) -> SafetyVerdict | None:
    """Run output-side safety. Returns first blocking verdict, or None."""
    s = get_settings()

    if s.safety_llamaguard:
        try:
            v = llamaguard.check_assistant_output(query, response)
            if not v.allowed:
                return v
        except Exception as e:
            log.warning("llamaguard_output_failed", error=str(e))

    if s.safety_constitutional:
        try:
            with _span("constitutional_critic") as sp:
                if sp is not None:
                    sp.set_attribute("safety.layer", "constitutional")
                t0 = time.perf_counter()
                v = constitutional.critique(query, context, response)
                RAG_CRITIC_DURATION_SECONDS.observe(time.perf_counter() - t0)
                if sp is not None:
                    sp.set_attribute("safety.allowed", v.allowed)
            if not v.allowed:
                return v
        except Exception as e:
            log.warning("constitutional_failed", error=str(e))

    return None
