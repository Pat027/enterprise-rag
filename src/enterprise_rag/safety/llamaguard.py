"""LlamaGuard 3 8B served by local vLLM.

LlamaGuard outputs either "safe" or "unsafe\\n<categories>" — we parse both forms.
Self-hosted via vLLM on a dedicated L40S GPU for sub-100ms classification with
no network dependency and no per-call cost.
"""

from __future__ import annotations

from openai import OpenAI

from ..config import get_settings
from .types import SafetyVerdict


def _client() -> OpenAI:
    s = get_settings()
    return OpenAI(api_key=s.vllm_guard_api_key, base_url=s.vllm_guard_url)


def moderate(text: str, role: str = "user") -> SafetyVerdict:
    """Classify text as safe/unsafe. `role` is 'user' (input) or 'assistant' (output)."""
    if not text.strip():
        return SafetyVerdict(layer="llamaguard", allowed=True)

    s = get_settings()
    completion = _client().chat.completions.create(
        model=s.vllm_guard_model,
        messages=[{"role": role, "content": text}],
        temperature=0,
        max_tokens=128,
    )
    raw = (completion.choices[0].message.content or "").strip()
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return SafetyVerdict(layer="llamaguard", allowed=True, reason="empty response")

    verdict = lines[0].lower()
    if verdict == "safe":
        return SafetyVerdict(layer="llamaguard", allowed=True)

    categories = lines[1].split(",") if len(lines) > 1 else []
    categories = [c.strip() for c in categories if c.strip()]
    return SafetyVerdict(
        layer="llamaguard",
        allowed=False,
        reason=f"LlamaGuard flagged: {', '.join(categories) or 'unsafe'}",
        categories=categories,
    )
