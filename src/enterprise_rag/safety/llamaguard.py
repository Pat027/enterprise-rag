"""LlamaGuard 3 8B via Groq: policy-aware safety classification.

LlamaGuard outputs either "safe" or "unsafe\\n<categories>" — we parse both forms.
Groq runs LlamaGuard at sub-100ms latency for fractions of a cent per call.
"""

from __future__ import annotations

from groq import Groq

from ..config import get_settings
from .types import SafetyVerdict


def _client() -> Groq:
    return Groq(api_key=get_settings().groq_api_key)


def moderate(text: str, role: str = "user") -> SafetyVerdict:
    """Classify text as safe/unsafe. `role` is 'user' (input) or 'assistant' (output)."""
    if not text.strip():
        return SafetyVerdict(layer="llamaguard", allowed=True)

    settings = get_settings()
    completion = _client().chat.completions.create(
        model=settings.groq_llamaguard_model,
        messages=[{"role": role, "content": text}],
        temperature=0,
        max_tokens=128,
    )
    raw = (completion.choices[0].message.content or "").strip()
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return SafetyVerdict(
            layer="llamaguard", allowed=True, reason="empty response"
        )

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
