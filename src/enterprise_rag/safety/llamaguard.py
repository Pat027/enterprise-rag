"""LlamaGuard 3 8B served by local vLLM.

LlamaGuard outputs either "safe" or "unsafe\\n<categories>" — we parse both forms.
LlamaGuard 3's chat template requires alternating user/assistant turns, so:
  - Input check: send [{"role":"user", "content":<text>}]
  - Output check: send [{"role":"user", "content":<original_query>},
                        {"role":"assistant", "content":<response>}]
The classifier judges the *last* turn against the rest as context.
"""

from __future__ import annotations

from openai import OpenAI

from ..config import get_settings
from .types import SafetyVerdict


def _client() -> OpenAI:
    s = get_settings()
    return OpenAI(api_key=s.vllm_guard_api_key, base_url=s.vllm_guard_url)


def _classify(messages: list[dict]) -> SafetyVerdict:
    s = get_settings()
    completion = _client().chat.completions.create(
        model=s.vllm_guard_model,
        messages=messages,
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


def check_user_input(text: str) -> SafetyVerdict:
    """Classify a user prompt."""
    if not text.strip():
        return SafetyVerdict(layer="llamaguard", allowed=True)
    return _classify([{"role": "user", "content": text}])


def check_assistant_output(user_query: str, assistant_response: str) -> SafetyVerdict:
    """Classify an assistant response in the context of the user query."""
    if not assistant_response.strip():
        return SafetyVerdict(layer="llamaguard", allowed=True)
    return _classify(
        [
            {"role": "user", "content": user_query or "(empty)"},
            {"role": "assistant", "content": assistant_response},
        ]
    )
