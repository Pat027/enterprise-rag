"""Grounded generation via OpenRouter (OpenAI-compatible API gateway)."""

from __future__ import annotations

from openai import OpenAI

from ..config import get_settings
from .prompts import SYSTEM_PROMPT, build_user_prompt


def _client() -> OpenAI:
    return OpenAI(
        api_key=get_settings().openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
    )


def generate(query: str, passages: list[dict]) -> str:
    """Produce a grounded answer with inline citations."""
    settings = get_settings()
    user_prompt = build_user_prompt(query, passages)

    completion = _client().chat.completions.create(
        model=settings.openrouter_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return (completion.choices[0].message.content or "").strip()
