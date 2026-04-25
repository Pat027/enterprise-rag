"""Grounded generation via local vLLM (OpenAI-compatible endpoint)."""

from __future__ import annotations

from openai import OpenAI

from ..config import get_settings
from .prompts import SYSTEM_PROMPT, build_user_prompt


def _client() -> OpenAI:
    s = get_settings()
    return OpenAI(api_key=s.vllm_gen_api_key, base_url=s.vllm_gen_url)


def generate(query: str, passages: list[dict]) -> str:
    """Produce a grounded answer with inline citations."""
    s = get_settings()
    user_prompt = build_user_prompt(query, passages)

    completion = _client().chat.completions.create(
        model=s.vllm_gen_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return (completion.choices[0].message.content or "").strip()
