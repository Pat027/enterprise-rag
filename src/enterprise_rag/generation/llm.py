"""Grounded generation via local vLLM (OpenAI-compatible endpoint)."""

from __future__ import annotations

from collections.abc import AsyncIterator

from openai import AsyncOpenAI, OpenAI

from ..config import get_settings
from .prompts import SYSTEM_PROMPT, build_user_prompt


def _client() -> OpenAI:
    s = get_settings()
    return OpenAI(api_key=s.vllm_gen_api_key, base_url=s.vllm_gen_url)


def _async_client() -> AsyncOpenAI:
    s = get_settings()
    return AsyncOpenAI(api_key=s.vllm_gen_api_key, base_url=s.vllm_gen_url)


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


async def generate_stream(query: str, passages: list[dict]) -> AsyncIterator[str]:
    """Stream a grounded answer token-by-token from vLLM (async)."""
    s = get_settings()
    user_prompt = build_user_prompt(query, passages)

    stream = await _async_client().chat.completions.create(
        model=s.vllm_gen_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        stream=True,
    )
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        text = getattr(delta, "content", None)
        if text:
            yield text
