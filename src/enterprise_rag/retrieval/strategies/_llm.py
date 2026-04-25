"""Tiny synchronous helper for query-rewriting LLM calls.

Strategies use this to generate hypothetical answers, query variants, etc.
We deliberately keep it separate from ``generation.llm`` so retrieval
strategies don't pull in citation-formatting logic intended for the final
generation step.
"""

from __future__ import annotations

from openai import OpenAI

from ...config import get_settings


def _client() -> OpenAI:
    s = get_settings()
    return OpenAI(api_key=s.vllm_gen_api_key, base_url=s.vllm_gen_url)


def rewrite(prompt: str, max_tokens: int = 256, temperature: float = 0.5) -> str:
    """Single-shot query rewrite — returns stripped string content."""
    s = get_settings()
    completion = _client().chat.completions.create(
        model=s.vllm_gen_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (completion.choices[0].message.content or "").strip()
