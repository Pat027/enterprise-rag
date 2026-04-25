"""OpenAI Moderation: free, ~50ms, coarse-grained category flags."""

from __future__ import annotations

from openai import OpenAI

from ..config import get_settings
from .types import SafetyVerdict


def _client() -> OpenAI:
    return OpenAI(api_key=get_settings().openai_api_key)


def moderate(text: str) -> SafetyVerdict:
    """Run text through OpenAI Moderation. Returns a verdict."""
    if not text.strip():
        return SafetyVerdict(layer="openai_moderation", allowed=True)

    resp = _client().moderations.create(model="omni-moderation-latest", input=text)
    result = resp.results[0]
    flagged = result.flagged
    categories = [k for k, v in result.categories.model_dump().items() if v]

    return SafetyVerdict(
        layer="openai_moderation",
        allowed=not flagged,
        reason=f"flagged categories: {', '.join(categories)}" if flagged else "",
        categories=categories,
    )
