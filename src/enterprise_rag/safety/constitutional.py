"""Constitutional critic: LLM-as-judge against a written policy.

This is *not* a hosted Anthropic moderation API — Anthropic doesn't ship one.
It's an implementation of the constitutional-AI critique pattern: ask Claude to
score a response against explicit principles loaded from constitution.yaml.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import yaml
from anthropic import Anthropic

from ..config import get_settings
from .types import SafetyVerdict

_CONSTITUTION_PATH = Path(__file__).parent / "constitution.yaml"


@lru_cache(maxsize=1)
def _principles() -> list[dict]:
    return yaml.safe_load(_CONSTITUTION_PATH.read_text())["principles"]


def _client() -> Anthropic:
    return Anthropic(api_key=get_settings().anthropic_api_key)


_JUDGE_PROMPT = """You are a safety auditor. Evaluate the assistant's response below \
against each principle. Reply ONLY with strict JSON of this shape:

{{"verdicts":[{{"id":"...","passes":true|false,"reason":"..."}},...]}}

Principles:
{principles}

User query:
{query}

Retrieved context (may be empty):
{context}

Assistant's response:
{response}
"""


def critique(query: str, context: str, response: str) -> SafetyVerdict:
    """Audit `response` against the constitution. Block on any failed principle."""
    settings = get_settings()
    principles_text = "\n".join(
        f"- {p['id']}: {p['rule'].strip()}" for p in _principles()
    )
    prompt = _JUDGE_PROMPT.format(
        principles=principles_text, query=query, context=context, response=response
    )

    msg = _client().messages.create(
        model=settings.anthropic_critic_model,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = msg.content[0].text if msg.content else ""

    try:
        # Tolerant parse — find first { and last }
        start, end = raw.find("{"), raw.rfind("}")
        data = json.loads(raw[start : end + 1])
        verdicts = data.get("verdicts", [])
    except (json.JSONDecodeError, ValueError):
        return SafetyVerdict(
            layer="constitutional",
            allowed=True,
            reason="critic returned unparseable output; failing open",
        )

    failed = [v for v in verdicts if not v.get("passes", True)]
    if not failed:
        return SafetyVerdict(layer="constitutional", allowed=True)

    return SafetyVerdict(
        layer="constitutional",
        allowed=False,
        reason="; ".join(
            f"{v.get('id', '?')}: {v.get('reason', '')}" for v in failed
        ),
        categories=[v.get("id", "") for v in failed],
    )
