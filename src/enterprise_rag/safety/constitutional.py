"""Constitutional critic: LLM-as-judge against a written policy.

Uses the same local vLLM as generation (Llama 3.1 70B FP8) with a different
system prompt — no separate model, no Anthropic dependency. The constitution
is loaded from constitution.yaml and rendered into a structured judging prompt.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import yaml
from openai import OpenAI

from ..config import get_settings
from .types import SafetyVerdict

_CONSTITUTION_PATH = Path(__file__).parent / "constitution.yaml"


@lru_cache(maxsize=1)
def _principles() -> list[dict]:
    return yaml.safe_load(_CONSTITUTION_PATH.read_text())["principles"]


def _client() -> OpenAI:
    s = get_settings()
    return OpenAI(api_key=s.vllm_gen_api_key, base_url=s.vllm_gen_url)


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
    s = get_settings()
    principles_text = "\n".join(
        f"- {p['id']}: {p['rule'].strip()}" for p in _principles()
    )
    prompt = _JUDGE_PROMPT.format(
        principles=principles_text, query=query, context=context, response=response
    )

    # Note: response_format={"type":"json_object"} (xgrammar guided decoding) is
    # *not* used here — vLLM 0.6.6 has an incompatibility between guided decoding
    # and speculative decoding. Our parse below is tolerant: it locates the first
    # `{` and last `}` and decodes the slice. The system instruction in the prompt
    # asks for strict JSON, which Llama 3.1 8B reliably produces.
    completion = _client().chat.completions.create(
        model=s.vllm_gen_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=512,
    )
    raw = (completion.choices[0].message.content or "").strip()

    try:
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
