"""Step-back prompting.

Zheng et al. 2023 (https://arxiv.org/abs/2310.06117). For complex or
narrowly-framed questions, generate a more *abstract* version (the
"step-back" question) and retrieve passages for both the original and the
step-back. This pulls in higher-level context (definitions, principles,
overviews) that the literal query might skip past.

The fused candidate set is reranked against the *original* question so the
final passages still answer what the user actually asked.
"""

from __future__ import annotations

from . import _llm
from .base import _base_retrieve, _rerank_and_tag, rrf_fuse

NAME = "step_back"

_STEP_BACK_PROMPT = """Rewrite the question below as a more abstract, general \
question that would help retrieve background context. Output ONE question \
only, no commentary.

Examples:
- Specific: "Was the 2008 financial crisis caused by subprime mortgages?"
  Step-back: "What causes financial crises?"
- Specific: "How do I configure max-model-len in vLLM 0.6.6?"
  Step-back: "How does vLLM allocate KV cache?"

Specific: {query}
Step-back:"""


def retrieve(query: str) -> list[dict]:
    from ...config import get_settings

    settings = get_settings()
    abstract = _llm.rewrite(
        _STEP_BACK_PROMPT.format(query=query), max_tokens=128, temperature=0.3
    )
    abstract = abstract.splitlines()[0].strip() if abstract else query

    direct_hits = _base_retrieve(query, top_k=settings.top_k_retrieve)
    abstract_hits = (
        _base_retrieve(abstract, top_k=settings.top_k_retrieve)
        if abstract and abstract.lower() != query.lower()
        else []
    )
    fused = rrf_fuse([direct_hits, abstract_hits], top_k=settings.top_k_retrieve)
    return _rerank_and_tag(query, fused, strategy=NAME)
