"""HyDE — Hypothetical Document Embeddings.

Gao et al. 2022 (https://arxiv.org/abs/2212.10496). Instead of embedding the
user's *question*, generate a plausible *answer* with the LLM, embed that, and
retrieve nearest passages. The intuition: a hypothetical answer lives in the
same semantic neighborhood as real answers, so it bridges the question/answer
representation gap that dense retrievers struggle with.

We retain the original query for the cross-encoder rerank step — the
hypothetical document only steers the *recall* stage.
"""

from __future__ import annotations

from . import _llm
from .base import _base_retrieve, _rerank_and_tag

NAME = "hyde"

_HYDE_PROMPT = """Write a short, factual paragraph that directly answers the \
question below as if it were an excerpt from a relevant document. Do not \
include caveats or meta-commentary. 3-5 sentences.

Question: {query}

Answer:"""


def retrieve(query: str) -> list[dict]:
    hypothetical = _llm.rewrite(
        _HYDE_PROMPT.format(query=query), max_tokens=200, temperature=0.5
    )
    # Retrieve using the hypothetical answer; rerank with the original question.
    candidates = _base_retrieve(hypothetical or query)
    return _rerank_and_tag(query, candidates, strategy=NAME)
