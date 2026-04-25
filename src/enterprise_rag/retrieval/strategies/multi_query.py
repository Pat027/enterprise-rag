"""Multi-Query / RAG-Fusion.

Generate N paraphrases of the user's query, retrieve passages for each, then
fuse the result lists with Reciprocal Rank Fusion (k=60). Variants help recall
when a single query phrasing misses synonyms or related framings the index
embedded under different surface forms.

References:
- Multi-Query Retriever pattern (LangChain): single-paper popularization
- RAG-Fusion (Adrian H. Raudaschl, 2023): adds RRF on top of multi-query
"""

from __future__ import annotations

from . import _llm
from .base import _base_retrieve, _rerank_and_tag, rrf_fuse

NAME = "multi_query"
_NUM_VARIANTS = 4

_REWRITE_PROMPT = """You are a query rewriter. Produce {n} alternative \
phrasings of the user's question that preserve the same intent but use \
different vocabulary, structure, or angle. Output exactly {n} lines, one \
question per line, no numbering or commentary.

Original question: {query}

Variants:"""


def _generate_variants(query: str, n: int = _NUM_VARIANTS) -> list[str]:
    raw = _llm.rewrite(_REWRITE_PROMPT.format(n=n, query=query), max_tokens=256)
    lines = [ln.strip(" -•\t") for ln in raw.splitlines() if ln.strip()]
    # Keep the original alongside variants so we don't lose recall on the user's
    # exact phrasing if the LLM drifts too far.
    variants = [v for v in lines[:n] if v]
    if query not in variants:
        variants.insert(0, query)
    return variants


def retrieve(query: str) -> list[dict]:
    from ...config import get_settings

    settings = get_settings()
    variants = _generate_variants(query)
    rankings = [_base_retrieve(v, top_k=settings.top_k_retrieve) for v in variants]
    fused = rrf_fuse(rankings, top_k=settings.top_k_retrieve)
    return _rerank_and_tag(query, fused, strategy=NAME)
