"""Direct retrieval — the user's query, as-is. Baseline 'naive RAG'."""

from __future__ import annotations

from .base import _base_retrieve, _rerank_and_tag

NAME = "direct"


def retrieve(query: str) -> list[dict]:
    """Use the configured retrieval mode with the unmodified query, then rerank."""
    candidates = _base_retrieve(query)
    return _rerank_and_tag(query, candidates, strategy=NAME)
