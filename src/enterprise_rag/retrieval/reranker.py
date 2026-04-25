"""BGE-Reranker-v2-m3: cross-encoder rerank, GPU-accelerated."""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import CrossEncoder

from ..config import get_settings
from .embedder import _resolve_device


@lru_cache(maxsize=1)
def _model() -> CrossEncoder:
    s = get_settings()
    return CrossEncoder(s.reranker_model, device=_resolve_device(s.embedder_device))


def rerank(query: str, documents: list[str], top_k: int) -> list[tuple[int, float]]:
    """Score (query, document) pairs and return the top_k indices with scores.

    Returns a list of (original_index, score), sorted by score descending.
    """
    if not documents:
        return []
    pairs = [(query, d) for d in documents]
    scores = _model().predict(pairs, show_progress_bar=False)
    indexed = list(enumerate(scores.tolist()))
    indexed.sort(key=lambda x: x[1], reverse=True)
    return indexed[:top_k]
