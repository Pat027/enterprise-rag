"""Two-stage retrieval: dense recall then cross-encoder rerank."""

from __future__ import annotations

from ..config import get_settings
from . import reranker, store


def retrieve(query: str) -> list[dict]:
    """Run dense retrieval + reranking. Returns top_k_rerank results with scores."""
    settings = get_settings()
    candidates = store.search(query, top_k=settings.top_k_retrieve)
    if not candidates:
        return []

    texts = [c["text"] for c in candidates]
    ranked = reranker.rerank(query, texts, top_k=settings.top_k_rerank)

    out = []
    for original_idx, rerank_score in ranked:
        item = dict(candidates[original_idx])
        item["rerank_score"] = float(rerank_score)
        out.append(item)
    return out
