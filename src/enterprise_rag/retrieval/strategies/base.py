"""Base interface and helpers for query-time RAG strategies.

Strategies orchestrate *how* the user's query is transformed and how retrieval
is invoked. The retrieval *mode* (dense | bm25 | hybrid) is orthogonal: any
strategy delegates the actual passage fetch to ``_base_retrieve()``, which
respects ``settings.retrieval_mode``.

Pattern: each strategy is a single function ``retrieve(query: str) -> list[dict]``
returning the same passage shape as ``store.search()``: dicts with at least
``text``, ``source``, ``page``, ``section_path``, ``rerank_score``.
"""

from __future__ import annotations

from collections.abc import Callable

from ...config import get_settings
from .. import bm25, reranker, store

_RRF_K = 60


def _base_retrieve(query: str, top_k: int | None = None) -> list[dict]:
    """Mode-aware retrieval (dense | bm25 | hybrid). No reranking applied here."""
    settings = get_settings()
    mode = (settings.retrieval_mode or "hybrid").lower()
    if mode not in {"dense", "bm25", "hybrid"}:
        mode = "hybrid"
    k = top_k or settings.top_k_retrieve

    if mode == "dense":
        return store.search(query, top_k=k)
    if mode == "bm25":
        return bm25.search(query, top_k=k)

    # hybrid → RRF fuse
    dense_hits = store.search(query, top_k=k)
    sparse_hits = bm25.search(query, top_k=k)
    return rrf_fuse([dense_hits, sparse_hits], top_k=k)


def rrf_fuse(rankings: list[list[dict]], top_k: int) -> list[dict]:
    """Reciprocal Rank Fusion across N rankings. Dedup by ``chunk_id``."""
    fused: dict[str, dict] = {}
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking):
            cid = hit.get("chunk_id")
            if cid is None:
                continue
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank + 1)
            fused.setdefault(cid, dict(hit))
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [{**fused[cid], "score": float(s)} for cid, s in ordered]


def _rerank_and_tag(query: str, candidates: list[dict], strategy: str) -> list[dict]:
    """Apply cross-encoder rerank and tag the result with the strategy name."""
    if not candidates:
        return []
    settings = get_settings()
    texts = [c["text"] for c in candidates]
    ranked = reranker.rerank(query, texts, top_k=settings.top_k_rerank)
    out = []
    for original_idx, score in ranked:
        item = dict(candidates[original_idx])
        item["rerank_score"] = float(score)
        item["retrieval_mode"] = settings.retrieval_mode
        item["rag_strategy"] = strategy
        out.append(item)
    return out


# Strategies populate this from their own modules; see strategies/__init__.py
StrategyFn = Callable[[str], list[dict]]
