"""Two-stage retrieval: dense and/or BM25 recall, RRF fusion, then cross-encoder rerank."""

from __future__ import annotations

from ..config import get_settings
from . import bm25, reranker, store

_RRF_K = 60


def _rrf_fuse(
    dense_hits: list[dict], sparse_hits: list[dict], top_k: int
) -> list[dict]:
    """Reciprocal Rank Fusion. Dedup by chunk_id; preserve richest payload."""
    fused: dict[str, dict] = {}
    scores: dict[str, float] = {}

    for rank, hit in enumerate(dense_hits):
        cid = hit.get("chunk_id")
        if cid is None:
            continue
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank + 1)
        fused.setdefault(cid, dict(hit))

    for rank, hit in enumerate(sparse_hits):
        cid = hit.get("chunk_id")
        if cid is None:
            continue
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank + 1)
        fused.setdefault(cid, dict(hit))

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    out = []
    for cid, rrf_score in ordered:
        item = dict(fused[cid])
        item["score"] = float(rrf_score)
        out.append(item)
    return out


def retrieve(query: str) -> list[dict]:
    """Run retrieval (dense | bm25 | hybrid) + reranking. Returns top_k_rerank results."""
    settings = get_settings()
    mode = (settings.retrieval_mode or "hybrid").lower()
    if mode not in {"dense", "bm25", "hybrid"}:
        mode = "hybrid"

    if mode == "dense":
        candidates = store.search(query, top_k=settings.top_k_retrieve)
    elif mode == "bm25":
        candidates = bm25.search(query, top_k=settings.top_k_retrieve)
    else:  # hybrid
        dense_hits = store.search(query, top_k=settings.top_k_retrieve)
        sparse_hits = bm25.search(query, top_k=settings.top_k_retrieve)
        candidates = _rrf_fuse(dense_hits, sparse_hits, top_k=settings.top_k_retrieve)

    if not candidates:
        return []

    texts = [c["text"] for c in candidates]
    ranked = reranker.rerank(query, texts, top_k=settings.top_k_rerank)

    out = []
    for original_idx, rerank_score in ranked:
        item = dict(candidates[original_idx])
        item["rerank_score"] = float(rerank_score)
        item["retrieval_mode"] = mode
        out.append(item)
    return out
