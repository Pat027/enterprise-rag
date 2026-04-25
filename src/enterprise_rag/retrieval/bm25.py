"""BM25 sparse retrieval, persisted to disk alongside the Qdrant store."""

from __future__ import annotations

import pickle
import re
import threading
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from ..ingestion.types import Chunk

_INDEX_PATH = Path("./qdrant_storage/bm25.pkl")
_TOKEN_RE = re.compile(r"\w+")
_lock = threading.Lock()

# In-memory cache: (BM25Okapi instance, list[payload dict], list[tokenized doc])
_state: dict[str, Any] | None = None


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _payload_from_chunk(chunk: Chunk) -> dict:
    return {
        "chunk_id": chunk.id,
        "text": chunk.text,
        "source": chunk.source,
        "page": chunk.page,
        "section_path": chunk.section_path,
        "element_type": chunk.element_type,
        **chunk.metadata,
    }


def _load() -> dict[str, Any] | None:
    global _state
    if _state is not None:
        return _state
    if not _INDEX_PATH.exists():
        return None
    with _INDEX_PATH.open("rb") as f:
        _state = pickle.load(f)
    return _state


def _save(state: dict[str, Any]) -> None:
    _INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _INDEX_PATH.with_suffix(".pkl.tmp")
    with tmp.open("wb") as f:
        pickle.dump(state, f)
    tmp.replace(_INDEX_PATH)


def add_chunks(chunks: list[Chunk]) -> None:
    """Append chunks to the BM25 index (rebuilds from full corpus; small enough)."""
    if not chunks:
        return
    global _state
    with _lock:
        existing = _load() or {"payloads": [], "tokens": []}
        existing_ids = {p["chunk_id"] for p in existing["payloads"]}

        payloads = list(existing["payloads"])
        tokens = list(existing["tokens"])
        for c in chunks:
            if c.id in existing_ids:
                # Update in place
                idx = next(i for i, p in enumerate(payloads) if p["chunk_id"] == c.id)
                payloads[idx] = _payload_from_chunk(c)
                tokens[idx] = _tokenize(c.text)
            else:
                payloads.append(_payload_from_chunk(c))
                tokens.append(_tokenize(c.text))
                existing_ids.add(c.id)

        bm25 = BM25Okapi(tokens) if tokens else None
        _state = {"bm25": bm25, "payloads": payloads, "tokens": tokens}
        _save(_state)


def search(query: str, top_k: int) -> list[dict]:
    """BM25 retrieval. Returns dicts mirroring store.search shape."""
    state = _load()
    if not state or not state.get("bm25") or not state.get("payloads"):
        return []
    bm25: BM25Okapi = state["bm25"]
    payloads: list[dict] = state["payloads"]
    q_tokens = _tokenize(query)
    if not q_tokens:
        return []
    scores = bm25.get_scores(q_tokens)
    # Sort by score desc and take top_k
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [{"score": float(scores[i]), **payloads[i]} for i in order]
