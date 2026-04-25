"""BGE-M3 dense embeddings via sentence-transformers, GPU-accelerated."""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from ..config import get_settings


def _resolve_device(requested: str) -> str:
    """Honor settings.embedder_device but fall back to cpu if CUDA is unavailable."""
    if requested == "cpu":
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return requested  # cuda or cuda:N
    except ImportError:
        pass
    return "cpu"


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    s = get_settings()
    return SentenceTransformer(s.embedding_model, device=_resolve_device(s.embedder_device))


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed a batch of document texts. Returns a list of dense vectors."""
    if not texts:
        return []
    vectors = _model().encode(
        texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
    )
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a single query."""
    vector = _model().encode(text, normalize_embeddings=True, show_progress_bar=False)
    return vector.tolist()


def embedding_dimension() -> int:
    return _model().get_sentence_embedding_dimension()
