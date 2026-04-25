"""BGE-M3 dense embeddings via sentence-transformers."""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from ..config import get_settings


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    settings = get_settings()
    return SentenceTransformer(settings.embedding_model)


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
