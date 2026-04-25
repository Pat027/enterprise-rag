"""Qdrant-backed vector store for chunked documents."""

from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from ..config import get_settings
from ..ingestion.types import Chunk
from .embedder import embed_documents, embed_query, embedding_dimension


def _client() -> QdrantClient:
    return QdrantClient(url=get_settings().qdrant_url)


def ensure_collection() -> None:
    """Create the collection if it doesn't exist. Idempotent."""
    settings = get_settings()
    client = _client()
    existing = {c.name for c in client.get_collections().collections}
    if settings.qdrant_collection in existing:
        return
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=qm.VectorParams(
            size=embedding_dimension(), distance=qm.Distance.COSINE
        ),
    )


def upsert_chunks(chunks: list[Chunk]) -> int:
    """Embed and upsert chunks into the vector store. Returns count inserted."""
    if not chunks:
        return 0
    ensure_collection()
    settings = get_settings()
    client = _client()

    vectors = embed_documents([c.text for c in chunks])
    points = [
        qm.PointStruct(
            id=_uuid_from_chunk_id(c.id),
            vector=vec,
            payload={
                "chunk_id": c.id,
                "text": c.text,
                "source": c.source,
                "page": c.page,
                "section_path": c.section_path,
                "element_type": c.element_type,
                **c.metadata,
            },
        )
        for c, vec in zip(chunks, vectors, strict=True)
    ]
    client.upsert(collection_name=settings.qdrant_collection, points=points)
    return len(points)


def search(query: str, top_k: int) -> list[dict]:
    """Dense retrieval. Returns raw payloads + score, in similarity-desc order."""
    settings = get_settings()
    client = _client()
    qvec = embed_query(query)
    response = client.query_points(
        collection_name=settings.qdrant_collection, query=qvec, limit=top_k
    )
    return [{"score": p.score, **(p.payload or {})} for p in response.points]


def _uuid_from_chunk_id(chunk_id: str) -> str:
    """Qdrant requires UUID or int point IDs. Derive a deterministic UUID from chunk_id."""
    import uuid

    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))
