"""Data types for the ingestion pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A single retrievable unit produced by the ingestion pipeline."""

    id: str
    text: str
    source: str
    page: int | None = None
    section_path: list[str] = Field(default_factory=list)
    element_type: str = "text"  # text | table | figure_caption | code
    metadata: dict[str, object] = Field(default_factory=dict)
