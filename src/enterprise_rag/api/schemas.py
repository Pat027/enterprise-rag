"""Pydantic schemas for API requests and responses."""

from __future__ import annotations

from pydantic import BaseModel


class IngestResponse(BaseModel):
    source: str
    chunks_indexed: int


class QueryRequest(BaseModel):
    query: str


class Citation(BaseModel):
    index: int
    source: str
    page: int | None = None
    section_path: list[str] = []
    rerank_score: float | None = None


class QueryResponse(BaseModel):
    answer: str | None = None
    refusal: str | None = None
    citations: list[Citation] = []
    blocked_by: str | None = None


class HealthResponse(BaseModel):
    status: str
    package_version: str
