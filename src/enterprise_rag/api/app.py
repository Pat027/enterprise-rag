"""FastAPI application — /ingest, /query, /query/stream, /health."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

import structlog
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .. import __version__, graph, ingestion, retrieval
from .auth import warn_if_auth_disabled
from .ratelimit import rate_limit
from .schemas import (
    Citation,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)

log = structlog.get_logger()

app = FastAPI(
    title="Enterprise RAG API",
    description="Advanced document processing and grounded generation.",
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup() -> None:
    warn_if_auth_disabled()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", package_version=__version__)


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile = File(...),  # noqa: B008 — FastAPI idiom
    caller_id: str = Depends(rate_limit),  # noqa: B008 — FastAPI idiom
) -> IngestResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="missing filename")

    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        log.info("ingest_start", source=file.filename, caller=caller_id)
        chunks = ingestion.parse_document(tmp_path)
        # Preserve original filename in chunks (parser used the temp path)
        for c in chunks:
            c.source = file.filename
        count = retrieval.upsert_chunks(chunks)
        log.info("ingest_done", source=file.filename, chunks=count, caller=caller_id)
        return IngestResponse(source=file.filename, chunks_indexed=count)
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/query", response_model=QueryResponse)
def query(
    req: QueryRequest,
    caller_id: str = Depends(rate_limit),
) -> QueryResponse:
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="empty query")

    log.info("query_start", caller=caller_id)
    result = graph.run_query(req.query)
    blocking = result.get("blocking_verdict")

    citations = [
        Citation(
            index=i + 1,
            source=p.get("source", "unknown"),
            page=p.get("page"),
            section_path=p.get("section_path", []),
            rerank_score=p.get("rerank_score"),
        )
        for i, p in enumerate(result.get("passages", []))
    ]

    return QueryResponse(
        answer=result.get("answer") if not blocking else None,
        refusal=result.get("refusal"),
        citations=citations if not blocking else [],
        blocked_by=blocking.layer if blocking else None,
    )


async def _sse_event_stream(query_text: str) -> AsyncIterator[bytes]:
    """Adapt run_query_stream() event dicts to SSE wire format."""
    async for event in graph.run_query_stream(query_text):
        payload = json.dumps(event, ensure_ascii=False)
        yield f"data: {payload}\n\n".encode()


@app.post("/query/stream")
async def query_stream(
    req: QueryRequest,
    caller_id: str = Depends(rate_limit),
) -> StreamingResponse:
    """Stream the RAG pipeline as Server-Sent Events.

    Each event is a JSON object: see ``graph.streaming`` for shapes.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="empty query")

    log.info("query_stream_start", caller=caller_id)
    return StreamingResponse(
        _sse_event_stream(req.query),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )
