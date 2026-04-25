"""Retrieval entry point — dispatches to a configurable RAG strategy.

The orthogonal axes:
  - ``settings.retrieval_mode``  → dense | bm25 | hybrid (vector retrieval algo)
  - ``settings.rag_strategy``    → direct | hyde | multi_query | step_back
                                   (query orchestration on top of the algo)

Each strategy independently performs query rewriting / expansion and then
delegates to the configured retrieval mode. The registry lives in
``strategies/__init__.py``; documentation in ``docs/rag-strategies.md``.
"""

from __future__ import annotations

from ..config import get_settings
from . import strategies


def retrieve(query: str) -> list[dict]:
    settings = get_settings()
    strategy = strategies.get(settings.rag_strategy)
    return strategy(query)
