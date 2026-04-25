"""Shared state passed between LangGraph nodes."""

from __future__ import annotations

from typing import TypedDict

from ..safety.types import SafetyVerdict


class RAGState(TypedDict, total=False):
    query: str
    passages: list[dict]
    context_text: str
    answer: str
    refusal: str
    blocking_verdict: SafetyVerdict | None
