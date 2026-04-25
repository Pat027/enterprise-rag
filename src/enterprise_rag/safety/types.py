"""Common types for safety verdicts."""

from __future__ import annotations

from pydantic import BaseModel


class SafetyVerdict(BaseModel):
    layer: str           # which moderator produced this verdict
    allowed: bool
    reason: str = ""
    categories: list[str] = []
