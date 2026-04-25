"""Grounded generation with inline citations."""

from .llm import generate
from .prompts import format_context

__all__ = ["format_context", "generate"]
