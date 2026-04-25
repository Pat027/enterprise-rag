"""Prompt templates for grounded generation."""

from __future__ import annotations

SYSTEM_PROMPT = """You are an expert assistant that answers questions strictly from \
the provided context. Cite sources inline using [n] notation matching the numbered \
context entries. If the context is insufficient, say so explicitly rather than \
guessing. Be precise; do not invent facts that are not supported by the context."""


def format_context(passages: list[dict]) -> str:
    """Render retrieved passages as a numbered context block."""
    if not passages:
        return "(no context retrieved)"
    blocks = []
    for i, p in enumerate(passages, start=1):
        source = p.get("source", "unknown")
        page = p.get("page")
        page_str = f", p.{page}" if page is not None else ""
        section = " > ".join(p.get("section_path", []) or [])
        section_str = f" — {section}" if section else ""
        blocks.append(
            f"[{i}] ({source}{page_str}{section_str})\n{p.get('text', '')}"
        )
    return "\n\n".join(blocks)


def build_user_prompt(query: str, passages: list[dict]) -> str:
    return f"""Context:
{format_context(passages)}

Question: {query}

Answer with inline [n] citations:"""
