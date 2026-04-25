"""Layout-aware document parsing via Docling.

Docling produces a structured document tree (sections, paragraphs, tables, figures)
that we flatten into retrievable chunks while preserving structural metadata.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from docling.document_converter import DocumentConverter

from .types import Chunk

_converter: DocumentConverter | None = None


def _get_converter() -> DocumentConverter:
    global _converter
    if _converter is None:
        _converter = DocumentConverter()
    return _converter


def _chunk_id(source: str, idx: int, text: str) -> str:
    digest = hashlib.sha1(f"{source}:{idx}:{text[:64]}".encode()).hexdigest()[:16]
    return f"{Path(source).stem}-{idx:04d}-{digest}"


def parse_document(file_path: str | Path) -> list[Chunk]:
    """Parse a document into structured chunks.

    Supports PDF, DOCX, HTML, images, and more — whatever Docling handles.
    Tables become their own chunks (with markdown-rendered content), figures
    contribute their captions, and prose is split by structural section.
    """
    path = Path(file_path)
    converter = _get_converter()
    result = converter.convert(str(path))
    doc = result.document

    chunks: list[Chunk] = []
    section_stack: list[str] = []
    source = path.name

    for idx, item in enumerate(doc.iterate_items()):
        # Each item is (NodeItem, level) in newer docling versions
        node = item[0] if isinstance(item, tuple) else item
        label = getattr(node, "label", None)
        text = getattr(node, "text", None) or ""

        if label == "section_header":
            level = getattr(node, "level", len(section_stack))
            section_stack = section_stack[:level]
            section_stack.append(text.strip())
            continue

        if label == "table":
            # Render tables as markdown so the LLM can read structure
            try:
                rendered = node.export_to_markdown(doc=doc)
            except Exception:
                rendered = text
            if rendered.strip():
                chunks.append(
                    Chunk(
                        id=_chunk_id(source, idx, rendered),
                        text=rendered,
                        source=source,
                        page=getattr(node, "page_no", None),
                        section_path=list(section_stack),
                        element_type="table",
                    )
                )
            continue

        if label in {"picture", "figure"}:
            caption = getattr(node, "caption_text", None) or text
            if caption and caption.strip():
                chunks.append(
                    Chunk(
                        id=_chunk_id(source, idx, caption),
                        text=f"[Figure] {caption}",
                        source=source,
                        page=getattr(node, "page_no", None),
                        section_path=list(section_stack),
                        element_type="figure_caption",
                    )
                )
            continue

        if not text.strip():
            continue

        chunks.append(
            Chunk(
                id=_chunk_id(source, idx, text),
                text=text,
                source=source,
                page=getattr(node, "page_no", None),
                section_path=list(section_stack),
                element_type="text",
            )
        )

    return _merge_short_chunks(chunks)


def _merge_short_chunks(chunks: list[Chunk], min_chars: int = 200) -> list[Chunk]:
    """Merge consecutive prose chunks under min_chars within the same section.

    Tables and figures are preserved as-is — only prose merges. This keeps
    chunk size meaningful for embedding while preserving structural elements.
    """
    if not chunks:
        return chunks

    merged: list[Chunk] = []
    buffer: Chunk | None = None

    for c in chunks:
        if c.element_type != "text":
            if buffer is not None:
                merged.append(buffer)
                buffer = None
            merged.append(c)
            continue

        if buffer is None:
            buffer = c
            continue

        same_section = buffer.section_path == c.section_path
        if same_section and len(buffer.text) < min_chars:
            buffer = buffer.model_copy(
                update={"text": f"{buffer.text}\n\n{c.text}"}
            )
        else:
            merged.append(buffer)
            buffer = c

    if buffer is not None:
        merged.append(buffer)

    return merged
