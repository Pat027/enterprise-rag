"""Unit tests for ingestion.parser pure helpers."""

from __future__ import annotations

from enterprise_rag.ingestion.parser import _chunk_id, _merge_short_chunks
from enterprise_rag.ingestion.types import Chunk


def _text_chunk(idx: int, text: str, section: list[str] | None = None) -> Chunk:
    return Chunk(
        id=f"x-{idx:04d}",
        text=text,
        source="doc.pdf",
        section_path=section or ["Intro"],
        element_type="text",
    )


def test_chunk_id_is_deterministic():
    a = _chunk_id("doc.pdf", 3, "hello world")
    b = _chunk_id("doc.pdf", 3, "hello world")
    assert a == b
    assert a.startswith("doc-0003-")
    # Different idx changes the id
    assert _chunk_id("doc.pdf", 4, "hello world") != a
    # Different text changes the id
    assert _chunk_id("doc.pdf", 3, "different") != a


def test_merge_short_chunks_within_section():
    chunks = [
        _text_chunk(0, "short one", ["Intro"]),
        _text_chunk(1, "short two", ["Intro"]),
        _text_chunk(2, "short three", ["Intro"]),
    ]
    merged = _merge_short_chunks(chunks, min_chars=200)
    assert len(merged) == 1
    assert "short one" in merged[0].text
    assert "short two" in merged[0].text
    assert "short three" in merged[0].text


def test_merge_does_not_cross_sections():
    chunks = [
        _text_chunk(0, "alpha", ["Intro"]),
        _text_chunk(1, "beta", ["Methods"]),
    ]
    merged = _merge_short_chunks(chunks, min_chars=200)
    assert len(merged) == 2
    assert merged[0].section_path == ["Intro"]
    assert merged[1].section_path == ["Methods"]


def test_merge_stops_at_non_text_chunks():
    table = Chunk(
        id="t-1",
        text="| col |\n| --- |",
        source="doc.pdf",
        section_path=["Intro"],
        element_type="table",
    )
    chunks = [
        _text_chunk(0, "before", ["Intro"]),
        table,
        _text_chunk(2, "after", ["Intro"]),
    ]
    merged = _merge_short_chunks(chunks, min_chars=200)
    assert len(merged) == 3
    assert merged[0].element_type == "text"
    assert merged[1].element_type == "table"
    assert merged[2].element_type == "text"


def test_merge_empty_input_returns_empty():
    assert _merge_short_chunks([]) == []


def test_merge_keeps_long_chunks_separate():
    long_text = "x" * 300
    chunks = [
        _text_chunk(0, long_text, ["Intro"]),
        _text_chunk(1, "short follow", ["Intro"]),
    ]
    merged = _merge_short_chunks(chunks, min_chars=200)
    # First buffer is already >= min_chars, so the next one starts a new buffer.
    assert len(merged) == 2
