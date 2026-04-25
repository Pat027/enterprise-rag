"""Unit tests for prompts.format_context and prompts.build_user_prompt."""

from __future__ import annotations

from enterprise_rag.generation.prompts import build_user_prompt, format_context


def test_format_context_empty_passages():
    assert format_context([]) == "(no context retrieved)"


def test_format_context_numbers_passages_starting_at_one():
    passages = [
        {"source": "a.pdf", "page": 1, "text": "alpha"},
        {"source": "b.pdf", "page": 2, "text": "beta"},
        {"source": "c.pdf", "text": "gamma"},
    ]
    rendered = format_context(passages)
    assert "[1]" in rendered
    assert "[2]" in rendered
    assert "[3]" in rendered
    assert "alpha" in rendered and "beta" in rendered and "gamma" in rendered
    # No-page passage shouldn't render ", p."
    assert "(c.pdf)" in rendered


def test_format_context_renders_section_path():
    passages = [
        {
            "source": "doc.pdf",
            "page": 4,
            "section_path": ["Methods", "Training"],
            "text": "we trained for 3 epochs",
        }
    ]
    rendered = format_context(passages)
    assert "Methods > Training" in rendered
    assert "p.4" in rendered


def test_build_user_prompt_includes_query_and_context():
    passages = [{"source": "x.pdf", "page": 1, "text": "info"}]
    prompt = build_user_prompt("What is X?", passages)
    assert "What is X?" in prompt
    assert "Context:" in prompt
    assert "[1]" in prompt
    assert "inline [n] citations" in prompt


def test_build_user_prompt_with_empty_passages():
    prompt = build_user_prompt("Why?", [])
    assert "(no context retrieved)" in prompt
    assert "Why?" in prompt
