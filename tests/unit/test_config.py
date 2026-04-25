"""Unit tests for config.Settings."""

from __future__ import annotations

from enterprise_rag.config import Settings


def test_settings_defaults(monkeypatch):
    # Disable .env loading and clear any host env vars that map onto Settings fields.
    for var in [
        "OPENAI_API_KEY",
        "QDRANT_URL",
        "QDRANT_COLLECTION",
        "EMBEDDING_MODEL",
        "API_PORT",
        "API_HOST",
        "SAFETY_OPENAI_MODERATION",
        "SAFETY_LLAMAGUARD",
        "SAFETY_CONSTITUTIONAL",
        "TOP_K_RETRIEVE",
        "TOP_K_RERANK",
    ]:
        monkeypatch.delenv(var, raising=False)
    s = Settings(_env_file=None)
    assert s.qdrant_collection == "documents"
    assert s.qdrant_url == "http://qdrant:6333"
    assert s.embedding_model == "BAAI/bge-m3"
    assert s.safety_openai_moderation is True
    assert s.safety_llamaguard is True
    assert s.safety_constitutional is True
    assert s.top_k_retrieve == 20
    assert s.top_k_rerank == 5
    assert s.api_port == 8000


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("QDRANT_COLLECTION", "custom_coll")
    monkeypatch.setenv("API_PORT", "9001")
    monkeypatch.setenv("SAFETY_LLAMAGUARD", "false")
    s = Settings(_env_file=None)
    assert s.qdrant_collection == "custom_coll"
    assert s.api_port == 9001
    assert s.safety_llamaguard is False


def test_settings_validates_top_k_bounds():
    # Below ge=1 should fail.
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings(_env_file=None, top_k_retrieve=0)
