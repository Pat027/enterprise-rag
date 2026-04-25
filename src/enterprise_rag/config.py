"""Application configuration loaded from environment variables."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openrouter_api_key: str = ""
    openrouter_model: str = "meta-llama/llama-3.3-70b-instruct"

    openai_api_key: str = ""
    groq_api_key: str = ""
    groq_llamaguard_model: str = "llama-guard-3-8b"

    anthropic_api_key: str = ""
    anthropic_critic_model: str = "claude-haiku-4-5-20251001"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "documents"

    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    safety_openai_moderation: bool = True
    safety_llamaguard: bool = True
    safety_constitutional: bool = True

    top_k_retrieve: int = Field(default=20, ge=1, le=100)
    top_k_rerank: int = Field(default=5, ge=1, le=20)

    api_host: str = "0.0.0.0"
    api_port: int = 8000


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
