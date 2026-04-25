"""Application configuration loaded from environment variables."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── Generation / Constitutional critic — local vLLM (Llama 3.1 70B FP8) ──
    vllm_gen_url: str = "http://vllm-gen:8000/v1"
    vllm_gen_model: str = "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8-dynamic"
    vllm_gen_api_key: str = "EMPTY"  # vLLM doesn't authenticate by default

    # ── LlamaGuard 3 8B served by a separate vLLM instance ──
    vllm_guard_url: str = "http://vllm-guard:8000/v1"
    vllm_guard_model: str = "meta-llama/Llama-Guard-3-8B"
    vllm_guard_api_key: str = "EMPTY"

    # ── OpenAI Moderation: only remaining hosted dependency (free, optional) ──
    openai_api_key: str = ""

    # ── Vector store ──
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "documents"

    # ── Embeddings + reranker (run inside the API container, GPU-accelerated) ──
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    embedder_device: str = "cuda"  # falls back to cpu if CUDA unavailable

    # ── Safety toggles ──
    safety_openai_moderation: bool = True
    safety_llamaguard: bool = True
    safety_constitutional: bool = True

    # ── Retrieval ──
    top_k_retrieve: int = Field(default=20, ge=1, le=100)
    top_k_rerank: int = Field(default=5, ge=1, le=20)
    retrieval_mode: str = "hybrid"  # "dense" | "bm25" | "hybrid"

    # ── API ──
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── Auth & rate limiting ──
    api_keys_csv: str = Field(default="", alias="API_KEYS")
    rate_limit_per_min: int = 60
    rate_limit_burst: int = 10

    def api_keys(self) -> set[str]:
        """Parse comma-separated API keys; empty set means auth disabled."""
        return {k.strip() for k in self.api_keys_csv.split(",") if k.strip()}


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
