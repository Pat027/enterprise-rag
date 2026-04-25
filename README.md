# Enterprise RAG

A production-oriented Retrieval-Augmented Generation platform for grounding LLMs in enterprise data — multimodal ingestion, hybrid retrieval, reranking, optional guardrails, and OpenAI-compatible APIs.

## What's inside

- **Multimodal ingestion** — text, tables, charts, infographics, and audio across many document types
- **Hybrid retrieval** — dense + sparse search with GPU-accelerated indexing, reranking, and pluggable vector databases (Milvus, Elasticsearch)
- **Query processing** — decomposition and dynamic filter expressions
- **Generation** — optional VLM support, document summarization, reflection-based accuracy refinement, and programmable content-safety guardrails
- **Evaluation** — RAGAS-based scripts
- **Deployment** — Docker (with or without hosted NIM endpoints), Kubernetes, NIM Operator, and Python library mode
- **APIs** — OpenAI-compatible, with telemetry and observability built in
- **Reference UI** — multi-turn, multi-session sample frontend

## Installation

### Prerequisites

- Python 3.11+ and [`uv`](https://docs.astral.sh/uv/)
- Docker + Docker Compose (for the recommended deployment path)
- An NVIDIA GPU for self-hosted models, or an NVIDIA API Catalog key for hosted endpoints
- Node.js + `pnpm` (only if you want to run the frontend locally)

### Backend setup

```bash
git clone https://github.com/Pat027/enterprise-rag.git
cd enterprise-rag
uv sync
```

### Deploy with Docker Compose (recommended)

Single-node deployment with self-hosted, on-premises models:

```bash
cd deploy/compose
docker compose --env-file .env up -d
```

For NVIDIA-hosted endpoints instead of self-hosted models, use `nvdev.env`:

```bash
docker compose --env-file nvdev.env up -d
```

Full step-by-step guide: [`docs/deploy-docker-self-hosted.md`](./docs/deploy-docker-self-hosted.md).

### Other deployment modes

- **Kubernetes / Helm** — `deploy/helm/nvidia-blueprint-rag/` (supports MIG GPU slicing)
- **Library mode** — `import nvidia_rag` for custom pipelines

See [`UPSTREAM_README.md`](./UPSTREAM_README.md) and [`docs/readme.md`](./docs/readme.md) for the full deployment matrix, configuration options, and troubleshooting.

### Frontend (optional)

```bash
cd frontend
pnpm install
pnpm run dev
```

### Tests

```bash
uv run pytest tests/unit/         # Unit tests (no network)
uv run pytest tests/integration/  # Integration tests
```

## Credits

Derived from [NVIDIA AI Blueprint: RAG](https://github.com/NVIDIA-AI-Blueprints/rag).

## License

Apache License 2.0 — see [`LICENSE`](./LICENSE).
