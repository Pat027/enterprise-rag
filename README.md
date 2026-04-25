# Enterprise RAG

An advanced document processing and retrieval-augmented generation system that runs **fully on-premises**: layout-aware ingestion, multi-layer safety, and self-hosted LLM inference on consumer-to-datacenter GPUs.

## Why this exists

Most RAG demos do the easy parts (embed text, vector search, prompt) and skip what actually breaks in production: parsing real documents with tables and scans, blocking unsafe inputs and outputs without latency tax, grounding answers with citations the user can verify, and avoiding lock-in to hosted LLM APIs that leak data and cost money per token.

This system addresses all four — and runs on your own hardware.

## Architecture

```
            ┌─────────────────────────────────────────┐
            │  FastAPI  (/ingest, /query, /health)    │
            └───────┬────────────────────────┬────────┘
                    │                        │
            ┌───────▼─────────┐      ┌───────▼─────────────────────┐
            │  Ingestion      │      │  LangGraph state machine    │
            │  ─ Docling      │      │                             │
            │    layout +     │      │  input_safety               │
            │    tables +     │      │      ▼                      │
            │    OCR +        │      │  retrieve  (Qdrant + BGE-M3)│
            │    figures      │      │      ▼                      │
            │  ─ Hierarchical │      │  rerank    (BGE-Reranker)   │
            │    chunking     │      │      ▼                      │
            └───────┬─────────┘      │  generate  (vLLM Llama 70B) │
                    │                │      ▼                      │
            ┌───────▼─────────┐      │  output_safety              │
            │  Qdrant         │◀─────┤      ▼                      │
            │  vector store   │      │  return                     │
            └─────────────────┘      └─────────────────────────────┘

GPU layout (4× L40S 48GB reference):
  GPU 0+2 → vLLM serving Llama 3.1 70B FP8 (TP=2) for generation + critic
  GPU 1   → API container: BGE-M3 embeddings + BGE-Reranker
  GPU 3   → vLLM serving LlamaGuard 3 8B for safety classification

Safety layers (defense-in-depth, all on by default):
  1. OpenAI Moderation (hosted, free)  — fast, coarse categories
  2. LlamaGuard 3 (local vLLM)         — policy-aware, sub-100ms
  3. Constitutional critic (local vLLM) — Llama 70B as judge against constitution.yaml
```

## Stack

| Layer | Component | Why |
|---|---|---|
| Document parsing | [Docling](https://github.com/DS4SD/docling) | Layout, tables, OCR, figures, formulas — best open-source option |
| Embeddings | BGE-M3 | Multilingual, dense + sparse + multi-vector in one model |
| Vector DB | Qdrant | Purpose-built, hybrid search built in |
| Reranker | BGE-Reranker-v2-m3 | Strong cross-encoder, MIT-licensed |
| LLM serving | vLLM | High-throughput batched inference, OpenAI-compatible API |
| Generation model | Llama 3.1 70B Instruct (FP8) | Strong reasoning, fits 2× L40S with native FP8 |
| Safety classifier | LlamaGuard 3 8B | Open-weights policy classifier from Meta |
| Critic | Llama 3.1 70B (same vLLM, different prompt) | LLM-as-judge — no separate vendor |
| Orchestration | LangGraph | Explicit state machine, easier to reason about than chains |
| API | FastAPI | Standard choice |

## Hardware requirements

**Reference deployment**: 4× NVIDIA L40S 48GB on a single host.

**Will also run on**:
- 2× A100 80GB (consolidate gen + guard on same machine)
- 4× RTX 4090 24GB (use AWQ INT4 quantization instead of FP8 — change `VLLM_GEN_MODEL`)
- 2× H100 80GB (run Llama 3.3 70B BF16 unquantized for max quality)

For lighter setups, swap to Llama 3.1 8B and use a single GPU.

## Quickstart

### Prerequisites
- Linux host with 4× NVIDIA GPUs (L40S reference; see hardware section above)
- NVIDIA driver ≥ 535, CUDA 12.2+
- Docker + Docker Compose with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Hugging Face token with access to gated Llama models

### 1. Get HF access to Llama models
Both `meta-llama/Llama-3.1-70B-Instruct` and `meta-llama/Llama-Guard-3-8B` are gated. Accept the licenses on Hugging Face, then create a token at https://huggingface.co/settings/tokens with read scope.

### 2. Configure
```bash
git clone https://github.com/Pat027/enterprise-rag.git
cd enterprise-rag
cp .env.example .env
# Edit .env: set HF_TOKEN; OPENAI_API_KEY is optional
```

### 3. Launch
```bash
docker compose up --build -d
docker compose logs -f vllm-gen   # follow until "Application startup complete"
```

First boot downloads ~80GB of model weights into `./hf_cache/`. Subsequent starts are fast.

### 4. Ingest a document
```bash
curl -X POST http://localhost:8000/ingest -F "file=@./your-document.pdf"
```

### 5. Query
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What were the key findings in section 3?"}'
```

Response:
```json
{
  "answer": "The key findings were ... [1] ... [2]",
  "citations": [
    {"index": 1, "source": "your-document.pdf", "page": 12, "section_path": ["Section 3"]},
    {"index": 2, "source": "your-document.pdf", "page": 14}
  ],
  "blocked_by": null
}
```

If a safety layer blocks:
```json
{
  "answer": null,
  "refusal": "Your request was blocked by the safety layer (llamaguard): ...",
  "blocked_by": "llamaguard"
}
```

## Local dev (without Docker)

```bash
uv sync
docker run -d -p 6333:6333 qdrant/qdrant:v1.12.4
# Run vLLM locally — see https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
# Or point VLLM_GEN_URL/VLLM_GUARD_URL at any OpenAI-compatible endpoint
cp .env.example .env  # fill in
uv run enterprise-rag
```

## Configuration

All knobs live in `.env`. Notable:

| Var | Default | Notes |
|---|---|---|
| `VLLM_GEN_MODEL` | `neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8-dynamic` | Any vLLM-supported model |
| `VLLM_GUARD_MODEL` | `meta-llama/Llama-Guard-3-8B` | Safety classifier |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Any sentence-transformers model |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Any cross-encoder |
| `EMBEDDER_DEVICE` | `cuda` | `cuda`, `cuda:0`, or `cpu` |
| `TOP_K_RETRIEVE` | `20` | Dense recall depth |
| `TOP_K_RERANK` | `5` | Final passages sent to LLM |
| `SAFETY_*` | `true` | Toggle individual safety layers |

Constitutional policy is editable at `src/enterprise_rag/safety/constitution.yaml`.

## Project layout

```
src/enterprise_rag/
├── ingestion/      Docling parsing → structured chunks
├── retrieval/      Qdrant + BGE-M3 + BGE reranker
├── safety/         3-layer moderation pipeline
├── generation/     vLLM OpenAI-compatible client + prompts
├── graph/          LangGraph state machine
├── api/            FastAPI app
├── config.py       Pydantic settings
└── cli.py          uvicorn launcher
```

## Roadmap

- [ ] Frontend (Next.js + shadcn/ui)
- [ ] Hybrid search (BM25 + dense)
- [ ] Streaming responses
- [ ] OpenTelemetry → Grafana dashboard
- [ ] RAGAS evaluation harness
- [ ] Multi-collection routing

## License

MIT
