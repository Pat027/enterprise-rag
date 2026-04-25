# Enterprise RAG

An advanced document processing and retrieval-augmented generation system that runs **fully on-premises**: layout-aware ingestion, multi-layer safety, and self-hosted LLM inference on consumer-to-datacenter GPUs.

## Performance

Two rounds of vLLM optimization on a single L40S GPU, measured end-to-end through the full RAG pipeline (input safety → retrieve → rerank → generate → output safety → constitutional critic).

| Metric | Baseline | Round 1 (FP8 + KV + prefix cache) | Round 2 (+ speculative decoding) | Total Δ |
|---|---:|---:|---:|---:|
| Mean latency (sequential) | 9.27 s | 7.42 s | **4.86 s** | **−48 %** |
| **p50 latency** | 10.79 s | 7.78 s | **4.81 s** | **−55 %** |
| p95 latency | 11.66 s | 8.80 s | **5.86 s** | **−50 %** |
| Throughput (sequential) | 0.055 qps | 0.101 qps | **0.175 qps** | **+218 %** |
| Throughput (concurrent, 4 inflight) | — | 0.200 qps | **0.349 qps** | *+75 % vs round 1* |
| Generation-model weights memory | 14.99 GB | 8.49 GB | 8.49 GB | **−43 %** |

Workload: HR-interview Q&A pipeline against an indexed PDF (`benchmarks/bench.py`).
Per-flag rationale + trade-offs: **[`docs/vllm-optimization.md`](./docs/vllm-optimization.md)**.

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

## Performance

The vLLM serving config is tuned for L40S-class GPUs with FP8 weights, FP8 KV
cache, prefix caching, and headroom-aware GPU memory utilization. Measured p50
latency dropped 28 % vs an unoptimized BF16 baseline; throughput at concurrency 4
roughly doubled. See [`docs/vllm-optimization.md`](./docs/vllm-optimization.md)
for the per-flag rationale, trade-offs, measured numbers, and a mental model
for when to reach for each lever.

There is also a small benchmark script at [`benchmarks/bench.py`](./benchmarks/bench.py)
to reproduce these numbers against your own deployment.

**Observability**: OpenTelemetry traces + Prometheus metrics + provisioned
Grafana dashboards. See [`deploy/observability/README.md`](./deploy/observability/README.md)
for setup. We optimized vLLM (above); now we can OBSERVE it at runtime — KV
cache usage, TTFT distributions, generation throughput, and per-stage RAG
latency are all on the dashboards out of the box.

## Roadmap

Completed:
- [x] Frontend (Next.js + shadcn/ui) — see `frontend/`
- [x] Streaming responses (SSE on `/query/stream`)
- [x] Bearer-token auth + in-memory rate limiting
- [x] Hybrid search (BM25 + dense, RRF fusion)
- [x] Speculative decoding with Llama 3.2 1B draft (see `docs/vllm-optimization.md` §6)
- [x] OpenTelemetry traces + Prometheus metrics + Grafana dashboards (see `deploy/observability/`)
- [x] RAGAS evaluation harness with committed baseline (see `evals/`)
- [x] Helm chart for k8s with l40s/h100/cpu-dev profiles (see `deploy/helm/`)
- [x] CI on every push (lint, type-compile, compose-validate, pytest)

Still on the list:
- [ ] FlashInfer attention backend (custom image with `flashinfer-python`)
- [ ] EAGLE draft model as a comparison row in the optimization doc
- [ ] Multi-collection routing
- [ ] Long-context profile (16K+) for legal/medical workloads

## License

MIT
