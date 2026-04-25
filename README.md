# Enterprise RAG

An advanced document processing and retrieval-augmented generation system. Built around layout-aware ingestion (tables, figures, OCR), hybrid retrieval with reranking, multi-layer safety, and pluggable LLM providers.

## Why this exists

Most RAG demos do the easy parts (embed text, vector search, prompt) and skip what actually breaks in production: parsing real documents with tables and scans, blocking unsafe inputs and outputs without latency tax, grounding answers with citations the user can verify, and switching LLM providers without rewriting the pipeline.

This system addresses all four.

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
            └───────┬─────────┘      │  generate  (OpenRouter)     │
                    │                │      ▼                      │
            ┌───────▼─────────┐      │  output_safety              │
            │  Qdrant         │      │      ▼                      │
            │  vector store   │◀─────┤  return                     │
            └─────────────────┘      └─────────────────────────────┘

Safety layers (defense-in-depth, all on by default):
  1. OpenAI Moderation       — fast, free, coarse categories
  2. LlamaGuard 3 (Groq)     — policy-aware classification, sub-100ms
  3. Constitutional critic   — Claude as judge against constitution.yaml
```

## Stack

| Layer | Component | Why |
|---|---|---|
| Document parsing | [Docling](https://github.com/DS4SD/docling) | Layout, tables, OCR, figures, formulas — best open-source option |
| Embeddings | BGE-M3 | Multilingual, dense + sparse + multi-vector in one model |
| Vector DB | Qdrant | Purpose-built, hybrid search built in |
| Reranker | BGE-Reranker-v2-m3 | Strong cross-encoder, MIT-licensed |
| LLM | OpenRouter (default) | One API for many models; swap models with one env var |
| Safety L1 | OpenAI Moderation | Free, fast, catches obvious abuse |
| Safety L2 | LlamaGuard 3 (via Groq) | Open-weights policy classifier, hosted at sub-100ms |
| Safety L3 | Anthropic Claude | LLM-as-judge against a written constitution |
| Orchestration | LangGraph | Explicit state machine, easier to reason about than chains |
| API | FastAPI | Standard choice |

## Quickstart

### Prerequisites
- Docker + Docker Compose
- API keys: OpenRouter, OpenAI, Groq, Anthropic

### Run

```bash
git clone https://github.com/Pat027/enterprise-rag.git
cd enterprise-rag
cp .env.example .env
# edit .env with your API keys
docker compose up --build
```

The API is now at `http://localhost:8000`.

### Ingest a document

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@./your-document.pdf"
```

### Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What were the key findings in section 3?"}'
```

Response shape:
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

If a safety layer blocks the request:
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
cp .env.example .env  # fill in keys
uv run enterprise-rag
```

API at `http://localhost:8000`.

## Configuration

All knobs live in `.env`. Notable:

| Var | Default | Notes |
|---|---|---|
| `OPENROUTER_MODEL` | `meta-llama/llama-3.3-70b-instruct` | Any OpenRouter model id |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Any sentence-transformers model |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Any cross-encoder |
| `TOP_K_RETRIEVE` | `20` | Dense recall depth |
| `TOP_K_RERANK` | `5` | Final passages sent to LLM |
| `SAFETY_*` | `true` | Toggle individual safety layers |

The constitutional critic policy is editable at `src/enterprise_rag/safety/constitution.yaml`.

## Project layout

```
src/enterprise_rag/
├── ingestion/      Docling parsing → structured chunks
├── retrieval/      Qdrant + BGE-M3 + BGE reranker
├── safety/         3-layer moderation pipeline
├── generation/     OpenRouter LLM client + prompts
├── graph/          LangGraph state machine
├── api/            FastAPI app
├── config.py       Pydantic settings
└── cli.py          uvicorn launcher
```

## Roadmap

- [ ] Frontend (Next.js + shadcn/ui)
- [ ] Hybrid search (BM25 + dense)
- [ ] Multi-collection routing
- [ ] Streaming responses
- [ ] OpenTelemetry → Grafana dashboard
- [ ] RAGAS evaluation harness

## License

MIT
