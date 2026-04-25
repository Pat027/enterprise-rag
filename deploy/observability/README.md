# Observability

OpenTelemetry traces + Prometheus metrics + provisioned Grafana dashboards,
all wired through `docker-compose.yml`.

The story: `docs/vllm-optimization.md` explains how we *tuned* vLLM (FP8
weights, FP8 KV cache, prefix caching, speculative decoding). This stack is
how we **confirm those optimizations behave under real load** — and where we
look first when latency spikes.

## What gets stood up

| Service | Port | Role |
|---|---|---|
| `otel-collector` | 4317 (gRPC), 4318 (HTTP), 8889 (Prom export) | Receives OTLP from the API, fans out to Tempo + Prometheus |
| `tempo` | 3200 | Trace storage (single-binary, local disk) |
| `prometheus` | 9090 | Scrapes API + both vLLMs + collector |
| `grafana` | **3001** | UI — port 3000 is reserved for the future frontend |

## Quickstart

```bash
docker compose up -d otel-collector tempo prometheus grafana
open http://localhost:3001        # admin / admin (anonymous viewer also enabled)
```

Dashboards are pre-provisioned in the `Enterprise RAG` folder:

1. **vLLM Inference** — start here. KV cache %, TTFT p50/p95/p99, throughput
   tokens/sec, request queue depth. These are the metrics that prove the
   optimizations from `docs/vllm-optimization.md` are paying off.
2. **Enterprise RAG — API** — request rate, safety blocks by layer,
   per-stage RAG latency (retrieve / generate / critic), end-to-end p95.

## Trace correlation flow

The API injects the W3C `traceparent` (and a friendlier `X-Trace-Id`) into
every response. To follow a slow request:

```bash
curl -i -X POST http://localhost:8088/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"hello"}' | tee /dev/stderr | grep -i x-trace-id
# X-Trace-Id: 1a2b3c4d...
```

Paste the trace ID into Grafana → Explore → Tempo. You'll see one span tree:

```
POST /query                         (FastAPI auto-instrumentation)
├── input_safety
├── retrieve
├── generate
│   └── HTTP POST /v1/chat/completions   (httpx auto-instrumentation → vLLM)
├── output_safety
└── constitutional_critic
    └── HTTP POST /v1/chat/completions
```

## Prometheus metric names exposed by the API

- `rag_request_total{endpoint, status}`
- `rag_safety_block_total{layer}`
- `rag_retrieval_duration_seconds_bucket{mode}`
- `rag_generation_duration_seconds_bucket`
- `rag_critic_duration_seconds_bucket`
- `http_request_duration_seconds_*` (from `prometheus-fastapi-instrumentator`)

vLLM exposes its own native `vllm:*` series (`vllm:gpu_cache_usage_perc`,
`vllm:num_requests_running`, `vllm:time_to_first_token_seconds_bucket`,
`vllm:generation_tokens_total`, etc.).

## Default credentials

Grafana: `admin` / `admin` (anonymous viewer access is on; no login needed
to read dashboards).

## Toggling

Set `OTEL_ENABLED=false` in `.env` to disable trace export (Prometheus
metrics are always exposed at `/metrics`).
