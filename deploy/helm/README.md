# enterprise-rag Helm chart

Deploys the four-service Enterprise RAG stack to Kubernetes:

| Component   | Workload      | GPU | Notes                                                          |
| ----------- | ------------- | --- | -------------------------------------------------------------- |
| `api`       | Deployment    | 1   | FastAPI gateway. Embeddings + reranker run in-process.         |
| `vllm-gen`  | Deployment    | 1   | Llama 3.1 8B Instruct + Llama 3.2 1B speculative draft.        |
| `vllm-guard`| Deployment    | 1   | LlamaGuard 3 8B safety classifier.                             |
| `qdrant`    | StatefulSet   | 0   | Vector store with `PersistentVolumeClaim`.                     |

The chart mirrors the reference `docker-compose.yml`.

## Prerequisites

- Kubernetes 1.27+
- Helm 3.12+
- A storage class for `PersistentVolumeClaim`s (Qdrant + HF model cache)
- For GPU profiles, one of:
  - **NVIDIA GPU Operator** (sets up `nvidia` runtimeClass automatically), or
  - **NVIDIA Kubernetes device plugin** (set `gpuOperator.runtimeClassName: ""`)
- HuggingFace account with accepted licenses for:
  - `meta-llama/Llama-3.1-8B-Instruct` (or the FP8 dynamic variant on L40S)
  - `meta-llama/Llama-3.2-1B-Instruct`
  - `meta-llama/Llama-Guard-3-8B`
- An access token from <https://huggingface.co/settings/tokens>

## Build and push the API image

The chart references `rag-api:latest`. Build and push to a registry your
cluster can pull from:

```bash
docker build -t <registry>/rag-api:0.1.0 .
docker push <registry>/rag-api:0.1.0
```

Then either set `--set api.image.repository=<registry>/rag-api --set api.image.tag=0.1.0`
on the install command, or add it to your overrides file.

## Profiles

Three profiles ship with the chart. Pick one and pass it via `-f`:

| Profile             | When to use                                     | LLM stack                                            |
| ------------------- | ----------------------------------------------- | ---------------------------------------------------- |
| `values-l40s.yaml`  | Production reference — 1 L40S per LLM container | Llama 3.1 8B FP8 dynamic + spec decode, FP8 KV cache |
| `values-h100.yaml`  | Production on H100 nodes                        | Llama 3.1 8B BF16, FP8 KV cache, higher mem-util     |
| `values-cpu-dev.yaml` | Laptops, kind, minikube, CI                   | Ollama llama3.1:8b q4_K_M; **safety disabled**       |

### Why H100 drops weight FP8

Llama 3.1 8B at TP=1 on a single H100 has plenty of VRAM headroom and
plenty of compute. BF16 weights are simpler and avoid the small accuracy
hit FP8 weight quant introduces. We keep FP8 for the *KV cache* because
decode is KV-bandwidth-bound regardless of GPU.

### CPU-dev profile is NOT a real safety stack

The CPU-dev profile sets `SAFETY_LLAMAGUARD=false` and
`SAFETY_CONSTITUTIONAL=false` because LlamaGuard 3 8B is too slow on CPU
to be useful. **Do not run this profile in production.**

## Install

```bash
# 1) Render and review:
helm template enterprise-rag deploy/helm/enterprise-rag/ \
  -f deploy/helm/enterprise-rag/values-l40s.yaml \
  --set secrets.hfToken=hf_xxx \
  --set api.image.repository=ghcr.io/yourorg/rag-api \
  --set api.image.tag=0.1.0 \
  | less

# 2) Install:
helm upgrade --install enterprise-rag deploy/helm/enterprise-rag/ \
  --namespace rag --create-namespace \
  -f deploy/helm/enterprise-rag/values-l40s.yaml \
  --set secrets.hfToken=hf_xxx \
  --set secrets.openaiApiKey=sk-xxx \
  --set api.image.repository=ghcr.io/yourorg/rag-api \
  --set api.image.tag=0.1.0
```

## Production secret handling

Don't put real secrets in `values.yaml`. Pre-create a Secret and reference
it with `secrets.existingSecret`:

```bash
kubectl -n rag create secret generic rag-secrets \
  --from-literal=HF_TOKEN=hf_xxx \
  --from-literal=API_KEYS=sk-prod-1,sk-prod-2 \
  --from-literal=OPENAI_API_KEY=sk-xxx \
  --from-literal=ANTHROPIC_API_KEY="" \
  --from-literal=OPENROUTER_API_KEY=""

helm upgrade --install enterprise-rag deploy/helm/enterprise-rag/ \
  --namespace rag \
  -f deploy/helm/enterprise-rag/values-l40s.yaml \
  --set secrets.existingSecret=rag-secrets
```

## Scaling

- **api** — set `autoscaling.enabled=true` to turn on the HPA (CPU-based,
  defaults: min 2, max 10, target 70 %).
- **vllm-gen / vllm-guard** — vLLM is stateful per-Pod (KV cache, prefix
  cache, draft model). The chart hard-codes `replicas: 1`. To scale, deploy
  the chart again under a different release name and put your own
  load-balancer or routing layer in front.
- **qdrant** — single-node StatefulSet. For HA, use Qdrant Cloud or run the
  upstream Qdrant Helm chart in distributed mode.

## Ingress

Off by default. Turn on with:

```yaml
ingress:
  enabled: true
  className: nginx
  host: rag.example.com
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  tls:
    enabled: true
    secretName: rag-tls
```

## Verify

```bash
helm lint deploy/helm/enterprise-rag/
helm template enterprise-rag deploy/helm/enterprise-rag/ \
  -f deploy/helm/enterprise-rag/values-l40s.yaml \
  | kubectl apply --dry-run=client -f -
```

## Uninstall

```bash
helm uninstall enterprise-rag -n rag
# PVCs are NOT deleted automatically; remove with:
kubectl -n rag delete pvc -l app.kubernetes.io/instance=enterprise-rag
```
