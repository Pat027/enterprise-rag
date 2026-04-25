# vLLM Optimization for Enterprise RAG

This document explains the vLLM tuning choices in `docker-compose.yml`, why each one fits *this* workload, and the measured impact. The goal is reproducible performance reasoning — not a generic checklist.

## TL;DR

Two rounds of optimization. Round 1 applied five flags (FP8 weights, FP8 KV, prefix caching, tuned context length and GPU mem util). Round 2 added speculative decoding with a Llama 3.2 1B draft. Combined effect: **p50 latency cut 55 %**, **throughput up 218 %** sequentially and **75 %** higher at concurrency 4.

| Metric | Baseline | Round 1 (FP8 etc.) | Round 2 (+ Spec Decode) | Total Δ vs baseline |
| --- | --- | --- | --- | --- |
| Mean latency (sequential) | 9.27 s | 7.42 s | **4.86 s** | **−48 %** |
| p50 | 10.79 s | 7.78 s | **4.81 s** | **−55 %** |
| p95 | 11.66 s | 8.80 s | **5.86 s** | **−50 %** |
| Throughput (sequential) | 0.055 qps | 0.101 qps | **0.175 qps** | **+218 %** |
| Throughput (concurrent, 4 inflight) | — | 0.200 qps | **0.349 qps** | *vs round 1: +75 %* |
| Target-model weights memory | 14.99 GB | 8.49 GB | 8.49 GB | **−43 %** |
| Total weights memory | 14.99 GB | 8.49 GB | 10.81 GB (+ 2.32 GB draft) | **−28 %** |

Workload baseline is the `benchmarks/bench.py` HR-interview query set against an indexed PDF. Hardware: 1× NVIDIA L40S 48 GB.

---

## How to read this document

Each section follows the same structure:
1. **Flag** — what was set
2. **Mechanism** — what vLLM actually does internally
3. **Why it fits this workload** — why the trade-off makes sense for *enterprise-rag*, not generically
4. **Trade-off** — what we gave up
5. **Measured impact** — numbers from before/after

Optimizations are listed in order of leverage on this workload.

---

## 1. FP8 Model Weights — `--model=neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8-dynamic`

**Mechanism.** Stores model parameters as 8-bit floats (E4M3) instead of bfloat16. The L40S has native FP8 tensor cores (Ada Lovelace, compute capability 8.9), so the matmul kernels run in FP8 throughout — no dequantization to BF16 happens at runtime.

**Why it fits this workload.** L40S is the explicit target. On Ampere (A100) FP8 doesn't exist in hardware and you'd dequantize, getting only the memory benefit. On Ada you get both: memory and compute throughput.

**Trade-off.** Quality loss on standard benchmarks is typically <1 % for the `dynamic` (per-tensor static-scale) variant. neuralmagic publishes evals showing MMLU/GSM8K within noise of BF16. If you required maximum fidelity for legal/medical text, you'd revert to BF16.

**Measured impact.**
- Model memory: **14.99 GB → 8.49 GB** (−43 %)
- Decode throughput: ≈1.3–1.5× faster on L40S (vendor-published; consistent with our observed end-to-end gain)

---

## 2. FP8 KV Cache — `--kv-cache-dtype=fp8`

**Mechanism.** During autoregressive decode, every generated token writes K and V tensors into a per-sequence cache. By default this cache lives in BF16 (2 bytes per element). `fp8` stores it in 1 byte per element, halving the KV memory budget.

**Why it fits this workload.** KV cache, not weights, is the dominant runtime memory cost. With 4096 tokens of context × 32 layers × 8 KV heads × 128 head dim × 2 (K and V) × 2 bytes = ~17 MB per sequence in BF16. At concurrency=4 that's 68 MB just for the active sequences, plus pre-allocated capacity. Halving this lets vLLM pre-allocate roughly **2× more KV slots**, which directly increases the maximum number of concurrent batched sequences before requests queue.

**Trade-off.** Without per-layer scaling factors in the checkpoint (vLLM warns about this — neuralmagic's FP8 model ships only weight scales, not KV scales), vLLM uses scale=1.0 for KV. This can cost ~1 % accuracy on long-context tasks. Negligible for our retrieval flow where the LLM rarely sees more than 2K tokens.

**Bonus warning we hit.** vLLM 0.6.6 surfaces:
```
Cannot use FlashAttention-2 backend for FP8 KV cache.
Please use FlashInfer backend with FP8 KV Cache for better performance
```
FlashAttention-2 doesn't support FP8 KV; it falls back to a slower xformers path. To get the optimal kernel you'd install `flashinfer-python` and set `VLLM_ATTENTION_BACKEND=FLASHINFER`. Listed under "future levers" below since it requires a custom image.

---

## 3. Prefix Caching — `--enable-prefix-caching`

**Mechanism.** vLLM hashes prompt token prefixes and stores their KV in a content-addressable cache. When a new request shares any prefix with a cached one, those tokens are *not* re-prefilled — their KV is read directly. This turns prompt prefill from O(n) into O(n − cached) tokens.

**Why it fits this workload.** Our system prompt is **identical for every query**:
```
You are an expert assistant that answers questions strictly from the
provided context. Cite sources inline using [n] notation matching the
numbered context entries. If the context is insufficient, say so
explicitly rather than guessing. Be precise; do not invent facts that
are not supported by the context.
```
That's ~80 tokens of prefix that *every* request shares. The constitutional critic prompt is also constant. Prefix caching makes each subsequent request skip ~150–200 tokens of prefill — ~10–15 % of TTFT savings on first-token, more on warm cache.

**Trade-off.** Strict win. The cache is bounded by GPU memory and evicts LRU entries; in pathological multi-tenant cases you might evict useful prefixes, but our single-app workload never sees that.

**Measured impact.** Hard to isolate (it compounds with FP8) — but the −28 % p50 latency reduction is consistent with both prefill skipping and FP8 throughput gains stacking.

---

## 4. Reduced Max Model Length — `--max-model-len=4096`

**Mechanism.** vLLM pre-allocates KV cache slots sized to `max_model_len` per sequence. Halving this halves the memory needed per slot, doubling the number of slots vLLM can hold simultaneously.

**Why it fits this workload.** Concrete prompt budget audit:
- System prompt: ~80 tokens
- 5 retrieved passages × ~200 tokens each: ~1 000 tokens
- User query: ~30 tokens
- Constitutional critic (judging response): ~600 tokens (response + principles)
- Generated answer: ~500 tokens

Worst-case input + output for a single turn is well under 3 000 tokens. The default 8 192 was overkill. Setting it to 4 096 gives safety margin without wasting KV cache on context the workload never uses.

**Trade-off.** If a user uploads a document with paragraphs longer than ~3 000 tokens and it ends up in retrieval, generation could truncate. Mitigations: (a) chunk size in `ingestion/parser.py` already merges to ~200 chars; (b) for legal-document or long-context use cases, raise to 8 192 or 16 384.

**Measured impact.** Allowed Optimization 5 (higher GPU mem util) without OOM.

---

## 5. Higher GPU Memory Utilization — `--gpu-memory-utilization=0.90`

**Mechanism.** Tells vLLM what fraction of total GPU memory it can reserve for KV cache. The remaining 10 % is left as a safety margin for activation spikes, CUDA workspace, etc.

**Why it fits this workload.** Once Optimizations 1+2+4 freed up memory (smaller weights, smaller KV per token, smaller per-sequence allocation), there was headroom to grow the KV pool. We have a **dedicated GPU 0** for vllm-gen with no other processes — there's no reason to leave 45 % of memory unused. The previous `0.55` was a defensive default for when 70B FP16 was sharing the GPU.

**Trade-off.** If something else lands on GPU 0, this OOMs. We control the deployment so it's safe; in a multi-tenant shared-GPU scenario you'd dial this back to 0.70–0.80.

**Measured impact.** Combined with FP8 KV, this is what enables `concurrency=4` to nearly double throughput — there's enough KV cache for 4+ concurrent sequences.

---

## 6. Speculative Decoding — `--speculative-model=meta-llama/Llama-3.2-1B-Instruct --num-speculative-tokens=5`

**Mechanism.** A small "draft" model proposes N tokens at a time; the target model verifies all N in a single forward pass via [rejection sampling](https://arxiv.org/abs/2302.01318). When the draft tokens match what the target would have generated, you get N tokens for the price of one forward pass. When the draft is wrong, the target falls back to its own sampled token at the rejection point. Throughput scales with **acceptance rate** (the fraction of draft tokens the target accepts).

**Why it fits this workload.** Two reasons:

1. **Same-family pairing.** Llama 3.2 1B and Llama 3.1 8B share the same tokenizer (Llama 3 BPE, 128 K vocab) and were trained on related data. Same-family draft/target pairs typically achieve 60–80 % acceptance — much higher than cross-family or random-init drafts.

2. **The constitutional critic dominates total latency.** Generating ~500 tokens of structured JSON, sequentially, was the long pole at 25–40 % of total request time. Spec decoding shines on long generations because the per-token amortization of forward passes is what matters — short answers don't benefit as much.

**Trade-off.**
- **Memory.** Adds 2.32 GB for the BF16 1B draft on top of the 8.49 GB FP8 target. To make room, GPU memory utilization was dropped 0.90 → 0.80 (KV cache budget shrinks slightly).
- **vLLM 0.6.6 incompatibility with guided decoding.** vLLM's xgrammar-based JSON-mode (`response_format={"type":"json_object"}`) crashes when combined with speculative decoding (a known upstream bug; the sampler's `input_ids[-1]` indexing fails). Mitigation: removed `response_format` from the constitutional critic call in `src/enterprise_rag/safety/constitutional.py`. Our existing tolerant JSON parser (find first `{`, last `}`, decode the slice) handles the unguided output reliably — Llama 3.1 8B follows the explicit "reply ONLY with strict JSON of this shape" instruction in practice.
- **Async output disabled.** vLLM logs `Async output processing is not supported with speculative decoding currently` — this is a vLLM limitation, costs a small amount of theoretical throughput at very high concurrency. Negligible at our scale.

**Measured impact.**

| Metric | Without spec decode | With spec decode | Δ |
| --- | --- | --- | --- |
| Mean latency (sequential) | 7.42 s | **4.86 s** | **−35 %** |
| p50 | 7.78 s | **4.81 s** | **−38 %** |
| p95 | 8.80 s | **5.86 s** | **−33 %** |
| Throughput (sequential) | 0.101 qps | **0.175 qps** | **+73 %** |
| Throughput (concurrent, 4 inflight) | 0.200 qps | **0.349 qps** | **+75 %** |

The lift is consistent across sequential and concurrent regimes, which means we're not hitting an obvious bottleneck.

**Why this size of speedup is plausible.** The naive expectation for spec decoding with a 1B/8B family is 2–3× on memory-bound decode. We see ~1.7× on sequential workloads. The gap is explained by: (a) our target is already FP8 (compute-bound matters more, spec decode helps less than the BF16 case); (b) the workload has a meaningful prefill component (retrieved context + system prompt = ~1 K tokens of input) where spec decoding doesn't help; (c) safety pipeline calls (LlamaGuard input + output) sit on a separate vLLM instance and aren't sped up. Pure decode speedup on the 70 B BF16 case would be larger.

---

## Other levers, not yet enabled

### FlashInfer attention backend — `VLLM_ATTENTION_BACKEND=FLASHINFER`

The vLLM warning we hit: "Cannot use FlashAttention-2 backend for FP8 KV cache." FlashInfer has a kernel that handles FP8 KV efficiently. Not enabled because the upstream `vllm/vllm-openai:v0.6.6` image doesn't bundle `flashinfer-python` — would require a custom image (dependency build adds ~5 minutes to image build).

When to add it: when GPU memory is the binding constraint, since FlashInfer's FP8-aware kernels also save ~10 % KV pressure.

### Tuned `--max-num-seqs`

Default is 256, which is far more than we'll see. Lowering it to e.g. 32 reduces some bookkeeping overhead vLLM does per scheduling cycle. Marginal in practice.

When to add it: as part of a structured load-test grid where you sweep concurrency × max_num_seqs and pick the throughput-optimal point.

### Chunked prefill — `--enable-chunked-prefill`

Splits long prefill requests into chunks so they don't block decode. Default is on in vLLM ≥ 0.6.0 for models > 32K context. Our 4K context flow doesn't benefit measurably.

### Pipeline parallelism — `--pipeline-parallel-size`

Splits layers across GPUs (latency-tolerant scaling). Irrelevant for an 8B model on one GPU; would matter at 70B+ where the model itself doesn't fit.

---

## Reproducing these numbers

```bash
# Start the stack with the optimized config
docker compose up -d

# Wait for both vLLM containers to report healthy
docker compose ps

# Ingest a test document
curl -X POST http://localhost:8088/ingest -F "file=@./your.pdf"

# Run benchmark — sequential
python benchmarks/bench.py --warmup 2 --runs 8 --concurrency 1

# Run benchmark — concurrent (shows continuous-batching wins)
python benchmarks/bench.py --warmup 2 --runs 8 --concurrency 4
```

To compare against an unoptimized baseline, edit `docker-compose.yml`:
- Revert `--model` to `meta-llama/Llama-3.1-8B-Instruct`
- Remove `--kv-cache-dtype=fp8`
- Remove `--enable-prefix-caching`
- Set `--gpu-memory-utilization=0.55`
- Set `--max-model-len=8192`

…then `docker compose up -d --force-recreate vllm-gen` and re-run the benchmark.

---

## Mental model — what to tune when

| Symptom | First lever | Second lever |
| --- | --- | --- |
| Out-of-memory loading model | FP8 model weights, smaller model | Lower `gpu-memory-utilization` |
| Long TTFT on warm requests | Prefix caching | Lower `max-model-len` |
| Decode throughput plateaus under load | FP8 KV + higher `gpu-memory-utilization` | `--max-num-seqs` tuning |
| Long-output decode is the long pole | **Speculative decoding** (same-family draft) | Smaller target model |
| Multi-tenant GPU contention | Conservative `gpu-memory-utilization` | Pin to dedicated GPU |
| Long-document RAG (16K+ context) | Chunked prefill, higher `max-model-len` | Long-context attention kernels (FlashInfer) |
| Spec decoding + guided decoding crashes | Drop `response_format`, parse leniently | Wait for vLLM upstream fix |
