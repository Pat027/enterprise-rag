# RAGAS Evaluation Harness

End-to-end evaluation of the `enterprise-rag` `/query` API against a checked-in
gold set, scored with [RAGAS](https://docs.ragas.io/) using a local
OpenAI-compatible LLM as the judge.

## Layout

```
evals/
  runner.py              CLI harness
  gold_set.jsonl         14 question / ground_truth pairs
  results/
    <ts>-<label>.json    one file per run (committed)
  README.md              this file
```

## Metrics

| Metric | What it measures | Needs |
|---|---|---|
| `answer_relevancy` | How directly the generated answer addresses the question. RAGAS prompts the judge to generate candidate questions from the answer and embeds them; score is the mean cosine similarity to the original question. | question, answer |
| `faithfulness` | Fraction of claims in the answer that are entailed by the retrieved passages. | question, answer, retrieved contexts |
| `context_precision` | Fraction of retrieved chunks that are actually relevant, weighted by rank. | question, retrieved contexts, ground truth |
| `context_recall` | Fraction of the ground-truth answer's claims that are covered by the retrieved contexts. | question, retrieved contexts, ground truth |

The current `/query` response includes citation metadata (source, page, section
path, rerank score) but **not the passage text**. The harness therefore only
computes `answer_relevancy` and records the other three as skipped, with an
explanation, in every result JSON. Wiring up context-dependent metrics requires
either exposing passage text on the API response or re-running retrieval
inside the harness — both are out of scope for this initial commit.

## Running

Install the eval extras (one-time):

```bash
pip install -e ".[eval]"
```

Run against the live API (default `http://localhost:8088/query`):

```bash
TS=$(date +%Y%m%dT%H%M%S)
python evals/runner.py \
  --label baseline \
  --output evals/results/${TS}-baseline.json
```

### Judge endpoint

The harness needs an OpenAI-compatible chat endpoint to act as the RAGAS judge.
By default it points at `http://localhost:8000/v1` (the `vllm-gen` service when
its port is host-mapped). Override via `--judge-url` / `--judge-api-key` /
`--judge-model` or the matching `JUDGE_URL` / `JUDGE_API_KEY` / `JUDGE_MODEL`
env vars.

`vllm-gen` is **not** host-mapped in the default `docker-compose.yml`. Two ways
to reach it:

1. Use its docker-network IP from the host:

   ```bash
   IP=$(docker inspect enterprise-rag-vllm-gen \
     --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}')
   python evals/runner.py --label baseline \
     --judge-url http://${IP}:8000/v1 \
     --output evals/results/$(date +%Y%m%dT%H%M%S)-baseline.json
   ```

2. Add a temporary port mapping in `docker-compose.yml` (out of scope here).

If the judge is unreachable, the runner still produces a valid result JSON with
all metrics marked as skipped and the reason recorded. Use `--skip-metrics` to
deliberately collect only answers without judging.

### Embeddings

`answer_relevancy` also needs an embedding model. The harness uses
`BAAI/bge-m3` (the same model the API uses for retrieval) loaded locally via
`langchain_community.embeddings.HuggingFaceEmbeddings`. First run will download
the model into the local HF cache.

## Latest scores

Run: `evals/results/20260425T*-baseline.json`

| Metric | Score | N | Notes |
|---|---|---|---|
| answer_relevancy (mean) | see latest result JSON | 9 / 14 | NaNs from judge connection errors are excluded; rows with empty answers (refusals or 500s) are dropped |
| faithfulness | skipped | - | API does not expose retrieved passage text |
| context_precision | skipped | - | API does not expose retrieved passage text |
| context_recall | skipped | - | API does not expose retrieved passage text |

The exact aggregate is in the result JSON committed alongside this README; the
table is intentionally a pointer rather than a copy so we don't drift.

## Adding gold questions

Append a JSON line to `gold_set.jsonl`:

```json
{"question": "...", "ground_truth": "..."}
```

Keep ground truths concise (1-3 sentences). They are used as `reference` in
RAGAS and would feed `context_recall` once contexts are available.

## Result JSON schema

```jsonc
{
  "label": "baseline",
  "timestamp_utc": "2026-04-25T17:13:00+00:00",
  "settings": {
    "api_url": "http://localhost:8088/query",
    "judge_url": "http://172.31.0.5:8000/v1",
    "judge_model": "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8-dynamic",
    "embed_model": "BAAI/bge-m3",
    "retrieval_mode": "default (api defaults)"
  },
  "gold_set_size": 14,
  "rows": [
    {
      "question": "...",
      "ground_truth": "...",
      "answer": "...",
      "refusal": null,
      "blocked_by": null,
      "citations": [...],
      "latency_s": 4.21,
      "error": null
    }
  ],
  "metrics_skipped": {
    "faithfulness": "API /query response does not include retrieved passage text...",
    "context_precision": "...",
    "context_recall": "..."
  },
  "aggregate": {
    "answer_relevancy_mean": 0.81,
    "answer_relevancy_n": 9,
    "answer_relevancy_nan_count": 0
  },
  "per_question": [
    {"question": "...", "answer_relevancy": 0.87}
  ],
  "wall_time_s": 142.3
}
```
