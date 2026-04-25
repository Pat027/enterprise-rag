"""RAGAS evaluation harness for enterprise-rag.

Runs a gold-set of questions against the live `/query` endpoint, then scores
the answers with RAGAS using a local OpenAI-compatible LLM as the judge.

The `/query` endpoint does not return retrieved passage text (only citation
metadata), so context-dependent metrics (faithfulness, context_precision,
context_recall) are skipped unless `--contexts-from` provides them. The
default run computes `answer_relevancy` only, which needs just the question
and answer.

Usage:
    python evals/runner.py --label baseline \\
        --output evals/results/$(date +%Y%m%dT%H%M%S)-baseline.json

    # With a reachable judge (override default):
    python evals/runner.py --label baseline \\
        --judge-url http://172.31.0.5:8000/v1 \\
        --judge-model neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8-dynamic
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

DEFAULT_API_URL = "http://localhost:8088/query"
DEFAULT_JUDGE_URL = "http://localhost:8000/v1"
DEFAULT_JUDGE_MODEL = "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8-dynamic"
DEFAULT_EMBED_MODEL = "BAAI/bge-m3"

GOLD_SET = Path(__file__).parent / "gold_set.jsonl"


def load_gold_set(path: Path) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def query_api(api_url: str, question: str, timeout: float) -> dict[str, Any]:
    """POST to /query; return parsed response or an error envelope."""
    try:
        r = httpx.post(api_url, json={"query": question}, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:  # noqa: BLE001
        return {"_error": f"{type(e).__name__}: {e}"}


def build_judge(judge_url: str, judge_api_key: str, judge_model: str):
    """Build a RAGAS-compatible LLM wrapper. Returns None on failure."""
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    llm = ChatOpenAI(
        base_url=judge_url,
        api_key=judge_api_key,
        model=judge_model,
        temperature=0.0,
        timeout=120,
        max_retries=1,
    )
    return LangchainLLMWrapper(llm)


def build_embeddings(embed_model: str):
    """Build RAGAS-compatible embeddings using HuggingFace BGE-M3 locally."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
    from ragas.embeddings import LangchainEmbeddingsWrapper

    hf = HuggingFaceEmbeddings(model_name=embed_model)
    return LangchainEmbeddingsWrapper(hf)


def probe_judge(judge_url: str, judge_api_key: str, timeout: float = 5.0) -> tuple[bool, str]:
    try:
        r = httpx.get(
            f"{judge_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {judge_api_key}"},
            timeout=timeout,
        )
        if r.status_code == 200:
            return True, "ok"
        return False, f"HTTP {r.status_code}: {r.text[:200]}"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def run(
    label: str,
    api_url: str,
    judge_url: str,
    judge_api_key: str,
    judge_model: str,
    embed_model: str,
    output_path: Path,
    per_q_timeout: float,
    skip_metrics: bool,
) -> dict[str, Any]:
    gold = load_gold_set(GOLD_SET)
    started = time.time()

    # 1) Collect answers from the live API.
    rows: list[dict[str, Any]] = []
    print(f"[runner] querying {len(gold)} questions against {api_url}", flush=True)
    for i, item in enumerate(gold, 1):
        q = item["question"]
        gt = item["ground_truth"]
        t0 = time.time()
        resp = query_api(api_url, q, timeout=per_q_timeout)
        dt = time.time() - t0
        row: dict[str, Any] = {
            "question": q,
            "ground_truth": gt,
            "answer": resp.get("answer") or "",
            "refusal": resp.get("refusal"),
            "blocked_by": resp.get("blocked_by"),
            "citations": resp.get("citations") or [],
            "latency_s": round(dt, 3),
            "error": resp.get("_error"),
        }
        rows.append(row)
        print(
            f"  [{i:02d}/{len(gold)}] {dt:5.2f}s  ans_len={len(row['answer'])}  "
            f"err={row['error']}",
            flush=True,
        )

    result: dict[str, Any] = {
        "label": label,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "settings": {
            "api_url": api_url,
            "judge_url": judge_url,
            "judge_model": judge_model,
            "embed_model": embed_model,
            "retrieval_mode": "default (api defaults)",
        },
        "gold_set_size": len(gold),
        "rows": rows,
        "metrics_skipped": {},
        "aggregate": {},
        "per_question": [],
        "wall_time_s": None,
    }

    # 2) RAGAS scoring.
    # Context-dependent metrics need passage text, which /query does not return.
    result["metrics_skipped"]["faithfulness"] = (
        "API /query response does not include retrieved passage text; only citation "
        "metadata. Re-running retrieval here would duplicate the pipeline."
    )
    result["metrics_skipped"]["context_precision"] = result["metrics_skipped"]["faithfulness"]
    result["metrics_skipped"]["context_recall"] = result["metrics_skipped"]["faithfulness"]

    if skip_metrics:
        result["metrics_skipped"]["answer_relevancy"] = "--skip-metrics flag set"
        result["wall_time_s"] = round(time.time() - started, 2)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2))
        print(f"[runner] wrote {output_path} (metrics skipped)", flush=True)
        return result

    ok, msg = probe_judge(judge_url, judge_api_key)
    if not ok:
        result["metrics_skipped"]["answer_relevancy"] = (
            f"Judge endpoint {judge_url} unreachable: {msg}. "
            "Re-run with --judge-url pointing at a reachable OpenAI-compatible "
            "endpoint (e.g. http://172.31.0.5:8000/v1 for the rag_default docker network)."
        )
        result["wall_time_s"] = round(time.time() - started, 2)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2))
        print(f"[runner] wrote {output_path} (judge unreachable)", flush=True)
        return result

    print(f"[runner] judge reachable at {judge_url}; computing answer_relevancy", flush=True)
    try:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.metrics import ResponseRelevancy

        judge_llm = build_judge(judge_url, judge_api_key, judge_model)
        embeddings = build_embeddings(embed_model)

        samples = []
        scoreable_rows = []
        for row in rows:
            if not row["answer"] or row["error"]:
                continue
            samples.append(
                SingleTurnSample(
                    user_input=row["question"],
                    response=row["answer"],
                    reference=row["ground_truth"],
                    # answer_relevancy needs retrieved_contexts for some impls;
                    # pass an empty list for compatibility.
                    retrieved_contexts=[],
                )
            )
            scoreable_rows.append(row)

        if not samples:
            result["metrics_skipped"]["answer_relevancy"] = "no scoreable rows (all empty/errored)"
        else:
            ds = EvaluationDataset(samples=samples)
            res = evaluate(
                ds,
                metrics=[ResponseRelevancy()],
                llm=judge_llm,
                embeddings=embeddings,
                show_progress=False,
            )
            df = res.to_pandas()
            # per-question
            per_q = []
            col = (
                "answer_relevancy"
                if "answer_relevancy" in df.columns
                else next((c for c in df.columns if "relevanc" in c.lower()), None)
            )
            import math

            for i, row in enumerate(scoreable_rows):
                raw = float(df[col].iloc[i]) if col else float("nan")
                score = None if math.isnan(raw) else round(raw, 4)
                per_q.append({"question": row["question"], "answer_relevancy": score})
            result["per_question"] = per_q
            if col:
                vals = [p["answer_relevancy"] for p in per_q if p["answer_relevancy"] is not None]
                if vals:
                    result["aggregate"]["answer_relevancy_mean"] = round(sum(vals) / len(vals), 4)
                    result["aggregate"]["answer_relevancy_n"] = len(vals)
                    result["aggregate"]["answer_relevancy_nan_count"] = len(per_q) - len(vals)
    except Exception as e:  # noqa: BLE001
        result["metrics_skipped"]["answer_relevancy"] = (
            f"RAGAS scoring failed: {type(e).__name__}: {e}\n"
            f"{traceback.format_exc(limit=3)}"
        )

    result["wall_time_s"] = round(time.time() - started, 2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    print(f"[runner] wrote {output_path}", flush=True)
    return result


def main() -> int:
    p = argparse.ArgumentParser(description="RAGAS eval harness for enterprise-rag")
    p.add_argument("--label", default="baseline", help="Label for this run")
    p.add_argument("--api-url", default=os.environ.get("API_URL", DEFAULT_API_URL))
    p.add_argument("--judge-url", default=os.environ.get("JUDGE_URL", DEFAULT_JUDGE_URL))
    p.add_argument("--judge-api-key", default=os.environ.get("JUDGE_API_KEY", "EMPTY"))
    p.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", DEFAULT_JUDGE_MODEL))
    p.add_argument("--embed-model", default=os.environ.get("EMBED_MODEL", DEFAULT_EMBED_MODEL))
    p.add_argument("--output", required=True, help="Path for the JSON results file")
    p.add_argument(
        "--per-question-timeout",
        type=float,
        default=300.0,
        help="HTTP timeout per /query call (seconds)",
    )
    p.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Only collect answers; skip RAGAS scoring (useful when judge is unreachable)",
    )
    args = p.parse_args()

    out = Path(args.output)
    run(
        label=args.label,
        api_url=args.api_url,
        judge_url=args.judge_url,
        judge_api_key=args.judge_api_key,
        judge_model=args.judge_model,
        embed_model=args.embed_model,
        output_path=out,
        per_q_timeout=args.per_question_timeout,
        skip_metrics=args.skip_metrics,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
