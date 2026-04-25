"""LangGraph node implementations for the RAG pipeline."""

from __future__ import annotations

from .. import generation, retrieval, safety
from .state import RAGState


def input_safety(state: RAGState) -> RAGState:
    verdict = safety.check_input(state["query"])
    if verdict is not None:
        return {
            "blocking_verdict": verdict,
            "refusal": (
                "Your request was blocked by the safety layer "
                f"({verdict.layer}): {verdict.reason}"
            ),
        }
    return {}


def retrieve(state: RAGState) -> RAGState:
    passages = retrieval.retrieve(state["query"])
    return {
        "passages": passages,
        "context_text": generation.format_context(passages),
    }


def generate(state: RAGState) -> RAGState:
    answer = generation.generate(state["query"], state.get("passages", []))
    return {"answer": answer}


def output_safety(state: RAGState) -> RAGState:
    verdict = safety.check_output(
        state["query"], state.get("context_text", ""), state.get("answer", "")
    )
    if verdict is not None:
        return {
            "blocking_verdict": verdict,
            "refusal": (
                "The model's response was withheld by the safety layer "
                f"({verdict.layer}): {verdict.reason}"
            ),
        }
    return {}


def should_continue_after_input(state: RAGState) -> str:
    return "blocked" if state.get("blocking_verdict") else "ok"


def should_continue_after_output(state: RAGState) -> str:
    return "blocked" if state.get("blocking_verdict") else "ok"
