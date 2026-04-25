"""Compile the LangGraph state machine for query → answer."""

from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, StateGraph

from . import nodes
from .state import RAGState


@lru_cache(maxsize=1)
def build_graph():
    g = StateGraph(RAGState)
    g.add_node("input_safety", nodes.input_safety)
    g.add_node("retrieve", nodes.retrieve)
    g.add_node("generate", nodes.generate)
    g.add_node("output_safety", nodes.output_safety)

    g.set_entry_point("input_safety")
    g.add_conditional_edges(
        "input_safety",
        nodes.should_continue_after_input,
        {"ok": "retrieve", "blocked": END},
    )
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "output_safety")
    g.add_conditional_edges(
        "output_safety",
        nodes.should_continue_after_output,
        {"ok": END, "blocked": END},
    )

    return g.compile()


def run_query(query: str) -> dict:
    """Synchronously run a query through the full pipeline."""
    graph = build_graph()
    result = graph.invoke({"query": query})
    return result
