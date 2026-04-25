"""Query-time RAG strategies.

Selectable via ``settings.rag_strategy``. Each strategy returns the same
post-rerank passage shape so the rest of the pipeline (graph / api) is
unchanged. See ``docs/rag-strategies.md`` for the research overview and
trade-offs of each technique.
"""

from __future__ import annotations

from . import direct
from .base import StrategyFn

# Optional strategies — registered if importable. Each module exposes
# ``NAME: str`` and ``retrieve(query) -> list[dict]``.
_REGISTRY: dict[str, StrategyFn] = {direct.NAME: direct.retrieve}


def _register(module_name: str) -> None:
    try:
        module = __import__(
            f"enterprise_rag.retrieval.strategies.{module_name}",
            fromlist=["retrieve", "NAME"],
        )
        _REGISTRY[module.NAME] = module.retrieve
    except ImportError:
        # Strategy not yet implemented; that's fine — `direct` is always available.
        pass


for _m in ("hyde", "multi_query", "step_back"):
    _register(_m)


def available() -> list[str]:
    return sorted(_REGISTRY.keys())


def get(name: str | None) -> StrategyFn:
    """Look up a strategy. Falls back to ``direct`` if unknown."""
    if name and name in _REGISTRY:
        return _REGISTRY[name]
    return direct.retrieve
