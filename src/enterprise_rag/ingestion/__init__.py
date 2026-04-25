"""Document ingestion: layout-aware parsing into retrievable chunks."""

from .parser import parse_document
from .types import Chunk

__all__ = ["Chunk", "parse_document"]
