"""Test bootstrap: ensure src/ is importable and stub heavy optional deps if missing."""

from __future__ import annotations

import sys
import types
from pathlib import Path

# Make `src/` layout importable without an editable install.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _ensure_stub(mod_name: str) -> types.ModuleType:
    """Insert a no-op module into sys.modules if it isn't already importable."""
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    mod = types.ModuleType(mod_name)
    sys.modules[mod_name] = mod
    return mod


# docling is heavy and not required for unit tests of pure helpers — stub it.
try:  # pragma: no cover - import-time branch
    import docling  # noqa: F401
except ImportError:
    _ensure_stub("docling")
    dc = _ensure_stub("docling.document_converter")

    class _StubConverter:
        def convert(self, *_a, **_kw):  # pragma: no cover
            raise RuntimeError("docling stubbed in tests")

    dc.DocumentConverter = _StubConverter
