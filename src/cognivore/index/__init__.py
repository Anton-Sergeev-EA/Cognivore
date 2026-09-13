"""Vector index backends.

Two collection types (flat/exact and NSW/approximate) are implemented once
in C++ (``native/vector_index.cpp``) for speed, and exposed to Python via
the compiled ``cognivore._native`` pybind11 extension. If that extension
isn't available -- most commonly because no C++ compiler was present when
``pip install`` ran -- everything downstream keeps working against a pure
NumPy flat index instead, so the framework degrades gracefully rather than
refusing to start.

Everything else in the codebase imports ``FlatIndex``/``NSWIndex``/
``is_native`` from *this* module rather than reaching into ``_native`` or
``python_index`` directly, so the fallback is transparent.
"""

from __future__ import annotations

import logging
from typing import Any

from .python_index import FlatIndexPy
from .python_index import SearchResult as SearchResultPy

logger = logging.getLogger(__name__)

# These names are deliberately typed as `Any`: which concrete class each one
# points to is a *runtime* decision (whether the C++ extension built
# successfully), not a static one, so a precise static type here would
# either be a lie (claiming the native-only NSWIndex is always available)
# or require a Protocol duplicating both classes' signatures for no real
# benefit -- every caller either goes through `cognivore.index` (this
# module) and treats these duck-typed, or checks `is_native` first.
FlatIndex: Any
NSWIndex: Any
SearchResult: Any
is_native: bool

try:
    from cognivore import _native

    FlatIndex = _native.FlatIndex
    NSWIndex = _native.NSWIndex
    SearchResult = _native.SearchResult
    is_native = True
except ImportError:  # pragma: no cover - exercised in CI without a compiler
    logger.warning(
        "cognivore._native C++ extension not found; falling back to the pure "
        "NumPy FlatIndexPy. Run `pip install -e .` with a C++17 compiler "
        "available to build the fast native index (and unlock the "
        "approximate NSWIndex, which has no Python fallback)."
    )
    FlatIndex = FlatIndexPy
    NSWIndex = None
    SearchResult = SearchResultPy
    is_native = False

__all__ = ["FlatIndex", "NSWIndex", "SearchResult", "is_native"]
