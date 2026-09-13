"""Cognivore: a local-first, torch-free multimodal agent framework.

Public surface is intentionally small -- most of the interesting code lives
in submodules (``cognivore.agent``, ``cognivore.rag``, ``cognivore.tools``,
``cognivore.index``, ``cognivore.media``, ``cognivore.web``) which are
imported directly by the pieces that need them, to keep import time and
optional-dependency surface minimal.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
