"""Text embedding backends.

``FastEmbedModel`` wraps `fastembed` (ONNX Runtime under the hood -- no
PyTorch dependency, which matters a lot for a CPU-only laptop: it keeps the
whole stack, LLM included, free of a multi-GB torch install). It downloads
a small quantized model to a local cache on first use.

``HashingEmbedder`` needs no model download and no network access at all:
it's the classic "hashing trick" (feature hashing) used by Vowpal Wabbit and
scikit-learn's ``HashingVectorizer`` -- hash each word/word-bigram into a
slot of a fixed-size vector with a pseudo-random sign, then L2-normalize.
It's a real, if purely lexical, embedding technique (not a stub), and it's
what lets the RAG pipeline and its test suite run fully offline.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Protocol, runtime_checkable

import numpy as np

from cognivore.rag.tokenize import STOPWORDS, stem_tokens, tokenize

logger = logging.getLogger(__name__)


@runtime_checkable
class EmbeddingModel(Protocol):
    dim: int

    # A stable identifier of *which* vectors this model produces. Stored
    # alongside a persisted knowledge base so that switching models (or a
    # tokenizer change in HashingEmbedder) is detected on load and the
    # store is re-embedded, instead of silently comparing query vectors
    # from one model against document vectors from another.
    @property
    def id(self) -> str: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    # Bump the version whenever tokenization or hashing changes, so stores
    # embedded with the old scheme are re-embedded on load.
    _VERSION = 5

    def __init__(self, dim: int = 384, ngram_range: tuple[int, int] = (1, 2)) -> None:
        self.dim = dim
        self.ngram_range = ngram_range

    @property
    def id(self) -> str:
        lo, hi = self.ngram_range
        return f"hashing-v{self._VERSION}:{self.dim}:{lo}-{hi}"

    def _tokens(self, text: str) -> list[str]:
        # Function words ("les", "et", "der", "的") carry no topic and, being
        # the most frequent tokens, would otherwise dominate the hashed
        # vector; they are dropped unless nothing else is left.
        all_words = tokenize(text)
        words = [w for w in all_words if w not in STOPWORDS] or all_words
        # Stemmed, so that "тарифы" and "тариф" land in the same bucket.
        words = stem_tokens(words, text)
        if not words:
            return [text.lower()]
        lo, hi = self.ngram_range
        tokens: list[str] = []
        for n in range(lo, hi + 1):
            for i in range(len(words) - n + 1):
                tokens.append(" ".join(words[i : i + n]))
        return tokens or [text.lower()]

    def _embed_one(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in self._tokens(text):
            digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 1e-10:
            vec /= norm
        return vec.tolist()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]


class FastEmbedModel:
    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        dim: int = 384,
    ) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise RuntimeError(
                "fastembed is not installed. Install the default extras "
                "(`pip install cognivore`) or use HashingEmbedder for a "
                "zero-download offline embedder."
            ) from exc
        self._model = TextEmbedding(model_name=model_name)
        self.model_name = model_name
        self.dim = dim

    @property
    def id(self) -> str:
        return f"fastembed:{self.model_name}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [vec.tolist() for vec in self._model.embed(texts)]


def get_default_embedder(
    model_name: str, dim: int, prefer_fastembed: bool = True
) -> EmbeddingModel:
    """Tries the real semantic embedder first, falling back to the
    dependency-free hashing embedder if fastembed isn't installed or the
    model can't be downloaded (e.g. no network access)."""
    if prefer_fastembed:
        try:
            return FastEmbedModel(model_name=model_name, dim=dim)
        except Exception:
            logger.warning(
                "Semantic embedder unavailable, falling back to HashingEmbedder.",
                exc_info=True,
            )
    return HashingEmbedder(dim=dim)
