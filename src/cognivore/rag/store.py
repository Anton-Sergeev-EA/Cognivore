"""In-process document store: chunk metadata + a vector index + BM25, with
simple hybrid (vector + lexical) retrieval and score-level reranking.

Persistence is a single JSON sidecar file for chunk text/metadata plus the
vector index's own binary ``serialize()``/``deserialize()`` -- enough for a
single-machine, single-user local agent; swapping in a real database is a
matter of implementing the same three methods (``add_documents``,
``search``, ``save``/``load``) against Postgres/pgvector or similar.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from cognivore.index import FlatIndex, NSWIndex, is_native
from cognivore.rag.chunking import Chunk, split_text
from cognivore.rag.embeddings import EmbeddingModel
from cognivore.rag.tokenize import index_terms

logger = logging.getLogger(__name__)


def embedder_id(embedder: EmbeddingModel) -> str:
    """Identifier of the vectors ``embedder`` produces (see
    ``EmbeddingModel.id``); third-party embedders without one fall back to
    their class name and dimension."""
    return str(getattr(embedder, "id", f"{type(embedder).__name__}:{embedder.dim}"))


@dataclass
class RetrievedChunk:
    id: int
    text: str
    source: str
    score: float
    vector_score: float
    lexical_score: float


class DocumentStore:
    def __init__(
        self,
        embedder: EmbeddingModel,
        use_approximate_index: bool = True,
        vector_weight: float = 0.65,
    ) -> None:
        self.embedder = embedder
        self.vector_weight = vector_weight
        self._use_ann = use_approximate_index and is_native and NSWIndex is not None
        self._index = NSWIndex(embedder.dim, 16, 200) if self._use_ann else FlatIndex(embedder.dim)
        self._chunks: dict[int, str] = {}
        self._sources: dict[int, str] = {}
        self._vectors: dict[int, np.ndarray] = {}
        self._next_id = 0
        self._bm25: BM25Okapi | None = None
        self._bm25_ids: list[int] = []
        # Neither the C++ index nor BM25 is safe to mutate while another
        # thread searches it; the web server ingests and chats concurrently.
        self._lock = threading.RLock()
        # Bumped on every mutation, so derived views (e.g. the ML knowledge
        # map) can cache themselves and know when they're stale.
        self.version = 0
        # Set by ``load`` when the persisted vectors had to be recomputed
        # (embedder changed, or a store saved by an older version); callers
        # should ``save`` again so it only happens once.
        self.reindexed = False

    def __len__(self) -> int:
        return len(self._chunks)

    def sources(self) -> set[str]:
        """Distinct ``source`` labels of everything currently in the store
        (e.g. file paths from `cognivore ingest`, or a demo-doc label from
        seeding). Lets a caller check "is this document already in here?"
        before re-adding it, so repeatable operations (like seeding demo
        content) can be made idempotent instead of duplicating chunks on
        every run.
        """
        return set(self._sources.values())

    def add_text(
        self,
        text: str,
        source: str,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
    ) -> list[int]:
        chunks: list[Chunk] = split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunks:
            return []
        # Embedding is the slow part and touches no shared state, so it
        # runs outside the lock.
        vectors = self.embedder.embed([c.text for c in chunks])
        ids: list[int] = []
        with self._lock:
            for chunk, vector in zip(chunks, vectors, strict=True):
                cid = self._next_id
                self._next_id += 1
                self._add_vector(cid, vector)
                self._chunks[cid] = chunk.text
                self._sources[cid] = source
                ids.append(cid)
            self._rebuild_bm25()
            self.version += 1
        return ids

    def _add_vector(self, cid: int, vector: list[float]) -> None:
        self._index.add(cid, vector)
        self._vectors[cid] = np.asarray(vector, dtype=np.float32)

    def vector_matrix(self) -> tuple[list[int], np.ndarray]:
        """All chunk ids and their embeddings as one ``(n, dim)`` float32
        matrix, in the same order -- the input to the ML insight layer."""
        with self._lock:
            ids = list(self._chunks.keys())
            if not ids:
                return [], np.zeros((0, self.embedder.dim), dtype=np.float32)
            return ids, np.stack([self._vectors[i] for i in ids])

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(self.embedder.embed([text])[0], dtype=np.float32)

    def get_chunk(self, cid: int) -> tuple[str, str] | None:
        """``(text, source)`` of one chunk, or ``None`` if it doesn't exist."""
        with self._lock:
            if cid not in self._chunks:
                return None
            return self._chunks[cid], self._sources[cid]

    def _rebuild_bm25(self) -> None:
        self._bm25_ids = list(self._chunks.keys())
        corpus = [index_terms(self._chunks[i]) for i in self._bm25_ids]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def search_multi(
        self, queries: list[str], top_k: int = 5, ef: int = 150
    ) -> list[RetrievedChunk]:
        """Runs :meth:`search` once per (non-empty, de-duplicated) query and
        merges the results, keeping each chunk's best score across queries.

        Exists for a concrete, observed failure mode: a small local model
        asked to search rewrites (often translates) the query it was given
        rather than reusing the user's own words, and the offline/lexical
        parts of hybrid search only match literal token overlap -- so a
        query in the "wrong" language can miss a document that plainly
        answers the question. Trying the user's original wording alongside
        whatever query the model chose is a cheap way to not depend on the
        model getting that right.
        """
        seen: dict[int, RetrievedChunk] = {}
        for query in dict.fromkeys(q for q in queries if q.strip()):
            for hit in self.search(query, top_k=top_k, ef=ef):
                existing = seen.get(hit.id)
                if existing is None or hit.score > existing.score:
                    seen[hit.id] = hit
        return sorted(seen.values(), key=lambda h: h.score, reverse=True)[:top_k]

    def search(self, query: str, top_k: int = 5, ef: int = 150) -> list[RetrievedChunk]:
        if not self._chunks:
            return []
        query_vec = self.embedder.embed([query])[0]
        with self._lock:
            return self._search_locked(query, query_vec, top_k, ef)

    def _search_locked(
        self, query: str, query_vec: list[float], top_k: int, ef: int
    ) -> list[RetrievedChunk]:
        if not self._chunks:
            return []
        if self._use_ann:
            vector_hits = self._index.search(query_vec, min(top_k * 4, len(self._chunks)), ef)
        else:
            vector_hits = self._index.search(query_vec, min(top_k * 4, len(self._chunks)))
        vector_scores = {h.id: float(h.score) for h in vector_hits}

        lexical_scores: dict[int, float] = {}
        if self._bm25 is not None:
            raw = self._bm25.get_scores(index_terms(query))
            max_score = max(raw, default=0.0) or 1.0
            lexical_scores = {
                cid: float(s / max_score) for cid, s in zip(self._bm25_ids, raw, strict=True)
            }

        candidate_ids = set(vector_scores) | set(lexical_scores)
        results: list[RetrievedChunk] = []
        for cid in candidate_ids:
            v = vector_scores.get(cid, 0.0)
            lex = lexical_scores.get(cid, 0.0)
            combined = self.vector_weight * v + (1 - self.vector_weight) * lex
            results.append(
                RetrievedChunk(
                    id=cid,
                    text=self._chunks[cid],
                    source=self._sources[cid],
                    score=combined,
                    vector_score=v,
                    lexical_score=lex,
                )
            )
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    # -- persistence -----------------------------------------------------

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with self._lock:
            ids = list(self._chunks.keys())
            (directory / "index.bin").write_bytes(self._index.serialize())
            matrix = (
                np.stack([self._vectors[i] for i in ids])
                if ids
                else np.zeros((0, self.embedder.dim), dtype=np.float32)
            )
            np.save(directory / "vectors.npy", matrix, allow_pickle=False)
            meta: dict[str, Any] = {
                "format": 2,
                "embedder": embedder_id(self.embedder),
                "dim": self.embedder.dim,
                "next_id": self._next_id,
                "use_ann": self._use_ann,
                "chunks": [
                    {"id": cid, "text": self._chunks[cid], "source": self._sources[cid]}
                    for cid in ids
                ],
            }
        (directory / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: str | Path, embedder: EmbeddingModel) -> DocumentStore:
        """Restores a store saved by :meth:`save`.

        If the store was embedded by a *different* embedder than the one
        passed in (or saved by an older version that didn't record it), the
        persisted vectors are meaningless for new queries -- so instead of
        returning a store that silently retrieves garbage, every chunk is
        re-embedded from its saved text and ``reindexed`` is set.
        """
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        store = cls(embedder=embedder, use_approximate_index=meta["use_ann"])
        items = meta["chunks"]
        for item in items:
            store._chunks[item["id"]] = item["text"]
            store._sources[item["id"]] = item["source"]
        store._next_id = meta["next_id"]

        vectors_path = directory / "vectors.npy"
        same_embedder = meta.get("embedder") == embedder_id(embedder)
        matrix: np.ndarray | None = None
        if same_embedder and vectors_path.exists():
            matrix = np.load(vectors_path, allow_pickle=False)
            if matrix.shape != (len(items), embedder.dim):
                matrix = None

        if matrix is not None:
            blob = (directory / "index.bin").read_bytes()
            store._index = (
                NSWIndex.deserialize(blob) if store._use_ann else FlatIndex.deserialize(blob)
            )
            for item, vec in zip(items, matrix, strict=True):
                store._vectors[item["id"]] = vec.astype(np.float32, copy=False)
        else:
            logger.info(
                "Knowledge base at %s was embedded with %r; re-embedding %d chunks with %r.",
                directory,
                meta.get("embedder", "an unrecorded embedder"),
                len(items),
                embedder_id(embedder),
            )
            ids = [item["id"] for item in items]
            vectors = embedder.embed([item["text"] for item in items]) if items else []
            for cid, vec in zip(ids, vectors, strict=True):
                store._add_vector(cid, vec)
            store.reindexed = True

        store._rebuild_bm25()
        store.version += 1
        return store

    def list_chunks(self) -> list[dict[str, Any]]:
        """Returns lightweight metadata for every stored chunk (used by the
        web UI's "sources" panel and by tests), without running a search."""
        return [
            asdict(
                RetrievedChunk(
                    id=cid,
                    text=text,
                    source=self._sources[cid],
                    score=0.0,
                    vector_score=0.0,
                    lexical_score=0.0,
                )
            )
            for cid, text in list(self._chunks.items())
        ]
