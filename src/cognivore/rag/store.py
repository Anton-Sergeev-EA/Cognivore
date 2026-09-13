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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from cognivore.index import FlatIndex, NSWIndex, is_native
from cognivore.rag.chunking import Chunk, split_text
from cognivore.rag.embeddings import EmbeddingModel


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
        self._next_id = 0
        self._bm25: BM25Okapi | None = None
        self._bm25_ids: list[int] = []

    def __len__(self) -> int:
        return len(self._chunks)

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
        vectors = self.embedder.embed([c.text for c in chunks])
        ids: list[int] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            cid = self._next_id
            self._next_id += 1
            self._index.add(cid, vector)
            self._chunks[cid] = chunk.text
            self._sources[cid] = source
            ids.append(cid)
        self._rebuild_bm25()
        return ids

    def _rebuild_bm25(self) -> None:
        self._bm25_ids = list(self._chunks.keys())
        corpus = [self._chunks[i].lower().split() for i in self._bm25_ids]
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
        if self._use_ann:
            vector_hits = self._index.search(query_vec, min(top_k * 4, len(self._chunks)), ef)
        else:
            vector_hits = self._index.search(query_vec, min(top_k * 4, len(self._chunks)))
        vector_scores = {h.id: float(h.score) for h in vector_hits}

        lexical_scores: dict[int, float] = {}
        if self._bm25 is not None:
            raw = self._bm25.get_scores(query.lower().split())
            max_score = max(raw, default=0.0) or 1.0
            lexical_scores = {
                cid: float(s) / max_score for cid, s in zip(self._bm25_ids, raw, strict=True)
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
        (directory / "index.bin").write_bytes(self._index.serialize())
        meta: dict[str, Any] = {
            "next_id": self._next_id,
            "use_ann": self._use_ann,
            "chunks": [
                {"id": cid, "text": text, "source": self._sources[cid]}
                for cid, text in self._chunks.items()
            ],
        }
        (directory / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, directory: str | Path, embedder: EmbeddingModel) -> DocumentStore:
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text())
        store = cls(embedder=embedder, use_approximate_index=meta["use_ann"])
        blob = (directory / "index.bin").read_bytes()
        if store._use_ann:
            store._index = NSWIndex.deserialize(blob)
        else:
            store._index = FlatIndex.deserialize(blob)
        for item in meta["chunks"]:
            store._chunks[item["id"]] = item["text"]
            store._sources[item["id"]] = item["source"]
        store._next_id = meta["next_id"]
        store._rebuild_bm25()
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
            for cid, text in self._chunks.items()
        ]
