"""The knowledge map: every chunk of the knowledge base as a point on a 2-D
plane (PCA of its embedding), grouped into automatically discovered topics
(k-means++ with k picked by silhouette), each topic named by its c-TF-IDF
keywords. Rebuilt lazily, only when the store actually changed."""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field

import numpy as np

from cognivore.ml.clustering import auto_kmeans
from cognivore.ml.projection import PCAProjection, fit_pca_2d
from cognivore.ml.topics import cluster_keywords
from cognivore.rag.store import DocumentStore

_MARKDOWN_NOISE_RE = re.compile(r"[#*_`>|]+")
_WS_RE = re.compile(r"\s+")


def preview(text: str, limit: int = 160) -> str:
    clean = _WS_RE.sub(" ", _MARKDOWN_NOISE_RE.sub(" ", text)).strip()
    return clean if len(clean) <= limit else clean[: limit - 1].rstrip() + "…"


@dataclass
class MapPoint:
    id: int
    x: float
    y: float
    cluster: int
    source: str
    preview: str


@dataclass
class MapCluster:
    id: int
    keywords: list[str]
    size: int
    x: float
    y: float


@dataclass
class KnowledgeMap:
    points: list[MapPoint] = field(default_factory=list)
    clusters: list[MapCluster] = field(default_factory=list)
    total_chunks: int = 0
    silhouette: float = 0.0
    explained_variance: tuple[float, float] = (0.0, 0.0)
    version: int = -1
    projection: PCAProjection | None = None

    def project(self, vector: np.ndarray) -> tuple[float, float] | None:
        if self.projection is None or not self.points:
            return None
        v = np.asarray(vector, dtype=np.float64)
        norm = float(np.linalg.norm(v))
        if norm > 1e-12:
            v = v / norm
        x, y = self.projection.transform(v)[0]
        return float(x), float(y)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms > 1e-12, norms, 1.0)


def build_knowledge_map(
    store: DocumentStore, max_points: int = 1500, seed: int = 42
) -> KnowledgeMap:
    version = store.version
    ids, matrix = store.vector_matrix()
    total = len(ids)
    if total == 0:
        return KnowledgeMap(version=version)

    # Large stores are mapped from a deterministic sample: the picture
    # (and the topics) are statistically the same, the browser stays fast.
    if total > max_points:
        keep = np.sort(np.random.default_rng(seed).choice(total, max_points, replace=False))
        ids = [ids[i] for i in keep]
        matrix = matrix[keep]

    x = _normalize_rows(matrix.astype(np.float64))
    projection = fit_pca_2d(x)
    coords = projection.transform(x)
    # With a multilingual embedder, the same content in nine languages sits
    # together, and silhouette alone tends to settle on 2 very broad topics
    # for a knowledge base of a hundred-plus chunks; asking for at least 3
    # topics there keeps the map informative.
    k_min = 3 if len(ids) >= 40 else 2
    result, silhouette = auto_kmeans(x, k_min=k_min, seed=seed)
    labels = [int(label) for label in result.labels]

    texts: list[str] = []
    sources: list[str] = []
    for cid in ids:
        chunk = store.get_chunk(cid)
        text, source = chunk if chunk is not None else ("", "")
        texts.append(text)
        sources.append(source)
    keywords = cluster_keywords(texts, labels)

    points = [
        MapPoint(cid, round(float(cx), 4), round(float(cy), 4), label, src, preview(text))
        for cid, (cx, cy), label, src, text in zip(ids, coords, labels, sources, texts, strict=True)
    ]
    clusters: list[MapCluster] = []
    for label in sorted(set(labels)):
        member = np.array(labels) == label
        centre = coords[member].mean(0)
        clusters.append(
            MapCluster(
                id=label,
                keywords=keywords.get(label, []),
                size=int(member.sum()),
                x=round(float(centre[0]), 4),
                y=round(float(centre[1]), 4),
            )
        )
    return KnowledgeMap(
        points=points,
        clusters=clusters,
        total_chunks=total,
        silhouette=round(silhouette, 4),
        explained_variance=(
            round(projection.explained_variance_ratio[0], 4),
            round(projection.explained_variance_ratio[1], 4),
        ),
        version=version,
        projection=projection,
    )


class KnowledgeMapBuilder:
    """Caches the map per ``store.version`` (thread-safe)."""

    def __init__(self, store: DocumentStore, max_points: int = 1500, seed: int = 42) -> None:
        self.store = store
        self.max_points = max_points
        self.seed = seed
        self._lock = threading.Lock()
        self._cached: KnowledgeMap | None = None

    def get(self) -> KnowledgeMap:
        with self._lock:
            if self._cached is None or self._cached.version != self.store.version:
                self._cached = build_knowledge_map(self.store, self.max_points, self.seed)
            return self._cached
