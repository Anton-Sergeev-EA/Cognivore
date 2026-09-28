from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str


class TraceStepOut(BaseModel):
    thought: str
    action: str | None
    action_input: dict
    observation: str | None


class InsightHitOut(BaseModel):
    rank: int
    id: int
    source: str
    preview: str
    score: float
    vector_score: float
    lexical_score: float


class SentenceSupportOut(BaseModel):
    start: int
    end: int
    support: float
    source_rank: int | None
    citation: bool = False


class GroundingOut(BaseModel):
    score: float
    level: str
    sentences: list[SentenceSupportOut]


class InsightOut(BaseModel):
    """Explainability data for one chat turn (see ``cognivore.ml``)."""

    language: str
    used_knowledge_base: bool
    queries: list[str]
    hits: list[InsightHitOut]
    query_point: tuple[float, float] | None
    confidence: float
    grounding: GroundingOut | None
    gap: bool


class ChatResponse(BaseModel):
    answer: str
    trace: list[TraceStepOut]
    hit_step_limit: bool
    insight: InsightOut | None = None


class IngestTextRequest(BaseModel):
    source: str = Field(min_length=1, max_length=300)
    text: str


class IngestResponse(BaseModel):
    source: str
    chunks_added: int
    total_chunks: int


class ChunkOut(BaseModel):
    id: int
    text: str
    source: str


class HealthResponse(BaseModel):
    status: str
    llm_backend: str
    llm_model: str | None = None
    demo_mode: bool = False
    embedder: str = ""
    native_index: bool
    tools: list[str]
    knowledge_base_chunks: int


class MapPointOut(BaseModel):
    id: int
    x: float
    y: float
    cluster: int
    source: str
    preview: str


class MapClusterOut(BaseModel):
    id: int
    keywords: list[str]
    size: int
    x: float
    y: float


class KnowledgeMapOut(BaseModel):
    points: list[MapPointOut]
    clusters: list[MapClusterOut]
    total_chunks: int
    silhouette: float
    explained_variance: tuple[float, float]
    method: str = "PCA(2) · k-means++ (k by silhouette) · c-TF-IDF"


class GapOut(BaseModel):
    query: str
    confidence: float
    count: int
    last_seen: float
    language: str
