from __future__ import annotations

from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str


class TraceStepOut(BaseModel):
    thought: str
    action: str | None
    action_input: dict
    observation: str | None


class ChatResponse(BaseModel):
    answer: str
    trace: list[TraceStepOut]
    hit_step_limit: bool


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
    native_index: bool
    tools: list[str]
    knowledge_base_chunks: int
