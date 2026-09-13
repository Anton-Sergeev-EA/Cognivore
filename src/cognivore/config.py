"""Central configuration, loaded from environment variables / a ``.env``
file via pydantic-settings. Every setting has a sane CPU-only, fully-local
default so the framework runs out of the box with no cloud credentials.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="COGNIVORE_", env_file=".env", extra="ignore")

    # -- Storage -----------------------------------------------------
    data_dir: Path = Field(default=Path("./.cognivore"), description="Local data directory")

    # -- LLM -----------------------------------------------------------
    llm_model_path: str | None = Field(
        default=None,
        description="Path to a local GGUF model for llama-cpp-python. If unset, the CLI/API "
        "fall back to a deterministic FakeLLMBackend useful for demos and tests.",
    )
    llm_context_length: int = Field(default=4096, ge=512)
    llm_n_threads: int | None = Field(default=None, description="CPU threads; None = auto")
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=512, ge=1)

    # -- Embeddings ------------------------------------------------------
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5")
    embedding_dim: int = Field(default=384, description="Must match embedding_model's output dim")
    prefer_semantic_embedder: bool = Field(
        default=True,
        description="Try the real fastembed model before falling back to HashingEmbedder. Set "
        "to False to skip the network attempt entirely (used by the test suite for speed).",
    )

    # -- RAG -------------------------------------------------------------
    chunk_size: int = Field(default=800, ge=50)
    chunk_overlap: int = Field(default=120, ge=0)
    retrieval_top_k: int = Field(default=5, ge=1)
    use_approximate_index: bool = Field(
        default=True, description="Use the C++ NSWIndex when available instead of exact FlatIndex"
    )

    # -- Agent -----------------------------------------------------------
    max_agent_steps: int = Field(default=6, ge=1)

    # -- Audio / Video ----------------------------------------------------
    whisper_model_size: str = Field(default="base")
    video_scene_threshold: float = Field(default=30.0, description="Frame-diff threshold, 0-255")
    video_max_keyframes: int = Field(default=24, ge=1)

    # -- Web server --------------------------------------------------------
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8420)

    def ensure_data_dir(self) -> Path:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir


def get_settings() -> Settings:
    """Returns a fresh ``Settings`` instance (re-reads env each call, which
    keeps tests that monkeypatch environment variables simple)."""
    return Settings()
