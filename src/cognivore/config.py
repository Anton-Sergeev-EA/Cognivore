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
    seed_demo_kb: bool = Field(
        default=False,
        description="On a fresh (empty) knowledge base, seed it automatically with the "
        "bundled demo company handbooks (one per UI language) so there's something to ask "
        "about immediately -- handy for a first run, a live demo, or a fresh container with "
        "no prior volume. Never overwrites or touches an already-populated knowledge base. "
        "docker-compose.yml enables this by default; set to false to start empty instead.",
    )

    # -- LLM -----------------------------------------------------------
    llm_provider: str = Field(
        default="auto",
        description="'auto' (try Ollama, then a GGUF model, then FakeLLMBackend), 'ollama', "
        "'llama_cpp', or 'fake'.",
    )
    llm_model_path: str | None = Field(
        default=None,
        description="Path to a local GGUF model for llama-cpp-python. Used when llm_provider "
        "is 'llama_cpp', or by 'auto' if Ollama isn't reachable.",
    )
    llm_context_length: int = Field(default=4096, ge=512)
    llm_n_threads: int | None = Field(default=None, description="CPU threads; None = auto")
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=512, ge=1)

    # Ollama (https://ollama.com) is the easiest way most people run a local
    # LLM -- if it's already installed and serving, 'auto' uses it with zero
    # extra download or config, which matters a lot more in practice than
    # which inference engine is "more native" to this project.
    ollama_host: str = Field(default="http://127.0.0.1:11434")
    ollama_model: str | None = Field(
        default=None,
        description="Model name to request from Ollama (e.g. 'qwen2.5:3b'). If unset, 'auto' "
        "asks Ollama for whatever models are already pulled and uses the first one.",
    )
    ollama_request_timeout: float = Field(
        default=600.0,
        description="Seconds to wait for an Ollama response. A several-billion-parameter model "
        "on CPU can genuinely take minutes for a full ReAct turn (system prompt + tool "
        "definitions + generation) -- this needs to be generous, not just cover network latency.",
    )

    # -- Embeddings ------------------------------------------------------
    embedding_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        description="fastembed model name. The default is multilingual (50+ languages, "
        "including Russian and Chinese, 384-dim, ~220 MB) because the UI and demo content ship "
        "in Russian, English and Chinese; an English-only model such as "
        "'BAAI/bge-small-en-v1.5' is smaller but maps non-English text poorly. Changing it "
        "re-embeds an existing knowledge base automatically on next start.",
    )
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

    # -- Explainability (cognivore.ml) -----------------------------------
    gap_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Retrieval confidence below which a knowledge-base question is recorded "
        "as a knowledge gap (shown in the web UI's gap radar).",
    )
    map_max_points: int = Field(
        default=1500,
        ge=10,
        description="Most chunks drawn on the knowledge map; larger stores are sampled.",
    )

    auto_context: bool = Field(
        default=True,
        description="Before each turn, search the knowledge base and hand relevant passages "
        "to the model up front, instead of relying on the model to call the search tool "
        "itself (small local models often don't). Not applied to the offline demo backend, "
        "which does its own routing.",
    )
    auto_context_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum retrieval confidence for auto_context to add passages; below "
        "it the model gets no passages (and may still search on its own).",
    )
    strict_knowledge_answers: bool = Field(
        default=True,
        description="When auto_context searched the knowledge base for a question and found "
        "nothing, answer 'the knowledge base has no information about this' (in the "
        "question's language) directly instead of letting the model answer from its general "
        "knowledge. Set to false to let the model answer anyway.",
    )
    auto_context_relative_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="auto_context only passes passages scoring at least this fraction of the "
        "best passage's score, so loosely related sections can't be blended into an answer.",
    )

    # -- Agent -----------------------------------------------------------
    max_agent_steps: int = Field(default=6, ge=1)

    # -- Audio / Video ----------------------------------------------------
    whisper_model_size: str = Field(
        default="small",
        description="faster-whisper model for speech: tiny, base, small, medium, large-v3. 'small' "
        "(~480 MB, downloaded on first use) is the smallest that transcribes the nine UI languages "
        "reliably; 'base' (~150 MB) is faster but often mishears non-English speech.",
    )
    video_scene_threshold: float = Field(
        default=30.0,
        description="How much (0-255) a pixel's brightness must change between frames to count as "
        "changed when looking for the scenes (slides, screens) of a video.",
    )
    video_max_keyframes: int = Field(
        default=200, ge=1, description="Most scenes taken from one video; the rest is skipped."
    )
    ocr_languages: str = Field(
        default="auto",
        description="Tesseract language packs for reading on-screen text in videos: 'auto' (every "
        "installed pack of the nine UI languages: eng, rus, chi_sim, spa, hin, fra, deu, jpn, "
        "ita) or a '+'-joined list such as 'eng+rus'. Each frame is read with the packs of its "
        "script (Latin, Cyrillic, Chinese, Japanese, Devanagari), so a video may switch language. "
        "Packs come from the system package manager (tesseract-ocr-<lang>); the Docker image "
        "ships all nine. A listed pack that isn't installed is skipped with a warning.",
    )

    # -- Web server --------------------------------------------------------
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8420)
    max_document_upload_mb: int = Field(
        default=20, ge=1, description="Largest text/markdown file accepted for ingestion."
    )
    max_media_upload_mb: int = Field(
        default=512, ge=1, description="Largest audio/video file accepted by the media tools."
    )

    def ensure_data_dir(self) -> Path:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir


def get_settings() -> Settings:
    """Returns a fresh ``Settings`` instance (re-reads env each call, which
    keeps tests that monkeypatch environment variables simple)."""
    return Settings()
