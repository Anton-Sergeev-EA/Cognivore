"""Wires together settings -> embedder -> document store -> LLM backend ->
tool registry -> agent. Both the CLI and the web API build their ``Agent``
through this single module so the two entry points can never drift apart.
"""

from __future__ import annotations

import importlib.util
import logging

from cognivore.agent.core import Agent
from cognivore.agent.memory import ConversationBuffer, VectorMemory
from cognivore.config import Settings
from cognivore.llm.base import GenerationConfig, LLMBackend
from cognivore.llm.fake_backend import FakeLLMBackend
from cognivore.rag.embeddings import get_default_embedder
from cognivore.rag.store import DocumentStore
from cognivore.tools.base import ToolRegistry
from cognivore.tools.calculator import CalculatorTool
from cognivore.tools.rag_search import RagSearchTool

logger = logging.getLogger(__name__)


def _build_ollama_backend(settings: Settings) -> LLMBackend | None:
    from cognivore.llm.ollama_backend import (
        OllamaBackend,
        OllamaUnavailableError,
        is_ollama_running,
    )

    if not is_ollama_running(settings.ollama_host):
        return None
    try:
        backend = OllamaBackend(host=settings.ollama_host, model=settings.ollama_model)
    except OllamaUnavailableError as exc:
        logger.info("Ollama is running but unusable (%s); trying the next backend.", exc)
        return None
    logger.info("Using Ollama at %s with model '%s'.", settings.ollama_host, backend.model)
    return backend


def _build_llama_cpp_backend(settings: Settings) -> LLMBackend | None:
    if not settings.llm_model_path:
        return None
    from cognivore.llm.llama_cpp_backend import LlamaCppBackend

    logger.info("Loading local LLM from %s", settings.llm_model_path)
    return LlamaCppBackend(
        model_path=settings.llm_model_path,
        n_ctx=settings.llm_context_length,
        n_threads=settings.llm_n_threads,
    )


def build_llm_backend(settings: Settings) -> LLMBackend:
    provider = settings.llm_provider.lower()

    if provider == "fake":
        return FakeLLMBackend()

    if provider == "ollama":
        backend = _build_ollama_backend(settings)
        if backend is not None:
            return backend
        logger.warning(
            "COGNIVORE_LLM_PROVIDER=ollama but no Ollama server was reachable at %s; falling "
            "back to FakeLLMBackend.",
            settings.ollama_host,
        )
        return FakeLLMBackend()

    if provider == "llama_cpp":
        backend = _build_llama_cpp_backend(settings)
        if backend is not None:
            return backend
        logger.warning(
            "COGNIVORE_LLM_PROVIDER=llama_cpp but COGNIVORE_LLM_MODEL_PATH is not set; falling "
            "back to FakeLLMBackend."
        )
        return FakeLLMBackend()

    # "auto" (the default): prefer whatever needs the least setup from the
    # user. A running Ollama server means real local models with zero
    # download managed by this project; a configured GGUF path is the next
    # best thing; otherwise fall back to the deterministic offline backend
    # rather than failing to start.
    for builder in (_build_ollama_backend, _build_llama_cpp_backend):
        backend = builder(settings)
        if backend is not None:
            return backend
    logger.warning(
        "No local LLM found (no Ollama server running, no COGNIVORE_LLM_MODEL_PATH set); using "
        "FakeLLMBackend (deterministic offline demo mode). Run `ollama serve` with a model "
        "pulled, or point COGNIVORE_LLM_MODEL_PATH at a GGUF file, for real responses."
    )
    return FakeLLMBackend()


def build_document_store(settings: Settings) -> DocumentStore:
    embedder = get_default_embedder(
        settings.embedding_model,
        settings.embedding_dim,
        prefer_fastembed=settings.prefer_semantic_embedder,
    )
    return DocumentStore(embedder=embedder, use_approximate_index=settings.use_approximate_index)


def _is_importable(module_name: str) -> bool:
    """Checks a module can be imported without actually importing it (and,
    for these two, without triggering a model download or a slow native-lib
    load). Used to decide whether the audio/video tools are *really*
    available rather than just whether their thin wrapper module happens to
    import cleanly -- ``cognivore.tools.audio_transcribe`` imports fine even
    without ``faster-whisper`` installed, since that import is deferred
    until the tool actually runs (see ``cognivore.media.audio``), so a bare
    try/import of the tool module can't tell registered-but-broken apart
    from genuinely-usable.
    """
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def build_tool_registry(
    store: DocumentStore, settings: Settings, include_media: bool = False
) -> ToolRegistry:
    tools = ToolRegistry([CalculatorTool(), RagSearchTool(store)])
    if include_media:
        if _is_importable("faster_whisper"):
            from cognivore.tools.audio_transcribe import AudioTranscribeTool

            tools.register(AudioTranscribeTool(model_size=settings.whisper_model_size))
        else:
            logger.info("Audio tool unavailable (install the `audio` extra to enable it).")

        if _is_importable("cv2"):
            from cognivore.tools.video_analyze import VideoAnalyzeTool

            tools.register(
                VideoAnalyzeTool(
                    scene_threshold=settings.video_scene_threshold,
                    max_keyframes=settings.video_max_keyframes,
                )
            )
        else:
            logger.info("Video tool unavailable (install the `video` extra to enable it).")
    return tools


def build_agent(
    settings: Settings,
    store: DocumentStore | None = None,
    llm: LLMBackend | None = None,
    include_media: bool = True,
    use_vector_memory: bool = True,
) -> Agent:
    store = store or build_document_store(settings)
    llm = llm or build_llm_backend(settings)
    tools = build_tool_registry(store, settings, include_media=include_media)
    vector_memory = (
        VectorMemory(store.embedder, settings.use_approximate_index) if use_vector_memory else None
    )
    return Agent(
        llm=llm,
        tools=tools,
        max_steps=settings.max_agent_steps,
        generation_config=GenerationConfig(
            max_tokens=settings.llm_max_tokens, temperature=settings.llm_temperature
        ),
        conversation=ConversationBuffer(),
        vector_memory=vector_memory,
    )
