"""FastAPI app: a REST + SSE-streaming API for the agent, plus the static
chat UI. ``create_app()`` is a factory (rather than a module-level ``app``)
so tests can build a fresh, isolated instance per test with its own
in-memory store instead of sharing global state.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse
from starlette.responses import Response
from starlette.types import Scope

from cognivore import __version__
from cognivore.agent.core import AgentResult, TraceEntry
from cognivore.bootstrap import (
    build_agent,
    load_or_build_document_store,
    seed_demo_knowledge_base,
)
from cognivore.config import Settings, get_settings
from cognivore.index import is_native
from cognivore.llm.fake_backend import FakeLLMBackend
from cognivore.ml import KnowledgeGapTracker, KnowledgeMapBuilder, TurnInsight, build_turn_insight
from cognivore.ml.insight import SEARCH_TOOL
from cognivore.rag.store import DocumentStore, embedder_id
from cognivore.web.schemas import (
    ChatRequest,
    ChatResponse,
    ChunkOut,
    GapOut,
    HealthResponse,
    IngestResponse,
    IngestTextRequest,
    InsightOut,
    KnowledgeMapOut,
    TraceStepOut,
)

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"
_STORE_SUBDIR = "store"
_GAPS_FILE = "gaps.json"
_UPLOAD_CHUNK = 1024 * 1024


async def _read_limited(upload: UploadFile, limit_mb: int) -> bytes:
    """Reads an upload fully, but refuses (HTTP 413) as soon as it exceeds
    ``limit_mb`` -- instead of buffering an arbitrarily large body in RAM,
    which matters the moment the server is started with ``--host 0.0.0.0``.
    """
    limit = limit_mb * 1024 * 1024
    parts: list[bytes] = []
    size = 0
    while chunk := await upload.read(_UPLOAD_CHUNK):
        size += len(chunk)
        if size > limit:
            raise HTTPException(status_code=413, detail=f"file larger than {limit_mb} MB")
        parts.append(chunk)
    return b"".join(parts)


async def _save_limited(upload: UploadFile, limit_mb: int) -> str:
    """Streams an upload to a temporary file (same size cap as
    ``_read_limited``) and returns its path; the caller deletes it."""
    limit = limit_mb * 1024 * 1024
    size = 0
    with tempfile.NamedTemporaryFile(
        suffix=Path(upload.filename or "upload").suffix, delete=False
    ) as tmp:
        try:
            while chunk := await upload.read(_UPLOAD_CHUNK):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(status_code=413, detail=f"file larger than {limit_mb} MB")
                tmp.write(chunk)
        except HTTPException:
            tmp.close()
            Path(tmp.name).unlink(missing_ok=True)
            raise
        return tmp.name


def _trace_out(entry: TraceEntry) -> TraceStepOut:
    return TraceStepOut(
        thought=entry.thought,
        action=entry.action,
        action_input=entry.action_input,
        observation=entry.observation,
    )


def _insight_out(insight: TurnInsight) -> InsightOut:
    return InsightOut.model_validate(asdict(insight))


class _NoCacheStaticFiles(StaticFiles):
    """Plain ``StaticFiles`` sends no ``Cache-Control`` header at all, which
    leaves browsers free to apply their own heuristic caching -- and Chrome
    in particular can keep serving an old cached copy of e.g. ``app.js``
    for a long time after the file on disk (and its Last-Modified) has
    changed, with no error and nothing in the page to reveal it happened.
    Real-world impact of that: a UI change deployed to the server was
    invisible in an already-open browser even after a normal reload, and
    looked indistinguishable from the deploy having silently failed. Since
    this is a locally-run app rather than something served at CDN scale,
    trading away caching entirely is the right tradeoff -- a live demo
    should never be one browser-cache quirk away from showing stale UI.
    """

    async def get_response(self, path: str, scope: Scope) -> Response:
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store"
        return response


def _load_or_build_store(settings: Settings) -> DocumentStore:
    """Restores the knowledge base saved by a previous run, if any, so
    ingested documents survive a server restart (e.g. after changing
    ``.env`` to point at a real LLM). Falls back to a fresh store if
    nothing was saved yet, or if the saved store doesn't match the current
    embedder configuration.

    A *fresh* store (nothing was ever persisted at this data dir) is where
    ``COGNIVORE_SEED_DEMO_KB=true`` seeds the bundled demo documents --
    this only ever runs once per data dir: the seeded store is saved
    immediately below, so the next restart takes the "restore" path above
    instead and never re-seeds a knowledge base someone has since
    ingested their own documents into.
    """
    store_dir = settings.data_dir / _STORE_SUBDIR
    was_fresh = not (store_dir / "meta.json").exists()
    store = load_or_build_document_store(settings, store_dir)
    if was_fresh and settings.seed_demo_kb:
        added = seed_demo_knowledge_base(store, settings)
        if added:
            logger.info(
                "COGNIVORE_SEED_DEMO_KB=true: seeded a fresh knowledge base with %d demo chunks.",
                added,
            )
            try:
                store.save(store_dir)
            except OSError:
                logger.warning("Could not persist the seeded demo knowledge base.", exc_info=True)
    return store


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    settings.ensure_data_dir()

    app = FastAPI(title="Cognivore", version=__version__)

    store = _load_or_build_store(settings)
    agent = build_agent(settings, store=store, include_media=True)
    map_builder = KnowledgeMapBuilder(store, max_points=settings.map_max_points)
    gap_tracker = KnowledgeGapTracker(
        settings.data_dir / _GAPS_FILE, threshold=settings.gap_threshold
    )
    # One agent (and its conversation memory) serves every request, so
    # turns are serialized: two interleaved turns would corrupt each
    # other's conversation history. A local, single-user app is the
    # target; per-session agents are the path to multi-user.
    agent_lock = threading.Lock()

    app.state.settings = settings
    app.state.store = store
    app.state.agent = agent
    app.state.map_builder = map_builder
    app.state.gap_tracker = gap_tracker

    def _persist_store() -> None:
        try:
            store.save(settings.data_dir / _STORE_SUBDIR)
        except OSError:
            logger.warning("Failed to persist knowledge base.", exc_info=True)

    def _ingest(text: str, source: str) -> IngestResponse:
        ids = store.add_text(
            text,
            source=source,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        _persist_store()
        return IngestResponse(source=source, chunks_added=len(ids), total_chunks=len(store))

    def _turn_insight(question: str, result: AgentResult) -> InsightOut | None:
        """Explainability for one finished turn. Never allowed to break
        the chat itself: any failure is logged and the turn simply ships
        without an insight."""
        tool_queries = [
            str(t.action_input.get("query", "")) for t in result.trace if t.action == SEARCH_TOOL
        ]
        try:
            insight = build_turn_insight(
                store,
                question=question,
                answer=result.answer,
                tool_queries=tool_queries,
                used_knowledge_base=any(t.action == SEARCH_TOOL for t in result.trace),
                map_builder=map_builder,
                gap_tracker=gap_tracker,
                top_k=settings.retrieval_top_k,
            )
            return _insight_out(insight)
        except Exception:
            logger.warning("Could not compute the turn insight.", exc_info=True)
            return None

    @app.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            llm_backend=type(agent.llm).__name__,
            llm_model=getattr(agent.llm, "model", None),
            demo_mode=isinstance(agent.llm, FakeLLMBackend),
            embedder=embedder_id(store.embedder),
            native_index=is_native,
            tools=agent.tools.names(),
            knowledge_base_chunks=len(store),
        )

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="message must not be empty")

        def run() -> tuple[AgentResult, InsightOut | None]:
            with agent_lock:
                result = agent.run(request.message)
            return result, _turn_insight(request.message, result)

        result, insight = await asyncio.to_thread(run)
        return ChatResponse(
            answer=result.answer,
            trace=[_trace_out(t) for t in result.trace],
            hit_step_limit=result.hit_step_limit,
            insight=insight,
        )

    @app.get("/api/chat/stream")
    async def chat_stream(message: str) -> EventSourceResponse:
        if not message.strip():
            raise HTTPException(status_code=400, detail="message must not be empty")

        async def event_generator():
            # agent.run_stream() is a plain synchronous generator (the LLM
            # backends it drives are blocking HTTP/subprocess calls), so it
            # runs on a worker thread; each event it yields is handed back
            # to this coroutine through a queue as soon as it's produced,
            # which is what lets the final answer reach the browser
            # token-by-token as the model generates it instead of only
            # after the whole turn finishes.
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue = asyncio.Queue()
            _SENTINEL = object()

            def emit(kind: str, payload: Any) -> None:
                loop.call_soon_threadsafe(queue.put_nowait, (kind, payload))

            def worker() -> None:
                final: AgentResult | None = None
                try:
                    with agent_lock:
                        for kind, payload in agent.run_stream(message):
                            if kind == "final" and isinstance(payload, AgentResult):
                                final = payload
                            emit(kind, payload)
                    # Computed after the answer has fully streamed, so the
                    # explainability layer never delays the answer itself.
                    if final is not None:
                        insight = _turn_insight(message, final)
                        if insight is not None:
                            emit("insight", insight)
                except Exception as exc:  # pragma: no cover - surfaced to the client below
                    emit("error", str(exc))
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

            threading.Thread(target=worker, daemon=True).start()

            # Normally the final answer arrives entirely as "answer_delta"
            # events, live, as the model generates it. But a real model
            # doesn't always use the literal "Final Answer:" marker
            # (sometimes it just answers directly, or after a malformed
            # tool-call attempt) -- run_stream still recognizes that as the
            # final answer via AgentResult, but nothing gets streamed live
            # for it since no marker was ever seen. Without this fallback
            # that answer would silently never reach the client at all.
            got_any_delta = False

            while True:
                item = await queue.get()
                if item is _SENTINEL:
                    break
                kind, payload = item
                if kind == "trace":
                    yield {"event": "trace", "data": _trace_out(payload).model_dump_json()}
                elif kind == "answer_delta":
                    got_any_delta = True
                    yield {"event": "answer_chunk", "data": payload}
                elif kind == "final":
                    if not got_any_delta and payload.answer:
                        yield {"event": "answer_chunk", "data": payload.answer}
                    # The exact final text, so the client can map the
                    # grounding report's sentence offsets onto it.
                    yield {"event": "answer_final", "data": payload.answer}
                elif kind == "insight":
                    yield {"event": "insight", "data": payload.model_dump_json()}
                elif kind == "error":
                    yield {"event": "answer_chunk", "data": f"(error: {payload})"}
            yield {"event": "done", "data": ""}

        return EventSourceResponse(event_generator())

    @app.post("/api/ingest/text", response_model=IngestResponse)
    async def ingest_text(request: IngestTextRequest) -> IngestResponse:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="text must not be empty")
        return await asyncio.to_thread(_ingest, request.text, request.source)

    @app.post("/api/ingest/file", response_model=IngestResponse)
    async def ingest_file(file: UploadFile = File(...)) -> IngestResponse:
        raw = await _read_limited(file, settings.max_document_upload_mb)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(
                status_code=400, detail="only UTF-8 text/markdown files are supported"
            ) from exc
        # Embedding a document can take seconds with a real model; running
        # it on the event loop would freeze every other request meanwhile.
        return await asyncio.to_thread(_ingest, text, file.filename or "upload")

    @app.get("/api/knowledge_base", response_model=list[ChunkOut])
    async def knowledge_base() -> list[ChunkOut]:
        return [
            ChunkOut(id=c["id"], text=c["text"], source=c["source"]) for c in store.list_chunks()
        ]

    @app.get("/api/knowledge_base/{chunk_id}", response_model=ChunkOut)
    async def knowledge_base_chunk(chunk_id: int) -> ChunkOut:
        chunk = store.get_chunk(chunk_id)
        if chunk is None:
            raise HTTPException(status_code=404, detail="no such chunk")
        return ChunkOut(id=chunk_id, text=chunk[0], source=chunk[1])

    @app.get("/api/insights/map", response_model=KnowledgeMapOut)
    async def insights_map() -> KnowledgeMapOut:
        kmap = await asyncio.to_thread(map_builder.get)
        return KnowledgeMapOut(
            points=[asdict(p) for p in kmap.points],  # type: ignore[misc]
            clusters=[asdict(c) for c in kmap.clusters],  # type: ignore[misc]
            total_chunks=kmap.total_chunks,
            silhouette=kmap.silhouette,
            explained_variance=kmap.explained_variance,
        )

    @app.get("/api/insights/gaps", response_model=list[GapOut])
    async def insights_gaps(limit: int = 20) -> list[GapOut]:
        return [GapOut(**asdict(g)) for g in gap_tracker.items(limit=max(1, min(limit, 200)))]

    @app.delete("/api/insights/gaps")
    async def resolve_gap(query: str) -> dict:
        if not gap_tracker.resolve(query):
            raise HTTPException(status_code=404, detail="no such gap")
        return {"resolved": query}

    @app.post("/api/media/audio")
    async def media_audio(file: UploadFile = File(...)) -> dict:
        tool = agent.tools.get("transcribe_audio")
        if tool is None:
            raise HTTPException(
                status_code=503, detail="audio tool unavailable (install the `audio` extra)"
            )
        tmp_path = await _save_limited(file, settings.max_media_upload_mb)
        try:
            transcript = await asyncio.to_thread(tool.run, audio_path=tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        return {"transcript": transcript}

    @app.post("/api/media/video")
    async def media_video(file: UploadFile = File(...)) -> dict:
        tool = agent.tools.get("analyze_video")
        if tool is None:
            raise HTTPException(
                status_code=503, detail="video tool unavailable (install the `video` extra)"
            )
        tmp_path = await _save_limited(file, settings.max_media_upload_mb)
        try:
            analysis = await asyncio.to_thread(tool.run, video_path=tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        return {"analysis": analysis}

    if _STATIC_DIR.exists():
        app.mount("/", _NoCacheStaticFiles(directory=_STATIC_DIR, html=True), name="static")

    return app
