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
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse
from starlette.responses import Response
from starlette.types import Scope

from cognivore.bootstrap import (
    build_agent,
    load_or_build_document_store,
    seed_demo_knowledge_base,
)
from cognivore.config import Settings, get_settings
from cognivore.index import is_native
from cognivore.rag.store import DocumentStore
from cognivore.web.schemas import (
    ChatRequest,
    ChatResponse,
    ChunkOut,
    HealthResponse,
    IngestResponse,
    TraceStepOut,
)

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"
_STORE_SUBDIR = "store"


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

    app = FastAPI(title="Cognivore", version="0.1.0")

    store = _load_or_build_store(settings)
    agent = build_agent(settings, store=store, include_media=True)

    app.state.settings = settings
    app.state.store = store
    app.state.agent = agent

    def _persist_store() -> None:
        try:
            store.save(settings.data_dir / _STORE_SUBDIR)
        except OSError:
            logger.warning("Failed to persist knowledge base.", exc_info=True)

    @app.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            llm_backend=type(agent.llm).__name__,
            llm_model=getattr(agent.llm, "model", None),
            native_index=is_native,
            tools=agent.tools.names(),
            knowledge_base_chunks=len(store),
        )

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="message must not be empty")
        result = await asyncio.to_thread(agent.run, request.message)
        return ChatResponse(
            answer=result.answer,
            trace=[
                TraceStepOut(
                    thought=t.thought,
                    action=t.action,
                    action_input=t.action_input,
                    observation=t.observation,
                )
                for t in result.trace
            ],
            hit_step_limit=result.hit_step_limit,
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

            def worker() -> None:
                try:
                    for kind, payload in agent.run_stream(message):
                        loop.call_soon_threadsafe(queue.put_nowait, (kind, payload))
                except Exception as exc:  # pragma: no cover - surfaced to the client below
                    loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))
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
                    yield {
                        "event": "trace",
                        "data": TraceStepOut(
                            thought=payload.thought,
                            action=payload.action,
                            action_input=payload.action_input,
                            observation=payload.observation,
                        ).model_dump_json(),
                    }
                elif kind == "answer_delta":
                    got_any_delta = True
                    yield {"event": "answer_chunk", "data": payload}
                elif kind == "final":
                    if not got_any_delta and payload.answer:
                        yield {"event": "answer_chunk", "data": payload.answer}
                elif kind == "error":
                    yield {"event": "answer_chunk", "data": f"(error: {payload})"}
            yield {"event": "done", "data": ""}

        return EventSourceResponse(event_generator())

    @app.post("/api/ingest/text", response_model=IngestResponse)
    async def ingest_text(source: str, text: str) -> IngestResponse:
        ids = store.add_text(
            text,
            source=source,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        _persist_store()
        return IngestResponse(source=source, chunks_added=len(ids), total_chunks=len(store))

    @app.post("/api/ingest/file", response_model=IngestResponse)
    async def ingest_file(file: UploadFile = File(...)) -> IngestResponse:
        raw = await file.read()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(
                status_code=400, detail="only UTF-8 text/markdown files are supported"
            ) from exc
        ids = store.add_text(
            text,
            source=file.filename or "upload",
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        _persist_store()
        return IngestResponse(
            source=file.filename or "upload", chunks_added=len(ids), total_chunks=len(store)
        )

    @app.get("/api/knowledge_base", response_model=list[ChunkOut])
    async def knowledge_base() -> list[ChunkOut]:
        return [
            ChunkOut(id=c["id"], text=c["text"], source=c["source"]) for c in store.list_chunks()
        ]

    @app.post("/api/media/audio")
    async def media_audio(file: UploadFile = File(...)) -> dict:
        tool = agent.tools.get("transcribe_audio")
        if tool is None:
            raise HTTPException(
                status_code=503, detail="audio tool unavailable (install the `audio` extra)"
            )
        with tempfile.NamedTemporaryFile(
            suffix=Path(file.filename or "audio").suffix, delete=False
        ) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
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
        with tempfile.NamedTemporaryFile(
            suffix=Path(file.filename or "video").suffix, delete=False
        ) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        try:
            analysis = await asyncio.to_thread(tool.run, video_path=tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        return {"analysis": analysis}

    if _STATIC_DIR.exists():
        app.mount("/", _NoCacheStaticFiles(directory=_STATIC_DIR, html=True), name="static")

    return app
