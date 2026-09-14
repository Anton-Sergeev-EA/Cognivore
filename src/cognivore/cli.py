"""Command-line entry point: ``cognivore chat|ingest|serve|bench``."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
import uvicorn
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from cognivore.bootstrap import (
    build_agent,
    load_or_build_document_store,
    seed_demo_knowledge_base,
)
from cognivore.config import get_settings

app = typer.Typer(add_completion=False, help="Cognivore: a local-first multimodal agent.")
console = Console()


@app.command()
def chat(
    show_trace: bool = typer.Option(
        False, "--trace", help="Print the agent's Thought/Action steps."
    ),
) -> None:
    """Starts an interactive chat session with the agent (Ctrl-D / Ctrl-C to exit)."""
    settings = get_settings()
    settings.ensure_data_dir()
    agent = build_agent(settings)
    console.print(
        Panel.fit(
            "Cognivore agent ready. Type a message and press Enter.\n"
            f"LLM backend: {type(agent.llm).__name__} | tools: {', '.join(agent.tools.names())}",
            title="cognivore chat",
        )
    )
    while True:
        try:
            user_input = console.input("[bold cyan]you>[/bold cyan] ")
        except (EOFError, KeyboardInterrupt):
            console.print("\nbye!")
            break
        if not user_input.strip():
            continue
        result = agent.run(user_input)
        if show_trace:
            for i, entry in enumerate(result.trace, start=1):
                console.print(f"[dim]step {i} thought: {entry.thought}[/dim]")
                if entry.action:
                    console.print(
                        f"[dim]  action: {entry.action}({entry.action_input}) -> {entry.observation}[/dim]"
                    )
        console.print(Panel(Markdown(result.answer), title="cognivore", border_style="green"))


@app.command()
def ingest(
    path: Path = typer.Argument(..., help="File or directory of text/markdown files to ingest."),
    store_dir: Path = typer.Option(Path("./.cognivore/store"), help="Where to persist the store."),
) -> None:
    """Ingests text files into the local RAG store."""
    settings = get_settings()
    store = load_or_build_document_store(settings, store_dir)
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    total_chunks = 0
    for file_path in files:
        if file_path.suffix.lower() not in {".txt", ".md", ".markdown", ".rst"}:
            continue
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        ids = store.add_text(
            text,
            source=str(file_path),
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        total_chunks += len(ids)
        console.print(f"[green]+[/green] {file_path} -> {len(ids)} chunks")
    store.save(store_dir)
    console.print(f"Ingested {total_chunks} chunks from {len(files)} file(s) into {store_dir}")


@app.command("seed-demo")
def seed_demo(
    store_dir: Path = typer.Option(Path("./.cognivore/store"), help="Where to persist the store."),
) -> None:
    """Seeds the local knowledge base with the bundled demo company
    handbooks (English + Russian) so there's something to search and ask
    about immediately -- handy for a first run or a live demo without
    ``COGNIVORE_SEED_DEMO_KB`` (that env var does the same thing
    automatically for `serve`, but only on an empty knowledge base;
    this command is for adding the demo docs to an *existing* store on
    demand)."""
    settings = get_settings()
    store = load_or_build_document_store(settings, store_dir)
    added = seed_demo_knowledge_base(store, settings)
    store.save(store_dir)
    console.print(f"[green]+[/green] seeded {added} chunks into {store_dir}")


@app.command()
def serve(
    host: str | None = typer.Option(None),
    port: int | None = typer.Option(None),
    reload: bool = typer.Option(False),
) -> None:
    """Starts the FastAPI web server (chat UI + REST/streaming API)."""
    settings = get_settings()
    uvicorn.run(
        "cognivore.web.app:create_app",
        factory=True,
        host=host or settings.host,
        port=port or settings.port,
        reload=reload,
    )


@app.command()
def bench(
    n: int = typer.Option(3000, help="Number of random vectors to index."),
    dim: int = typer.Option(384, help="Vector dimensionality."),
    queries: int = typer.Option(200, help="Number of queries to time."),
    scaling: bool = typer.Option(
        False, "--scaling", help="Also run the search-latency-vs-size scaling benchmark."
    ),
) -> None:
    """Benchmarks the native C++ index against the pure-NumPy fallback."""
    from cognivore.benchmark import run_benchmark, run_scaling_benchmark

    run_benchmark(n=n, dim=dim, queries=queries, console=console)
    if scaling:
        run_scaling_benchmark(dim=dim, console=console)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    app()


if __name__ == "__main__":
    main()
