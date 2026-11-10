"""Persistent agent checkpoints and staged reports."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID, uuid4

from pydantic import TypeAdapter

from cognivore.agent.core import Agent, AgentCheckpoint

_CHECKPOINT = TypeAdapter(AgentCheckpoint)


@dataclass(frozen=True)
class ComparisonRun:
    run_id: str
    report: str
    status: str


class RunStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        with closing(sqlite3.connect(path)) as db, db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS comparison_runs (
                    id TEXT PRIMARY KEY,
                    report TEXT NOT NULL,
                    status TEXT NOT NULL
                )
            """)
            db.execute("""
                CREATE TABLE IF NOT EXISTS agent_tasks (
                    id TEXT PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    checkpoint BLOB,
                    revision INTEGER NOT NULL DEFAULT 0
                )
            """)

    def create_task(self, prompt: str) -> str:
        run_id = uuid4().hex
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("INSERT INTO agent_tasks (id, prompt) VALUES (?, ?)", (run_id, prompt))
        return run_id

    def resume_task(self, run_id: str, agent: Agent) -> ComparisonRun:
        with closing(sqlite3.connect(self.path)) as db:
            row = db.execute(
                "SELECT task.prompt, task.checkpoint, task.revision, report.report, report.status "
                "FROM agent_tasks AS task LEFT JOIN comparison_runs AS report ON task.id = report.id "
                "WHERE task.id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            raise ValueError(f"Unknown agent run: {run_id}")
        prompt, payload, revision, report, status = row
        if status is not None:
            return ComparisonRun(run_id, report, status)

        def save(state: AgentCheckpoint) -> None:
            nonlocal revision
            with closing(sqlite3.connect(self.path)) as db, db:
                updated = db.execute(
                    "UPDATE agent_tasks SET checkpoint = ?, revision = revision + 1 "
                    "WHERE id = ? AND revision = ?",
                    (_CHECKPOINT.dump_json(state), run_id, revision),
                )
                if updated.rowcount != 1:
                    raise RuntimeError("Another process advanced this run. Reopen it to continue.")
            revision += 1

        checkpoint = _CHECKPOINT.validate_json(payload) if payload is not None else None
        result = agent.run(prompt, checkpoint=checkpoint, save=save)
        with closing(sqlite3.connect(self.path)) as db, db:
            updated = db.execute(
                "UPDATE agent_tasks SET revision = revision + 1 WHERE id = ? AND revision = ?",
                (run_id, revision),
            )
            if updated.rowcount != 1:
                raise RuntimeError("Another process advanced this run. Reopen it to continue.")
            db.execute(
                "INSERT INTO comparison_runs (id, report, status) VALUES (?, ?, ?)",
                (run_id, result.answer, "pending"),
            )
        return ComparisonRun(run_id, result.answer, "pending")

    def create(self, report: str) -> ComparisonRun:
        run = ComparisonRun(uuid4().hex, report, "pending")
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute(
                "INSERT INTO comparison_runs (id, report, status) VALUES (?, ?, ?)",
                (run.run_id, run.report, run.status),
            )
        return run

    def get(self, run_id: str) -> ComparisonRun:
        with closing(sqlite3.connect(self.path)) as db:
            row = db.execute(
                "SELECT id, report, status FROM comparison_runs WHERE id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise ValueError(f"Unknown comparison run: {run_id}")
        return ComparisonRun(*row)

    def approve(self, run_id: str) -> Path:
        canonical_id = UUID(run_id).hex
        if canonical_id != run_id:
            raise ValueError("Expected a canonical comparison run ID.")
        output_dir = self.path.parent / "reports"
        output = output_dir / f"{run_id}.md"
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT report, status FROM comparison_runs WHERE id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"Unknown comparison run: {run_id}")
            report, status = row
            if status == "approved":
                return output

            output_dir.mkdir(exist_ok=True)
            # Publish the complete file without replacing an existing report.
            with TemporaryDirectory(dir=output_dir) as staging:
                draft = Path(staging) / "report"
                draft.write_text(report, encoding="utf-8")
                try:
                    output.hardlink_to(draft)
                except FileExistsError:
                    # Recover if publication succeeded before the transaction committed.
                    if output.is_symlink() or output.read_bytes() != draft.read_bytes():
                        raise
            db.execute("UPDATE comparison_runs SET status = ? WHERE id = ?", ("approved", run_id))
        return output
