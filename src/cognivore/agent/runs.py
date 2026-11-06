"""Persistent state for document comparison runs."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID, uuid4


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
