"""Persistent state for document comparison runs."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4


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
