"""Browser specification review API and SQLite persistence."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel
from starlette.responses import PlainTextResponse

from cognivore.rag.embeddings import EmbeddingModel
from cognivore.review import Finding, compare_citations
from cognivore.review_documents import (
    MAX_UPLOAD_BYTES,
    DocumentSupportUnavailable,
    ReviewDocument,
    read_review_document,
)

Decision = Literal["accepted", "rejected", "pending"]


class DecisionInput(BaseModel):
    decision: Decision


class Review(BaseModel):
    id: UUID
    before_name: str
    after_name: str
    findings: list[Finding]
    decisions: list[Decision]
    published: bool = False
    warnings: list[str] = []

    def report(self) -> str:
        lines = [
            "# Specification review",
            "",
            f"    {self.before_name} → {self.after_name}",
            "",
        ]
        if self.warnings:
            lines.extend(["## Import notes", "", *(f"    {note}" for note in self.warnings), ""])
        for finding, decision in zip(self.findings, self.decisions, strict=True):
            if decision != "accepted":
                continue
            lines.extend([f"## {finding.kind.title()}", ""])
            if finding.signals:
                lines.extend([", ".join(finding.signals), ""])
            citations = [("Before", finding.before), ("After", finding.after)]
            citations.extend(("Alternative", candidate) for candidate in finding.alternatives)
            for side, citation in citations:
                if citation:
                    location = f"{citation.source}:{citation.line}"
                    if citation.page is not None:
                        location = f"{citation.source}, page {citation.page}"
                    elif citation.paragraph is not None:
                        location = f"{citation.source}, paragraph {citation.paragraph}"
                    lines.extend(
                        [
                            f"    {side} | {location}",
                            *([f"    {citation.section}"] if citation.section else []),
                            f"    {citation.text}",
                            "",
                        ]
                    )
        if "accepted" not in self.decisions:
            lines.extend(["No findings accepted.", ""])
        return "\n".join(lines)


class ReviewStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        with self.connection() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS specification_reviews ("
                "id TEXT PRIMARY KEY, state TEXT NOT NULL, report TEXT)"
            )

    @contextmanager
    def connection(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        with closing(sqlite3.connect(self.path)) as db, db:
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db

    def create(self, review: Review) -> None:
        with self.connection(write=True) as db:
            db.execute(
                "INSERT INTO specification_reviews (id, state) VALUES (?, ?)",
                (review.id.hex, review.model_dump_json()),
            )

    def load(self, db: sqlite3.Connection, review_id: UUID) -> tuple[Review, str | None]:
        row = db.execute(
            "SELECT state, report FROM specification_reviews WHERE id = ?", (review_id.hex,)
        ).fetchone()
        if row is None:
            raise HTTPException(404, "Review not found.")
        return Review.model_validate_json(row[0]), row[1]


def _read(file: UploadFile) -> tuple[str, ReviewDocument]:
    name = Path((file.filename or "document").replace("\\", "/")).name
    name = " ".join(name.split())[:160] or "document"
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Each file must be at most 8 MiB.")
    try:
        return name, read_review_document(name, data)
    except DocumentSupportUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


def review_router(path: Path, embedder: EmbeddingModel) -> APIRouter:
    router = APIRouter(prefix="/api/reviews")
    store = ReviewStore(path)

    @router.post("", status_code=201)
    def create(before: UploadFile, after: UploadFile) -> Review:
        before_name, old_document = _read(before)
        after_name, new_document = _read(after)
        findings = compare_citations(old_document.citations, new_document.citations, embedder)
        review = Review(
            id=uuid4(),
            before_name=before_name,
            after_name=after_name,
            findings=findings,
            decisions=["pending"] * len(findings),
            warnings=old_document.warnings + new_document.warnings,
        )
        store.create(review)
        return review

    @router.get("/{review_id}")
    def get(review_id: UUID) -> Review:
        with store.connection() as db:
            review, _ = store.load(db, review_id)
        return review

    @router.put("/{review_id}/findings/{index}")
    def decide(review_id: UUID, index: int, choice: DecisionInput) -> Review:
        with store.connection(write=True) as db:
            review, _ = store.load(db, review_id)
            if review.published:
                raise HTTPException(409, "This review has already been published.")
            if not 0 <= index < len(review.decisions):
                raise HTTPException(404, "Finding not found.")
            review.decisions[index] = choice.decision
            db.execute(
                "UPDATE specification_reviews SET state = ? WHERE id = ?",
                (review.model_dump_json(), review.id.hex),
            )
        return review

    @router.post("/{review_id}/publish")
    def publish(review_id: UUID) -> Review:
        with store.connection(write=True) as db:
            review, existing_report = store.load(db, review_id)
            if existing_report is not None:
                return review
            if "pending" in review.decisions:
                raise HTTPException(409, "Decide all findings before publishing.")
            review.published = True
            db.execute(
                "UPDATE specification_reviews SET state = ?, report = ? WHERE id = ?",
                (review.model_dump_json(), review.report(), review.id.hex),
            )
        return review

    @router.get("/{review_id}/report")
    def download(review_id: UUID) -> PlainTextResponse:
        with store.connection() as db:
            _, report = store.load(db, review_id)
        if report is None:
            raise HTTPException(409, "Publish the review before downloading the report.")
        return PlainTextResponse(
            report,
            headers={"Content-Disposition": f'attachment; filename="review-{review_id.hex}.md"'},
        )

    return router
