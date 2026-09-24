"""Knowledge-gap radar: remembers questions the knowledge base could not
answer well, merging near-duplicate phrasings, so the owner sees *what to
add* ranked by how often people asked for it."""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from cognivore.rag.tokenize import content_tokens

logger = logging.getLogger(__name__)


@dataclass
class GapRecord:
    query: str
    confidence: float
    count: int
    last_seen: float
    language: str = "en"


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class KnowledgeGapTracker:
    def __init__(
        self,
        path: Path | None = None,
        threshold: float = 0.3,
        max_items: int = 200,
        merge_similarity: float = 0.6,
    ) -> None:
        self.path = path
        self.threshold = threshold
        self.max_items = max_items
        self.merge_similarity = merge_similarity
        self._lock = threading.Lock()
        self._records: list[GapRecord] = self._load()

    def _load(self) -> list[GapRecord]:
        if self.path is None or not self.path.exists():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return [GapRecord(**item) for item in raw]
        except (OSError, ValueError, TypeError):
            logger.warning("Ignoring unreadable knowledge-gap file %s.", self.path)
            return []

    def _save(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = [asdict(r) for r in self._records]
            self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), "utf-8")
        except OSError:
            logger.warning("Could not persist knowledge gaps to %s.", self.path, exc_info=True)

    def observe(self, query: str, confidence: float, language: str = "en") -> bool:
        """Records ``query`` if ``confidence`` is below the threshold.
        Returns whether it was treated as a gap."""
        query = query.strip()
        if not query or confidence >= self.threshold:
            return False
        tokens = set(content_tokens(query))
        with self._lock:
            for record in self._records:
                same = record.query.lower() == query.lower()
                if same or _jaccard(tokens, set(content_tokens(record.query))) >= (
                    self.merge_similarity
                ):
                    record.count += 1
                    record.last_seen = time.time()
                    record.confidence = min(record.confidence, confidence)
                    break
            else:
                self._records.append(
                    GapRecord(query, round(confidence, 4), 1, time.time(), language)
                )
                if len(self._records) > self.max_items:
                    self._records.sort(key=lambda r: (r.count, r.last_seen))
                    del self._records[: len(self._records) - self.max_items]
            self._save()
        return True

    def items(self, limit: int = 20) -> list[GapRecord]:
        with self._lock:
            ranked = sorted(self._records, key=lambda r: (-r.count, -r.last_seen))
            return [GapRecord(**asdict(r)) for r in ranked[:limit]]

    def resolve(self, query: str) -> bool:
        """Drops a gap (e.g. after the owner uploaded the missing document)."""
        with self._lock:
            before = len(self._records)
            self._records = [r for r in self._records if r.query != query]
            changed = len(self._records) != before
            if changed:
                self._save()
            return changed
