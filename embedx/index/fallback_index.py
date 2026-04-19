"""Pure-Python cosine similarity index (no external dependencies)."""

from __future__ import annotations

import numpy as np

from embedx.storage.sqlite_store import Record, SQLiteStore
from embedx.utils.helpers import cosine_similarity


class FallbackIndex:
    """
    In-memory index backed by SQLite.
    Rebuilds from DB on initialization or explicit rebuild.
    Linear scan — adequate for up to ~50k records.
    """

    def __init__(self, store: SQLiteStore) -> None:
        self._store = store
        self._records: list[Record] = []
        self._matrix: np.ndarray | None = None
        self._dirty = True

    def rebuild(self) -> None:
        self._records = self._store.all_records()
        if self._records:
            self._matrix = np.array(
                [r.embedding for r in self._records], dtype=np.float32
            )
        else:
            self._matrix = None
        self._dirty = False

    def add(self, record: Record) -> None:
        self._records.append(record)
        vec = np.array(record.embedding, dtype=np.float32).reshape(1, -1)
        if self._matrix is None:
            self._matrix = vec
        else:
            self._matrix = np.vstack([self._matrix, vec])

    def mark_dirty(self) -> None:
        self._dirty = True

    def search(
        self, query: list[float], top_k: int = 5
    ) -> list[tuple[Record, float]]:
        if self._dirty:
            self.rebuild()
        if self._matrix is None or len(self._records) == 0:
            return []

        qvec = np.array(query, dtype=np.float32)
        norm_q = np.linalg.norm(qvec)
        if norm_q == 0.0:
            return []

        norms = np.linalg.norm(self._matrix, axis=1)
        norms[norms == 0] = 1e-10
        scores = (self._matrix @ qvec) / (norms * norm_q)

        k = min(top_k, len(self._records))
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        return [(self._records[i], float(scores[i])) for i in top_idx]

    def size(self) -> int:
        return len(self._records)
