"""L2 Semantic cache: similarity-threshold lookup against stored embeddings."""

from __future__ import annotations

from typing import Optional

from embedx.index.fallback_index import FallbackIndex
from embedx.storage.sqlite_store import Record, SQLiteStore
from embedx.utils.helpers import cosine_similarity


class SemanticCache:
    """
    Returns an existing record when a query embedding is within
    `threshold` cosine similarity of a stored embedding.
    """

    def __init__(
        self,
        store: SQLiteStore,
        index: FallbackIndex,
        threshold: float = 0.92,
    ) -> None:
        self._store = store
        self._index = index
        self._threshold = threshold

    def get(self, query_embedding: list[float]) -> Optional[Record]:
        results = self._index.search(query_embedding, top_k=1)
        if not results:
            return None
        record, score = results[0]
        if score >= self._threshold:
            self._store.touch(record.id)
            self._store.increment_stat("semantic_hits")
            return record
        return None

    @property
    def threshold(self) -> float:
        return self._threshold
