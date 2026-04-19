"""L1 Exact cache: hash-based O(1) lookup."""

from __future__ import annotations

from typing import Optional

from embedx.storage.sqlite_store import Record, SQLiteStore
from embedx.utils.helpers import hash_text


class ExactCache:
    """Wraps SQLite hash lookup for zero-compute cache hits."""

    def __init__(self, store: SQLiteStore) -> None:
        self._store = store

    def get(self, text: str) -> Optional[Record]:
        h = hash_text(text)
        record = self._store.get_by_hash(h)
        if record:
            self._store.touch(record.id)
            self._store.increment_stat("exact_hits")
        return record

    def put(
        self,
        text: str,
        embedding: list[float],
        metadata: Optional[dict] = None,
        namespace: str = "default",
        importance: float = 0.5,
    ) -> int:
        h = hash_text(text)
        return self._store.upsert(text, h, embedding, metadata, namespace=namespace, importance=importance)
