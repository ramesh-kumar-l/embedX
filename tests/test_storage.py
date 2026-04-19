"""Tests for SQLite storage layer."""

from __future__ import annotations

from embedx.storage.sqlite_store import SQLiteStore
from embedx.utils.helpers import hash_text


class TestSQLiteStore:
    def test_upsert_and_retrieve(self, store):
        text = "hello storage"
        h = hash_text(text)
        vec = [0.1, 0.2, 0.3, 0.4]
        store.upsert(text, h, vec, {"key": "val"})
        record = store.get_by_hash(h)
        assert record is not None
        assert record.text == text
        assert len(record.embedding) == 4

    def test_upsert_idempotent(self, store):
        text = "same text"
        h = hash_text(text)
        store.upsert(text, h, [0.1])
        store.upsert(text, h, [0.1])
        assert store.count() == 1

    def test_all_records(self, store):
        for i in range(5):
            store.upsert(f"text {i}", hash_text(f"text {i}"), [float(i)])
        assert len(store.all_records()) == 5

    def test_delete(self, store):
        h = hash_text("delete me")
        store.upsert("delete me", h, [0.1])
        assert store.delete_by_hash(h)
        assert store.get_by_hash(h) is None

    def test_stats(self, store):
        store.increment_stat("hits", 5)
        store.increment_stat("hits", 3)
        assert store.get_stat("hits") == 8.0

    def test_clear(self, store):
        store.upsert("doc", hash_text("doc"), [0.1])
        store.clear()
        assert store.count() == 0

    def test_metadata_roundtrip(self, store):
        meta = {"nested": {"a": 1}, "list": [1, 2, 3]}
        h = hash_text("meta test")
        store.upsert("meta test", h, [0.1], meta)
        record = store.get_by_hash(h)
        assert record.metadata == meta
