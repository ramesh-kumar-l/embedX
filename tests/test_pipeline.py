"""Integration tests for the full EmbedX pipeline."""

from __future__ import annotations

import pytest

from tests.conftest import DummyProvider


class TestPipeline:
    def test_add_and_search(self, db):
        db.add("The quick brown fox jumps over the lazy dog")
        db.add("Machine learning is a subset of artificial intelligence")
        results = db.search("fox jumps", top_k=2)
        assert len(results) > 0
        assert all("text" in r and "score" in r for r in results)

    def test_exact_cache_hit(self, db):
        db.add("hello world")
        result = db.add("hello world")
        assert result["status"] == "exact_hit"

    def test_search_returns_scores(self, db):
        db.add("Python is a programming language")
        results = db.search("programming")
        assert results[0]["score"] > 0

    def test_add_batch(self, db):
        texts = ["text one", "text two", "text three"]
        results = db.add_batch(texts)
        assert len(results) == 3
        assert all(r["status"] in ("added", "exact_hit") for r in results)

    def test_get_stats(self, db):
        db.add("sample text")
        db.search("sample")
        stats = db.get_stats()
        assert "document_count" in stats
        assert "hit_rate" in stats
        assert "latency" in stats

    def test_clear(self, db):
        db.add("some document")
        db.clear()
        assert db._store.count() == 0

    def test_context_manager(self, tmp_db):
        from embedx.api.public import EmbedX

        with EmbedX(db_path=tmp_db, provider=DummyProvider()) as db:
            db.add("test")
            results = db.search("test")
            assert len(results) > 0

    def test_empty_search(self, db):
        results = db.search("query when empty")
        assert results == []

    def test_search_top_k(self, db):
        for i in range(10):
            db.add(f"document number {i} about various topics")
        results = db.search("document", top_k=3)
        assert len(results) <= 3

    def test_metadata_stored(self, db):
        db.add("tagged document", metadata={"tag": "important", "priority": 1})
        results = db.search("tagged document", top_k=1)
        assert results[0]["metadata"].get("tag") == "important"

    def test_rebuild_index(self, db):
        db.add("document before rebuild")
        db.rebuild_index()
        results = db.search("document")
        assert len(results) > 0
