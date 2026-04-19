"""Tests for L1 exact cache and L2 semantic cache."""

from __future__ import annotations

from embedx.cache.exact_cache import ExactCache
from embedx.utils.helpers import hash_text


class TestExactCache:
    def test_miss_returns_none(self, store):
        cache = ExactCache(store)
        assert cache.get("hello world") is None

    def test_put_and_get(self, store):
        cache = ExactCache(store)
        vec = [0.1, 0.2, 0.3]
        cache.put("hello world", vec)
        record = cache.get("hello world")
        assert record is not None
        assert record.text == "hello world"

    def test_normalization(self, store):
        """Different whitespace / casing should not match (hash is normalized)."""
        cache = ExactCache(store)
        vec = [0.1, 0.2, 0.3]
        cache.put("Hello World", vec)
        # Same after normalization: "hello world"
        record = cache.get("hello world")
        assert record is not None

    def test_deduplication(self, store):
        cache = ExactCache(store)
        vec = [0.1, 0.2, 0.3]
        id1 = cache.put("text", vec)
        id2 = cache.put("text", vec)
        assert id1 == id2

    def test_increments_use_count(self, store):
        cache = ExactCache(store)
        cache.put("hello", [0.1, 0.2])
        cache.get("hello")
        cache.get("hello")
        record = cache.get("hello")
        assert record.use_count >= 2
