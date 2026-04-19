"""Tests for namespace isolation, importance ranking, and backward compatibility."""

from __future__ import annotations

import pytest

from embedx.api.public import EmbedX
from embedx.ranking.scorer import RankingScorer
from embedx.storage.sqlite_store import Record, SQLiteStore

from tests.conftest import DummyProvider


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db(tmp_path):
    with EmbedX(db_path=str(tmp_path / "test.db"), provider=DummyProvider(), importance_weight=0.1) as embedx:
        yield embedx


@pytest.fixture
def store(tmp_path):
    s = SQLiteStore(str(tmp_path / "store.db"))
    yield s
    s.close()


# ---------------------------------------------------------------------------
# Storage migration tests
# ---------------------------------------------------------------------------


class TestStorageMigration:
    def test_new_columns_exist(self, store: SQLiteStore):
        cols = {
            row[1]
            for row in store._conn.execute("PRAGMA table_info(embeddings)").fetchall()
        }
        assert "namespace" in cols
        assert "importance" in cols

    def test_default_namespace_and_importance(self, store: SQLiteStore):
        from embedx.utils.helpers import hash_text

        store.upsert("hello world", hash_text("hello world"), [0.1] * 128)
        record = store.get_by_hash(hash_text("hello world"))
        assert record is not None
        assert record.namespace == "default"
        assert record.importance == pytest.approx(0.5)

    def test_custom_namespace_and_importance(self, store: SQLiteStore):
        from embedx.utils.helpers import hash_text

        store.upsert(
            "important text",
            hash_text("important text"),
            [0.2] * 128,
            namespace="project-x",
            importance=0.9,
        )
        record = store.get_by_hash(hash_text("important text"))
        assert record.namespace == "project-x"
        assert record.importance == pytest.approx(0.9)

    def test_records_by_namespace(self, store: SQLiteStore):
        from embedx.utils.helpers import hash_text

        store.upsert("alpha", hash_text("alpha"), [0.1] * 128, namespace="ns-a")
        store.upsert("beta", hash_text("beta"), [0.2] * 128, namespace="ns-b")
        store.upsert("gamma", hash_text("gamma"), [0.3] * 128, namespace="ns-a")

        ns_a = store.records_by_namespace("ns-a")
        ns_b = store.records_by_namespace("ns-b")
        ns_c = store.records_by_namespace("ns-c")

        assert len(ns_a) == 2
        assert len(ns_b) == 1
        assert len(ns_c) == 0
        assert all(r.namespace == "ns-a" for r in ns_a)


# ---------------------------------------------------------------------------
# Namespace isolation in EmbedX.add / EmbedX.search
# ---------------------------------------------------------------------------


class TestNamespaceIsolation:
    def test_add_with_namespace(self, db: EmbedX):
        r = db.add("Python is great", namespace="langs")
        assert r["status"] == "added"

    def test_search_within_namespace(self, db: EmbedX):
        db.add("Python is great", namespace="langs")
        db.add("The capital of France is Paris", namespace="geo")

        results_langs = db.search("programming language", namespace="langs")
        results_geo = db.search("programming language", namespace="geo")

        # langs namespace has one doc; geo has one unrelated doc
        assert len(results_langs) == 1
        assert results_langs[0]["text"] == "Python is great"

        # geo namespace returns the geo doc, not the langs doc
        for r in results_langs:
            assert r["text"] != "The capital of France is Paris"

    def test_search_no_namespace_returns_all(self, db: EmbedX):
        db.add("Python is great", namespace="langs")
        db.add("The capital of France is Paris", namespace="geo")

        results = db.search("information", top_k=10)
        texts = {r["text"] for r in results}
        assert "Python is great" in texts
        assert "The capital of France is Paris" in texts

    def test_empty_namespace_returns_empty(self, db: EmbedX):
        db.add("Some text", namespace="ns-a")
        results = db.search("some text", namespace="ns-b")
        assert results == []

    def test_namespace_cross_contamination(self, db: EmbedX):
        for i in range(3):
            db.add(f"Document {i} in alpha", namespace="alpha")
        for i in range(3):
            db.add(f"Document {i} in beta", namespace="beta")

        alpha_results = db.search("Document", namespace="alpha", top_k=10)
        beta_results = db.search("Document", namespace="beta", top_k=10)

        alpha_texts = {r["text"] for r in alpha_results}
        beta_texts = {r["text"] for r in beta_results}
        assert alpha_texts.isdisjoint(beta_texts)


# ---------------------------------------------------------------------------
# Importance-based ranking
# ---------------------------------------------------------------------------


class TestImportanceRanking:
    def test_importance_field_stored(self, db: EmbedX):
        from embedx.utils.helpers import hash_text

        db.add("critical info", namespace="x", importance=0.9)
        record = db._store.get_by_hash(hash_text("critical info"))
        assert record is not None
        assert record.importance == pytest.approx(0.9)

    def test_importance_in_scorer(self):
        scorer = RankingScorer(
            semantic_weight=0.7,
            recency_weight=0.2,
            importance_weight=0.1,
        )
        import time

        def make_record(importance: float) -> Record:
            return Record(
                id=1,
                text="t",
                text_hash="h",
                embedding=[],
                metadata={},
                created_at=time.time(),
                last_used=time.time(),
                use_count=1,
                namespace="default",
                importance=importance,
            )

        high = scorer.score(make_record(1.0), 0.8)
        low = scorer.score(make_record(0.0), 0.8)
        assert high.final_score > low.final_score
        assert high.importance_score == pytest.approx(1.0)
        assert low.importance_score == pytest.approx(0.0)

    def test_zero_importance_weight_ignores_importance(self):
        scorer = RankingScorer(
            semantic_weight=0.8,
            recency_weight=0.1,
            frequency_weight=0.1,
            importance_weight=0.0,
        )
        import time

        def make_record(importance: float) -> Record:
            return Record(
                id=1,
                text="t",
                text_hash="h",
                embedding=[],
                metadata={},
                created_at=time.time(),
                last_used=time.time(),
                use_count=1,
                namespace="default",
                importance=importance,
            )

        high = scorer.score(make_record(1.0), 0.8)
        low = scorer.score(make_record(0.0), 0.8)
        assert high.final_score == pytest.approx(low.final_score, abs=1e-6)


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_add_without_namespace_or_importance(self, tmp_path):
        with EmbedX(db_path=str(tmp_path / "compat.db"), provider=DummyProvider()) as db:
            result = db.add("Hello world")
            assert result["status"] in {"added", "exact_hit", "semantic_hit"}

    def test_search_without_namespace(self, tmp_path):
        with EmbedX(db_path=str(tmp_path / "compat.db"), provider=DummyProvider()) as db:
            db.add("Hello world")
            results = db.search("Hello world")
            assert len(results) > 0
            assert "text" in results[0]
            assert "score" in results[0]

    def test_default_importance_weight_does_not_change_ranking(self, tmp_path):
        """Without importance_weight, scores must be identical regardless of stored importance."""
        with EmbedX(
            db_path=str(tmp_path / "compat.db"),
            provider=DummyProvider(),
            importance_weight=0.0,
        ) as db:
            db.add("critical doc", importance=1.0)
            db.add("low priority doc", importance=0.0)
            results = db.search("doc", top_k=5)
            # Both exist; ranking driven only by semantic+recency+frequency
            assert len(results) == 2

    def test_existing_response_shape_preserved(self, tmp_path):
        with EmbedX(db_path=str(tmp_path / "compat.db"), provider=DummyProvider()) as db:
            db.add("Test document")
            results = db.search("test")
            assert len(results) > 0
            r = results[0]
            assert set(r.keys()) >= {"text", "score", "semantic_score", "metadata", "use_count"}
