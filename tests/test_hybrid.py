"""Tests for hybrid retrieval: BM25, fusion, reranker, and end-to-end."""

from __future__ import annotations

import math

import pytest

from embedx.api.public import EmbedX
from embedx.retrieval.bm25 import BM25Index, _tokenize
from embedx.retrieval.fusion import fuse
from embedx.retrieval.reranker import rerank

from tests.conftest import DummyProvider


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------


def test_bm25_tokenize() -> None:
    assert _tokenize("Hello, World!") == ["hello", "world"]
    assert _tokenize("foo_bar baz") == ["foo_bar", "baz"]
    assert _tokenize("") == []


def test_bm25_returns_empty_on_no_docs(tmp_db: str) -> None:
    from embedx.storage.sqlite_store import SQLiteStore

    store = SQLiteStore(tmp_db)
    idx = BM25Index(store)
    results = idx.search("anything", top_k=5)
    assert results == []
    store.close()


def test_bm25_basic_ranking(tmp_db: str) -> None:
    with EmbedX(db_path=tmp_db, provider=DummyProvider()) as db:
        db.add("transformers use self attention mechanisms")
        db.add("recurrent neural networks process sequences")
        db.add("transformers replaced recurrent networks in most NLP tasks")

        from embedx.retrieval.bm25 import BM25Index

        idx = BM25Index(db._store)
        results = idx.search("transformers", top_k=5)

    # Both docs containing "transformers" should score above 0; doc with
    # more occurrences/better BM25 score should rank first.
    assert len(results) >= 2
    texts = [r.text for r, _ in results]
    assert any("transformers" in t.lower() for t in texts)
    # Scores are positive
    assert all(sc > 0 for _, sc in results)
    # Sorted descending
    scores = [sc for _, sc in results]
    assert scores == sorted(scores, reverse=True)


def test_bm25_no_match_returns_empty(tmp_db: str) -> None:
    with EmbedX(db_path=tmp_db, provider=DummyProvider()) as db:
        db.add("dogs are loyal pets")
        from embedx.retrieval.bm25 import BM25Index

        idx = BM25Index(db._store)
        results = idx.search("quantum physics", top_k=5)

    assert results == []


def test_bm25_top_k_respected(tmp_db: str) -> None:
    with EmbedX(db_path=tmp_db, provider=DummyProvider()) as db:
        for i in range(10):
            db.add(f"machine learning document number {i}")

        from embedx.retrieval.bm25 import BM25Index

        idx = BM25Index(db._store)
        results = idx.search("machine learning", top_k=3)

    assert len(results) <= 3


def test_bm25_mark_dirty_triggers_rebuild(tmp_db: str) -> None:
    with EmbedX(db_path=tmp_db, provider=DummyProvider()) as db:
        db.add("cats are independent animals")

        from embedx.retrieval.bm25 import BM25Index

        idx = BM25Index(db._store)
        _ = idx.search("cats", top_k=5)
        assert not idx._dirty

        idx.mark_dirty()
        assert idx._dirty

        _ = idx.search("cats", top_k=5)
        assert not idx._dirty


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


def _make_record(record_id: int, text: str):
    """Build a minimal Record-like object for fusion tests."""
    from dataclasses import dataclass

    @dataclass
    class FakeRecord:
        id: int
        text: str

    return FakeRecord(id=record_id, text=text)


def test_fusion_semantic_only() -> None:
    r1 = _make_record(1, "doc one")
    r2 = _make_record(2, "doc two")
    sem = [(r1, 0.9), (r2, 0.5)]
    result = fuse(sem, [], w_semantic=0.7, w_keyword=0.3, top_k=5)
    assert len(result) == 2
    assert result[0][0].id == 1  # higher semantic score first


def test_fusion_keyword_only() -> None:
    r1 = _make_record(1, "doc one")
    r2 = _make_record(2, "doc two")
    kw = [(r1, 10.0), (r2, 2.0)]
    result = fuse([], kw, w_semantic=0.7, w_keyword=0.3, top_k=5)
    assert len(result) == 2
    assert result[0][0].id == 1


def test_fusion_combines_both_lists() -> None:
    r1 = _make_record(1, "doc one")
    r2 = _make_record(2, "doc two")
    r3 = _make_record(3, "doc three")

    # r1 strong semantic, r3 strong keyword, r2 weak both
    sem = [(r1, 0.9), (r2, 0.3)]
    kw = [(r3, 8.0), (r2, 1.0)]

    result = fuse(sem, kw, w_semantic=0.5, w_keyword=0.5, top_k=5)

    ids = [r.id for r, _ in result]
    assert set(ids) == {1, 2, 3}


def test_fusion_scores_deterministic() -> None:
    r1 = _make_record(1, "doc one")
    r2 = _make_record(2, "doc two")
    sem = [(r1, 0.8), (r2, 0.4)]
    kw = [(r1, 5.0), (r2, 2.5)]

    a = fuse(sem, kw, top_k=5)
    b = fuse(sem, kw, top_k=5)
    assert [r.id for r, _ in a] == [r.id for r, _ in b]
    assert [sc for _, sc in a] == [sc for _, sc in b]


def test_fusion_top_k_respected() -> None:
    records = [_make_record(i, f"doc {i}") for i in range(10)]
    sem = [(r, float(i) / 10) for i, r in enumerate(records)]
    kw = [(r, float(i)) for i, r in enumerate(records)]
    result = fuse(sem, kw, top_k=3)
    assert len(result) == 3


def test_fusion_empty_inputs() -> None:
    assert fuse([], [], top_k=5) == []


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------


def test_reranker_exact_phrase_bonus() -> None:
    r1 = _make_record(1, "machine learning is a subset of AI")
    r2 = _make_record(2, "deep learning uses neural networks")

    # Both have same base score; r1 contains exact phrase
    results = [(r1, 0.5), (r2, 0.5)]
    reranked = rerank("machine learning", results)
    assert reranked[0][0].id == 1


def test_reranker_preserves_order_when_no_bonus() -> None:
    r1 = _make_record(1, "completely unrelated text about cats")
    r2 = _make_record(2, "another unrelated text about dogs")
    results = [(r1, 0.8), (r2, 0.3)]
    reranked = rerank("quantum physics", results)
    # r1 had higher base score and neither triggers bonuses meaningfully
    assert reranked[0][0].id == 1


def test_reranker_empty_input() -> None:
    assert rerank("query", []) == []


def test_reranker_sorted_descending() -> None:
    records = [_make_record(i, f"sample document number {i}") for i in range(5)]
    results = [(r, float(i) / 5) for i, r in enumerate(records)]
    reranked = rerank("sample", results)
    scores = [sc for _, sc in reranked]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# End-to-end hybrid search
# ---------------------------------------------------------------------------


def test_hybrid_search_returns_results(tmp_db: str) -> None:
    with EmbedX(db_path=tmp_db, provider=DummyProvider()) as db:
        db.add("Python is a high-level programming language")
        db.add("Java is a statically typed language")
        db.add("Rust provides memory safety without garbage collection")

        results = db.search("programming language", top_k=3, hybrid=True)

    assert len(results) > 0
    assert all("text" in r for r in results)
    assert all("score" in r for r in results)
    assert all("metadata" in r for r in results)


def test_hybrid_search_response_format_unchanged(tmp_db: str) -> None:
    """Hybrid results must have the same shape as semantic-only results."""
    with EmbedX(db_path=tmp_db, provider=DummyProvider()) as db:
        db.add("semantic search uses vector embeddings")
        db.add("keyword search uses inverted indexes")

        semantic = db.search("search", top_k=2, hybrid=False)
        hybrid = db.search("search", top_k=2, hybrid=True)

    for r in hybrid:
        assert set(r.keys()) == set(semantic[0].keys())


def test_hybrid_instance_flag(tmp_db: str) -> None:
    """EmbedX(hybrid=True) enables hybrid for all searches by default."""
    with EmbedX(db_path=tmp_db, provider=DummyProvider(), hybrid=True) as db:
        db.add("neural networks learn representations")
        db.add("gradient descent optimizes model parameters")

        results = db.search("neural learning", top_k=2)

    assert len(results) > 0


def test_hybrid_top_k_respected(tmp_db: str) -> None:
    with EmbedX(db_path=tmp_db, provider=DummyProvider()) as db:
        for i in range(10):
            db.add(f"document about topic number {i}")

        results = db.search("document topic", top_k=3, hybrid=True)

    assert len(results) <= 3


def test_hybrid_default_unchanged(tmp_db: str) -> None:
    """Calling search() without hybrid=True must use the existing code path."""
    with EmbedX(db_path=tmp_db, provider=DummyProvider()) as db:
        db.add("this is a test document")
        results = db.search("test", top_k=1)
        # _hybrid_retriever should not be created when hybrid was never requested
        assert db._hybrid_retriever is None

    assert len(results) > 0
