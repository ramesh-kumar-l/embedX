"""Tests for the observability trace layer."""

from __future__ import annotations

import json

import pytest

from embedx.api.public import EmbedX
from tests.conftest import DummyProvider

_DOCS = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning enables computers to learn from data",
    "Natural language processing understands human text",
]


@pytest.fixture
def db(tmp_db: str):
    with EmbedX(db_path=tmp_db, provider=DummyProvider()) as embedx:
        for doc in _DOCS:
            embedx.add(doc)
        yield embedx


@pytest.fixture
def hybrid_db(tmp_db: str):
    with EmbedX(db_path=tmp_db, provider=DummyProvider(), hybrid=True) as embedx:
        for doc in _DOCS:
            embedx.add(doc)
        yield embedx


# ---------------------------------------------------------------------------
# Return-type contract
# ---------------------------------------------------------------------------


def test_trace_false_returns_list(db: EmbedX) -> None:
    results = db.search("machine learning", trace=False)
    assert isinstance(results, list)


def test_trace_true_returns_tuple(db: EmbedX) -> None:
    out = db.search("machine learning", trace=True)
    assert isinstance(out, tuple)
    results, trace = out
    assert isinstance(results, list)
    assert isinstance(trace, dict)


# ---------------------------------------------------------------------------
# Results consistency — trace must not alter results
# ---------------------------------------------------------------------------


def test_trace_results_match_no_trace(db: EmbedX) -> None:
    plain = db.search("machine learning", top_k=2)
    traced, _ = db.search("machine learning", top_k=2, trace=True)
    assert plain == traced


# ---------------------------------------------------------------------------
# Trace structure
# ---------------------------------------------------------------------------


def test_trace_top_level_keys(db: EmbedX) -> None:
    _, trace = db.search("machine learning", trace=True)
    assert {"query", "mode", "namespace", "cache", "timings", "retrieval", "ranking"} <= trace.keys()


def test_trace_cache_keys(db: EmbedX) -> None:
    _, trace = db.search("machine learning", trace=True)
    assert set(trace["cache"].keys()) == {"l1_hit", "l2_hit"}


def test_trace_timings_keys(db: EmbedX) -> None:
    _, trace = db.search("machine learning", trace=True)
    assert set(trace["timings"].keys()) == {
        "embedding_ms", "vector_ms", "bm25_ms", "fusion_ms", "ranking_ms", "total_ms"
    }


def test_trace_retrieval_keys(db: EmbedX) -> None:
    _, trace = db.search("machine learning", trace=True)
    assert set(trace["retrieval"].keys()) == {
        "semantic_candidates", "bm25_candidates", "final_candidates"
    }


def test_trace_ranking_entry_keys(db: EmbedX) -> None:
    results, trace = db.search("machine learning", top_k=2, trace=True)
    assert len(trace["ranking"]) == len(results)
    for entry in trace["ranking"]:
        assert {"doc_id", "semantic", "bm25", "recency", "importance", "final_score"} <= entry.keys()


# ---------------------------------------------------------------------------
# Timings sanity
# ---------------------------------------------------------------------------


def test_trace_total_ms_positive(db: EmbedX) -> None:
    _, trace = db.search("machine learning", trace=True)
    assert trace["timings"]["total_ms"] > 0


def test_trace_all_timings_non_negative(db: EmbedX) -> None:
    _, trace = db.search("machine learning", trace=True)
    for key, val in trace["timings"].items():
        assert val >= 0, f"{key} was negative: {val}"


# ---------------------------------------------------------------------------
# Mode field
# ---------------------------------------------------------------------------


def test_trace_mode_semantic(db: EmbedX) -> None:
    _, trace = db.search("machine learning", trace=True)
    assert trace["mode"] == "semantic"


def test_trace_mode_hybrid(hybrid_db: EmbedX) -> None:
    _, trace = hybrid_db.search("machine learning", trace=True)
    assert trace["mode"] == "hybrid"


def test_trace_mode_hybrid_per_query(db: EmbedX) -> None:
    _, trace = db.search("machine learning", hybrid=True, trace=True)
    assert trace["mode"] == "hybrid"


# ---------------------------------------------------------------------------
# Cache hit recording
# ---------------------------------------------------------------------------


def test_trace_l1_cache_hit(db: EmbedX) -> None:
    # Exact text match → L1 hit
    db.add("cache hit test phrase")
    _, trace = db.search("cache hit test phrase", trace=True)
    assert trace["cache"]["l1_hit"] is True
    assert trace["cache"]["l2_hit"] is False


def test_trace_no_cache_hit(db: EmbedX) -> None:
    _, trace = db.search("completely novel query xyz", trace=True)
    assert trace["cache"]["l1_hit"] is False
    assert trace["cache"]["l2_hit"] is False


# ---------------------------------------------------------------------------
# Hybrid-specific trace fields
# ---------------------------------------------------------------------------


def test_trace_hybrid_candidate_counts(hybrid_db: EmbedX) -> None:
    _, trace = hybrid_db.search("fox", trace=True)
    ret = trace["retrieval"]
    assert ret["semantic_candidates"] >= 0
    assert ret["bm25_candidates"] >= 0
    assert ret["final_candidates"] >= 0


def test_trace_hybrid_timings_populated(hybrid_db: EmbedX) -> None:
    _, trace = hybrid_db.search("machine learning", trace=True)
    t = trace["timings"]
    assert t["vector_ms"] > 0
    assert t["bm25_ms"] >= 0
    assert t["fusion_ms"] > 0
    assert t["ranking_ms"] >= 0


# ---------------------------------------------------------------------------
# Namespace
# ---------------------------------------------------------------------------


def test_trace_namespace_recorded(db: EmbedX) -> None:
    db.add("namespaced document", namespace="ns1")
    _, trace = db.search("namespaced document", namespace="ns1", trace=True)
    assert trace["namespace"] == "ns1"


# ---------------------------------------------------------------------------
# JSON serialisability
# ---------------------------------------------------------------------------


def test_trace_json_serializable(db: EmbedX) -> None:
    _, trace = db.search("machine learning", trace=True)
    serialized = json.dumps(trace)
    assert len(serialized) > 0
    restored = json.loads(serialized)
    assert restored["query"] == trace["query"]
