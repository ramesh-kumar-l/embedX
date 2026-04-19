"""Server endpoint tests — requires fastapi and httpx."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from tests.conftest import DummyProvider


@pytest.fixture
def client(tmp_path):
    import server.main as srv

    # Patch provider so tests stay offline and fast
    original_embedx_cls = srv.EmbedX

    class PatchedEmbedX(original_embedx_cls):
        def __init__(self, db_path="embedx.db", **kwargs):
            super().__init__(db_path=db_path, provider=DummyProvider(), **kwargs)

    srv.EmbedX = PatchedEmbedX

    import os
    os.environ["EMBEDX_DB"] = str(tmp_path / "srv_test.db")

    with TestClient(srv.app) as c:
        yield c

    srv.EmbedX = original_embedx_cls


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# /add
# ---------------------------------------------------------------------------

def test_add_new_document(client):
    r = client.post("/add", json={"text": "hello world"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "added"
    assert "id" in body
    assert "latency_ms" in body


def test_add_duplicate_is_exact_hit(client):
    text = "duplicate document"
    client.post("/add", json={"text": text})
    r = client.post("/add", json={"text": text})
    assert r.status_code == 200
    assert r.json()["status"] == "exact_hit"


def test_add_with_metadata(client):
    r = client.post("/add", json={"text": "doc with meta", "metadata": {"source": "test"}})
    assert r.status_code == 200
    assert r.json()["status"] == "added"


# ---------------------------------------------------------------------------
# /search
# ---------------------------------------------------------------------------

def test_search_returns_list(client):
    client.post("/add", json={"text": "semantic memory for LLM apps"})
    r = client.post("/search", json={"query": "semantic memory", "top_k": 3})
    assert r.status_code == 200
    results = r.json()
    assert isinstance(results, list)


def test_search_result_shape(client):
    client.post("/add", json={"text": "vector search is fast"})
    r = client.post("/search", json={"query": "vector search", "top_k": 1})
    assert r.status_code == 200
    results = r.json()
    if results:
        keys = results[0].keys()
        assert "text" in keys
        assert "score" in keys
        assert "metadata" in keys


def test_search_top_k_respected(client):
    for i in range(5):
        client.post("/add", json={"text": f"document number {i} about retrieval"})
    r = client.post("/search", json={"query": "retrieval", "top_k": 2})
    assert r.status_code == 200
    assert len(r.json()) <= 2


# ---------------------------------------------------------------------------
# /stats
# ---------------------------------------------------------------------------

def test_stats_keys(client):
    r = client.get("/stats")
    assert r.status_code == 200
    body = r.json()
    assert "document_count" in body
    assert "hit_rate" in body
    assert "latency" in body


def test_stats_document_count_increments(client):
    r0 = client.get("/stats")
    before = r0.json()["document_count"]
    client.post("/add", json={"text": "new unique document xyz987"})
    r1 = client.get("/stats")
    assert r1.json()["document_count"] == before + 1


# ---------------------------------------------------------------------------
# Integration: add then search
# ---------------------------------------------------------------------------

def test_add_then_search_integration(client):
    text = "EmbedX reduces embedding costs through intelligent caching"
    client.post("/add", json={"text": text})
    r = client.post("/search", json={"query": "embedding cost reduction", "top_k": 5})
    assert r.status_code == 200
    texts = [res["text"] for res in r.json()]
    assert text in texts
