"""EmbedX public API — the single entry point for all operations."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

from embedx.cache.exact_cache import ExactCache
from embedx.cache.semantic_cache import SemanticCache
from embedx.index.fallback_index import FallbackIndex
from embedx.metrics.cost import CostTracker
from embedx.metrics.stats import LatencyTracker
from embedx.providers.base import BaseProvider
from embedx.ranking.scorer import RankedResult, RankingScorer
from embedx.storage.sqlite_store import SQLiteStore
from embedx.utils.helpers import hash_text
from embedx.utils.trace import TraceContext, timer


class EmbedX:
    """
    Cost-aware semantic memory and retrieval for LLM applications.

    Usage::

        db = EmbedX()
        db.add("AI is transforming software development")
        results = db.search("What is artificial intelligence?")
    """

    def __init__(
        self,
        db_path: str = "embedx.db",
        provider: Optional[BaseProvider] = None,
        semantic_threshold: float = 0.92,
        semantic_weight: float = 0.8,
        recency_weight: float = 0.1,
        frequency_weight: float = 0.1,
        importance_weight: float = 0.0,
        use_faiss: bool = False,
        hybrid: bool = False,
        hybrid_semantic_weight: float = 0.7,
        hybrid_keyword_weight: float = 0.3,
    ) -> None:
        self._store = SQLiteStore(db_path)
        self._index = FallbackIndex(self._store)

        if provider is None:
            from embedx.providers.local_model import LocalModelProvider

            provider = LocalModelProvider()
        self._provider = provider

        self._exact = ExactCache(self._store)
        self._semantic = SemanticCache(self._store, self._index, threshold=semantic_threshold)
        self._scorer = RankingScorer(semantic_weight, recency_weight, frequency_weight, importance_weight)
        self._cost = CostTracker(self._store)
        self._latency = LatencyTracker()

        if use_faiss:
            from embedx.index.faiss_index import FAISSIndex

            self._faiss_index: Optional[FAISSIndex] = FAISSIndex(
                self._store, self._provider.dimension
            )
        else:
            self._faiss_index = None

        self._use_hybrid = hybrid
        self._hybrid_retriever: Optional[Any] = None
        if hybrid:
            from embedx.retrieval.hybrid import HybridRetriever

            self._hybrid_retriever = HybridRetriever(
                self._store,
                self._index,
                self._scorer,
                w_semantic=hybrid_semantic_weight,
                w_keyword=hybrid_keyword_weight,
            )

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def add(
        self,
        text: str,
        metadata: Optional[dict] = None,
        namespace: str = "default",
        importance: float = 0.5,
    ) -> dict:
        """
        Add a text document. Returns cache status and record id.

        Skips embedding if an exact or near-duplicate already exists.
        ``namespace`` logically groups documents; ``importance`` (0–1) boosts
        ranking when importance_weight > 0 on the EmbedX instance.
        """
        t0 = time.perf_counter()

        existing = self._exact.get(text)
        if existing:
            ms = (time.perf_counter() - t0) * 1000
            self._latency.record(ms)
            return {"status": "exact_hit", "id": existing.id, "latency_ms": round(ms, 2)}

        result = self._provider.embed(text)
        sem_hit = self._semantic.get(result.embedding)
        if sem_hit and sem_hit.text == text:
            ms = (time.perf_counter() - t0) * 1000
            self._latency.record(ms)
            return {"status": "semantic_hit", "id": sem_hit.id, "latency_ms": round(ms, 2)}

        record_id = self._exact.put(text, result.embedding, metadata, namespace=namespace, importance=importance)
        self._cost.record_embedding(result.token_count, result.cost_usd)
        self._store.increment_stat("total_adds")

        record = self._store.get_by_hash(hash_text(text))
        if record:
            self._index.add(record)

        if self._hybrid_retriever is not None:
            self._hybrid_retriever.mark_dirty()

        ms = (time.perf_counter() - t0) * 1000
        self._latency.record(ms)
        return {"status": "added", "id": record_id, "latency_ms": round(ms, 2)}

    def search(
        self,
        query: str,
        top_k: int = 5,
        hybrid: bool = False,
        namespace: Optional[str] = None,
        trace: bool = False,
    ) -> "list[dict] | tuple[list[dict], dict]":
        """
        Search for semantically similar documents.

        Returns a list of dicts with text, score, metadata.
        Cache hits skip re-embedding the query.

        Set ``hybrid=True`` to combine semantic and BM25 keyword search.
        Set ``namespace`` to restrict results to a specific logical group.
        Set ``trace=True`` to return ``(results, trace_dict)`` with a full
        execution breakdown (timings, cache path, candidate counts, scores).
        """
        t0 = time.perf_counter()
        ctx: Optional[TraceContext] = TraceContext() if trace else None

        exact = self._exact.get(query)
        if exact:
            query_embedding = exact.embedding
            self._cost.record_cache_hit(len(query.split()))
            if ctx is not None:
                ctx.l1_hit = True
        else:
            with timer(ctx, "embedding_ms"):
                result = self._provider.embed(query)
            query_embedding = result.embedding
            sem_hit = self._semantic.get(query_embedding)
            if sem_hit:
                query_embedding = sem_hit.embedding
                self._cost.record_cache_hit(result.token_count)
                if ctx is not None:
                    ctx.l2_hit = True
            else:
                self._cost.record_embedding(result.token_count, result.cost_usd)

        self._store.increment_stat("total_searches")

        use_hybrid = hybrid or self._use_hybrid
        if namespace is not None:
            with timer(ctx, "vector_ms"):
                candidates = self._search_namespace(namespace, query_embedding, top_k)
            if ctx is not None:
                ctx.semantic_candidates = len(candidates)
            with timer(ctx, "ranking_ms"):
                ranked = self._scorer.rank(candidates)
        elif use_hybrid:
            if self._hybrid_retriever is None:
                from embedx.retrieval.hybrid import HybridRetriever

                self._hybrid_retriever = HybridRetriever(
                    self._store, self._index, self._scorer
                )
            ranked = self._hybrid_retriever.search(
                query, query_embedding, top_k=top_k, trace_ctx=ctx
            )
        else:
            with timer(ctx, "vector_ms"):
                candidates = self._index.search(query_embedding, top_k=top_k)
            if ctx is not None:
                ctx.semantic_candidates = len(candidates)
            with timer(ctx, "ranking_ms"):
                ranked = self._scorer.rank(candidates)

        ms = (time.perf_counter() - t0) * 1000
        self._latency.record(ms)

        results = [
            {
                "text": r.text,
                "score": round(r.final_score, 4),
                "semantic_score": round(r.semantic_score, 4),
                "metadata": r.metadata,
                "use_count": r.record.use_count,
            }
            for r in ranked
        ]

        if ctx is not None:
            ctx.total_ms = ms
            ctx.final_candidates = len(ranked)
            for r in ranked:
                doc_id = r.record.id
                ctx._doc_scores[doc_id] = {
                    "doc_id": doc_id,
                    "semantic": round(
                        ctx._pre_fusion_sem.get(doc_id, r.semantic_score), 4
                    ),
                    "bm25": round(ctx._pre_fusion_bm25.get(doc_id, 0.0), 4),
                    "recency": round(r.recency_score, 4),
                    "importance": round(r.importance_score, 4),
                    "final_score": round(r.final_score, 4),
                }
            mode = "hybrid" if use_hybrid else "semantic"
            return results, ctx.to_dict(query=query, mode=mode, namespace=namespace)

        return results

    def _search_namespace(
        self,
        namespace: str,
        query_embedding: list[float],
        top_k: int,
    ) -> list[tuple]:
        """Cosine search restricted to a single namespace (SQL-filtered)."""
        import numpy as np

        records = self._store.records_by_namespace(namespace)
        if not records:
            return []
        matrix = np.array([r.embedding for r in records], dtype=np.float32)
        qvec = np.array(query_embedding, dtype=np.float32)
        norm_q = float(np.linalg.norm(qvec))
        if norm_q == 0.0:
            return []
        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0] = 1e-10
        scores = (matrix @ qvec) / (norms * norm_q)
        k = min(top_k, len(records))
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [(records[i], float(scores[i])) for i in top_idx]

    def add_batch(self, texts: list[str], metadata: Optional[list[dict]] = None) -> list[dict]:
        """Add multiple texts efficiently using batch embedding."""
        metas = metadata or [None] * len(texts)
        to_embed: list[tuple[int, str]] = []
        results = [None] * len(texts)

        for i, text in enumerate(texts):
            existing = self._exact.get(text)
            if existing:
                results[i] = {"status": "exact_hit", "id": existing.id}
            else:
                to_embed.append((i, text))

        if to_embed:
            raw_texts = [t for _, t in to_embed]
            embed_results = self._provider.embed_batch(raw_texts)
            for (i, text), emb_result in zip(to_embed, embed_results):
                record_id = self._exact.put(text, emb_result.embedding, metas[i])
                self._cost.record_embedding(emb_result.token_count, emb_result.cost_usd)
                record = self._store.get_by_hash(hash_text(text))
                if record:
                    self._index.add(record)
                results[i] = {"status": "added", "id": record_id}

        self._index.mark_dirty()
        return results

    # ------------------------------------------------------------------
    # Stats and diagnostics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return JSON-serializable stats dict (VS Code extension friendly)."""
        cost_summary = self._cost.get_summary()
        latency_summary = self._latency.summary()
        return {
            "document_count": self._store.count(),
            **cost_summary,
            "latency": latency_summary,
        }

    def rebuild_index(self) -> None:
        """Force index rebuild from SQLite (useful after manual DB changes)."""
        self._index.mark_dirty()
        self._index.rebuild()

    def clear(self) -> None:
        """Delete all stored documents and reset stats."""
        self._store.clear()
        self._index.mark_dirty()

    def close(self) -> None:
        self._store.close()

    def __enter__(self) -> "EmbedX":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
