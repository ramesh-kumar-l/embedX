"""Hybrid retriever: semantic vector search + BM25 + score fusion + reranking."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from embedx.retrieval.bm25 import BM25Index
from embedx.retrieval.fusion import fuse
from embedx.retrieval.reranker import rerank
from embedx.utils.trace import TraceContext, timer

if TYPE_CHECKING:
    from embedx.index.fallback_index import FallbackIndex
    from embedx.ranking.scorer import RankedResult, RankingScorer
    from embedx.storage.sqlite_store import SQLiteStore


class HybridRetriever:
    """Combines semantic and keyword retrieval for improved recall.

    Uses BM25 for keyword matching and delegates to the existing
    FallbackIndex for semantic search. Scores are fused then optionally
    reranked before being passed to the composite RankingScorer.
    """

    def __init__(
        self,
        store: "SQLiteStore",
        index: "FallbackIndex",
        scorer: "RankingScorer",
        w_semantic: float = 0.7,
        w_keyword: float = 0.3,
        use_reranker: bool = True,
    ) -> None:
        self._store = store
        self._index = index
        self._scorer = scorer
        self._w_semantic = w_semantic
        self._w_keyword = w_keyword
        self._use_reranker = use_reranker
        self._bm25 = BM25Index(store)

    def mark_dirty(self) -> None:
        """Propagate index staleness after a document is added or removed."""
        self._bm25.mark_dirty()

    def search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int,
        trace_ctx: Optional[TraceContext] = None,
    ) -> list["RankedResult"]:
        """Run hybrid search and return RankedResults ready for the public API.

        Fetches a wider candidate pool (3× top_k, min 20) from both indexes,
        fuses the scores, optionally reranks, then trims to top_k before
        passing through the composite RankingScorer (recency + frequency).

        *trace_ctx* is optional — when provided, timing and candidate counts are
        recorded. When ``None`` all instrumentation is skipped with no overhead.
        """
        fetch_k = max(top_k * 3, 20)

        with timer(trace_ctx, "vector_ms"):
            semantic_candidates = self._index.search(query_embedding, top_k=fetch_k)

        with timer(trace_ctx, "bm25_ms"):
            keyword_candidates = self._bm25.search(query, top_k=fetch_k)

        if trace_ctx is not None:
            trace_ctx.semantic_candidates = len(semantic_candidates)
            trace_ctx.bm25_candidates = len(keyword_candidates)
            trace_ctx._pre_fusion_sem = {r.id: s for r, s in semantic_candidates}
            trace_ctx._pre_fusion_bm25 = {r.id: s for r, s in keyword_candidates}

        with timer(trace_ctx, "fusion_ms"):
            fused = fuse(
                semantic_candidates,
                keyword_candidates,
                w_semantic=self._w_semantic,
                w_keyword=self._w_keyword,
                top_k=fetch_k,
            )
            if self._use_reranker:
                fused = rerank(query, fused)

        top_candidates = fused[:top_k]

        with timer(trace_ctx, "ranking_ms"):
            ranked = self._scorer.rank(top_candidates)

        return ranked
