"""Lightweight trace/observability context for the EmbedX search pipeline."""

from __future__ import annotations

import contextlib
import time
from typing import Generator, Optional


class TraceContext:
    """Mutable context object populated during a traced search execution.

    Passed through the pipeline when trace=True; all instrumentation points
    check ``if ctx is not None`` so there is zero overhead when disabled.
    """

    __slots__ = (
        "l1_hit",
        "l2_hit",
        "embedding_ms",
        "vector_ms",
        "bm25_ms",
        "fusion_ms",
        "ranking_ms",
        "total_ms",
        "semantic_candidates",
        "bm25_candidates",
        "final_candidates",
        "_doc_scores",
        "_pre_fusion_sem",
        "_pre_fusion_bm25",
    )

    def __init__(self) -> None:
        self.l1_hit: bool = False
        self.l2_hit: bool = False
        self.embedding_ms: float = 0.0
        self.vector_ms: float = 0.0
        self.bm25_ms: float = 0.0
        self.fusion_ms: float = 0.0
        self.ranking_ms: float = 0.0
        self.total_ms: float = 0.0
        self.semantic_candidates: int = 0
        self.bm25_candidates: int = 0
        self.final_candidates: int = 0
        self._doc_scores: dict[int, dict] = {}
        # Pre-fusion per-doc scores set by HybridRetriever when tracing
        self._pre_fusion_sem: dict[int, float] = {}
        self._pre_fusion_bm25: dict[int, float] = {}

    def to_dict(self, query: str, mode: str, namespace: Optional[str]) -> dict:
        """Return a JSON-serializable snapshot of the trace."""
        return {
            "query": query,
            "mode": mode,
            "namespace": namespace,
            "cache": {
                "l1_hit": self.l1_hit,
                "l2_hit": self.l2_hit,
            },
            "timings": {
                "embedding_ms": round(self.embedding_ms, 2),
                "vector_ms": round(self.vector_ms, 2),
                "bm25_ms": round(self.bm25_ms, 2),
                "fusion_ms": round(self.fusion_ms, 2),
                "ranking_ms": round(self.ranking_ms, 2),
                "total_ms": round(self.total_ms, 2),
            },
            "retrieval": {
                "semantic_candidates": self.semantic_candidates,
                "bm25_candidates": self.bm25_candidates,
                "final_candidates": self.final_candidates,
            },
            "ranking": list(self._doc_scores.values()),
        }


@contextlib.contextmanager
def timer(ctx: Optional[TraceContext], field: str) -> Generator[None, None, None]:
    """Time a block and write elapsed ms to ``ctx.<field>``.

    When *ctx* is ``None`` (tracing disabled) this is a no-op generator — no
    measurable overhead on the hot path.
    """
    if ctx is None:
        yield
        return
    t0 = time.perf_counter()
    yield
    setattr(ctx, field, (time.perf_counter() - t0) * 1000)
