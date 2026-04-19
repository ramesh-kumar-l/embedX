"""Composite ranking engine: semantic + recency + frequency."""

from __future__ import annotations

import time
from dataclasses import dataclass

from embedx.storage.sqlite_store import Record
from embedx.utils.helpers import clamp


@dataclass
class RankedResult:
    record: Record
    semantic_score: float
    recency_score: float
    frequency_score: float
    importance_score: float
    final_score: float
    text: str
    metadata: dict


_DEFAULT_WEIGHTS = {"semantic": 0.8, "recency": 0.1, "frequency": 0.1, "importance": 0.0}
_RECENCY_HALF_LIFE = 7 * 24 * 3600  # 7 days in seconds


class RankingScorer:
    def __init__(
        self,
        semantic_weight: float = 0.8,
        recency_weight: float = 0.1,
        frequency_weight: float = 0.1,
        importance_weight: float = 0.0,
    ) -> None:
        total = semantic_weight + recency_weight + frequency_weight + importance_weight
        self.w_semantic = semantic_weight / total
        self.w_recency = recency_weight / total
        self.w_frequency = frequency_weight / total
        self.w_importance = importance_weight / total

    def score(
        self,
        record: Record,
        semantic_score: float,
        max_use_count: int = 1,
    ) -> RankedResult:
        recency = self._recency_score(record.last_used)
        frequency = self._frequency_score(record.use_count, max_use_count)
        importance = clamp(record.importance, 0.0, 1.0)

        final = (
            self.w_semantic * clamp(semantic_score, 0.0, 1.0)
            + self.w_recency * recency
            + self.w_frequency * frequency
            + self.w_importance * importance
        )

        return RankedResult(
            record=record,
            semantic_score=semantic_score,
            recency_score=recency,
            frequency_score=frequency,
            importance_score=importance,
            final_score=final,
            text=record.text,
            metadata=record.metadata,
        )

    def rank(
        self,
        candidates: list[tuple[Record, float]],
    ) -> list[RankedResult]:
        max_use = max((r.use_count for r, _ in candidates), default=1) or 1
        ranked = [self.score(r, s, max_use) for r, s in candidates]
        ranked.sort(key=lambda x: x.final_score, reverse=True)
        return ranked

    def _recency_score(self, last_used: float) -> float:
        age = time.time() - last_used
        return float(2 ** (-age / _RECENCY_HALF_LIFE))

    def _frequency_score(self, use_count: int, max_use: int) -> float:
        if max_use == 0:
            return 0.0
        return clamp(use_count / max_use, 0.0, 1.0)
