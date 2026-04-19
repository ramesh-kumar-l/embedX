"""Tests for the ranking engine."""

from __future__ import annotations

import time

from embedx.ranking.scorer import RankingScorer
from embedx.storage.sqlite_store import Record


def make_record(id: int, use_count: int = 1, age_secs: float = 0) -> Record:
    ts = time.time() - age_secs
    return Record(
        id=id,
        text=f"text_{id}",
        text_hash=f"hash_{id}",
        embedding=[0.1] * 10,
        metadata={},
        created_at=ts,
        last_used=ts,
        use_count=use_count,
    )


class TestRankingScorer:
    def test_higher_semantic_wins(self):
        scorer = RankingScorer()
        r1 = make_record(1, use_count=1)
        r2 = make_record(2, use_count=1)
        ranked = scorer.rank([(r1, 0.95), (r2, 0.60)])
        assert ranked[0].record.id == 1

    def test_weights_sum_to_one(self):
        scorer = RankingScorer(0.5, 0.3, 0.2)
        total = scorer.w_semantic + scorer.w_recency + scorer.w_frequency
        assert abs(total - 1.0) < 1e-6

    def test_frequency_boosts_score(self):
        scorer = RankingScorer(semantic_weight=0.5, frequency_weight=0.4, recency_weight=0.1)
        r_frequent = make_record(1, use_count=100)
        r_rare = make_record(2, use_count=1)
        ranked = scorer.rank([(r_frequent, 0.80), (r_rare, 0.80)])
        assert ranked[0].record.id == 1

    def test_deterministic(self):
        scorer = RankingScorer()
        r1 = make_record(1)
        r2 = make_record(2)
        ranked_a = scorer.rank([(r1, 0.9), (r2, 0.7)])
        ranked_b = scorer.rank([(r1, 0.9), (r2, 0.7)])
        assert [r.record.id for r in ranked_a] == [r.record.id for r in ranked_b]

    def test_empty_candidates(self):
        scorer = RankingScorer()
        assert scorer.rank([]) == []
