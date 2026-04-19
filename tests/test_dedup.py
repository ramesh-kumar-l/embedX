"""Tests for semantic deduplication."""

from __future__ import annotations

import numpy as np
import pytest

from embedx.dedup.similarity import deduplicate, is_duplicate


def unit_vec(dim: int, idx: int) -> list[float]:
    """One-hot unit vector."""
    v = [0.0] * dim
    v[idx] = 1.0
    return v


class TestDedup:
    def test_identical_is_duplicate(self):
        v = unit_vec(4, 0)
        assert is_duplicate(v, [v], threshold=0.99)

    def test_orthogonal_not_duplicate(self):
        v1 = unit_vec(4, 0)
        v2 = unit_vec(4, 1)
        assert not is_duplicate(v1, [v2], threshold=0.99)

    def test_deduplicate_removes_duplicates(self):
        v = unit_vec(4, 0)
        kept = deduplicate([v, v, v], threshold=0.99)
        assert len(kept) == 1

    def test_deduplicate_keeps_unique(self):
        vecs = [unit_vec(4, i) for i in range(4)]
        kept = deduplicate(vecs, threshold=0.99)
        assert len(kept) == 4

    def test_empty_candidates(self):
        v = unit_vec(4, 0)
        assert not is_duplicate(v, [], threshold=0.99)
