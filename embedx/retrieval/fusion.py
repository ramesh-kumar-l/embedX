"""Weighted score fusion for combining semantic and keyword search results."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from embedx.storage.sqlite_store import Record


def fuse(
    semantic_results: list[tuple["Record", float]],
    keyword_results: list[tuple["Record", float]],
    w_semantic: float = 0.7,
    w_keyword: float = 0.3,
    top_k: int = 10,
) -> list[tuple["Record", float]]:
    """Merge two ranked lists into one via weighted score fusion.

    Scores from each list are independently max-normalized to [0, 1]
    before combining, so the two scales are comparable regardless of
    whether they come from cosine similarity or BM25.
    """
    if not semantic_results and not keyword_results:
        return []

    all_records: dict[int, "Record"] = {}
    sem_map: dict[int, float] = {}
    kw_map: dict[int, float] = {}

    for rec, sc in semantic_results:
        all_records[rec.id] = rec
        sem_map[rec.id] = sc

    for rec, sc in keyword_results:
        all_records[rec.id] = rec
        kw_map[rec.id] = sc

    sem_max = max(sem_map.values(), default=1.0) or 1.0
    kw_max = max(kw_map.values(), default=1.0) or 1.0

    fused: list[tuple["Record", float]] = []
    for doc_id, rec in all_records.items():
        sem_n = sem_map.get(doc_id, 0.0) / sem_max
        kw_n = kw_map.get(doc_id, 0.0) / kw_max
        score = w_semantic * sem_n + w_keyword * kw_n
        fused.append((rec, score))

    fused.sort(key=lambda x: x[1], reverse=True)
    return fused[:top_k]
