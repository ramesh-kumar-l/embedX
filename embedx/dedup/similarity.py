"""Semantic deduplication helpers."""

from __future__ import annotations

from embedx.utils.helpers import cosine_similarity


def is_duplicate(
    embedding: list[float],
    candidates: list[list[float]],
    threshold: float = 0.98,
) -> bool:
    """Return True if embedding is within threshold of any candidate."""
    for candidate in candidates:
        if cosine_similarity(embedding, candidate) >= threshold:
            return True
    return False


def deduplicate(
    embeddings: list[list[float]],
    threshold: float = 0.98,
) -> list[int]:
    """Return indices of unique embeddings (greedy first-seen dedup)."""
    kept: list[int] = []
    kept_vecs: list[list[float]] = []
    for i, emb in enumerate(embeddings):
        if not is_duplicate(emb, kept_vecs, threshold):
            kept.append(i)
            kept_vecs.append(emb)
    return kept
