"""Lightweight heuristic reranker: no ML models, no extra dependencies."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from embedx.storage.sqlite_store import Record


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def rerank(
    query: str,
    results: list[tuple["Record", float]],
) -> list[tuple["Record", float]]:
    """Boost fused scores using exact-phrase match, token coverage, and length.

    Boosts are additive and intentionally small so they only break ties
    rather than invert the primary retrieval signal.
    """
    if not results:
        return results

    query_tokens = _token_set(query)
    query_lower = query.lower()

    reranked: list[tuple["Record", float]] = []
    for rec, score in results:
        doc_lower = rec.text.lower()
        doc_tokens = _token_set(rec.text)
        doc_len = len(re.findall(r"\w+", rec.text))

        # +0.10 if the full query phrase appears verbatim in the document
        exact_bonus = 0.10 if query_lower in doc_lower else 0.0

        # +0.05 * fraction of query tokens present in document
        if query_tokens:
            coverage = len(query_tokens & doc_tokens) / len(query_tokens)
        else:
            coverage = 0.0
        coverage_bonus = 0.05 * coverage

        # Slight penalty for very short documents (< 5 tokens), scales up to 1.0
        # at 20+ tokens — avoids surfacing single-word stubs above real content.
        length_factor = 0.80 + 0.20 * min(1.0, doc_len / 20.0)

        boosted = score * length_factor + exact_bonus + coverage_bonus
        reranked.append((rec, boosted))

    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked
