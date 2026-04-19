"""Evaluation engine: recall@k and keyword match metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from embedx.eval.dataset import EvalDataset

if TYPE_CHECKING:
    from embedx.api.public import EmbedX


@dataclass
class EvalMetrics:
    dataset_name: str
    num_cases: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    keyword_match: float
    per_case: list[dict] = field(default_factory=list)


def evaluate(db: "EmbedX", dataset: EvalDataset, top_k: int = 5) -> EvalMetrics:
    if dataset.corpus:
        for text in dataset.corpus:
            db.add(text)

    hits_at_1 = hits_at_3 = hits_at_5 = kw_hits = 0
    per_case = []

    for case in dataset.cases:
        results = db.search(case.query, top_k=top_k)
        retrieved_texts = [r["text"] for r in results]

        h1 = _recall(retrieved_texts[:1], case.expected)
        h3 = _recall(retrieved_texts[:3], case.expected)
        h5 = _recall(retrieved_texts[:5], case.expected)
        kw = _keyword_match(case.query, retrieved_texts)

        hits_at_1 += h1
        hits_at_3 += h3
        hits_at_5 += h5
        kw_hits += kw

        per_case.append(
            {
                "query": case.query,
                "recall@1": h1,
                "recall@3": h3,
                "recall@5": h5,
                "keyword_match": kw,
                "retrieved": retrieved_texts[:3],
            }
        )

    n = max(len(dataset.cases), 1)
    return EvalMetrics(
        dataset_name=dataset.name,
        num_cases=n,
        recall_at_1=round(hits_at_1 / n, 4),
        recall_at_3=round(hits_at_3 / n, 4),
        recall_at_5=round(hits_at_5 / n, 4),
        keyword_match=round(kw_hits / n, 4),
        per_case=per_case,
    )


def _recall(retrieved: list[str], expected: list[str]) -> float:
    if not expected:
        return 1.0
    hits = sum(1 for e in expected if any(e.lower() in r.lower() for r in retrieved))
    return hits / len(expected)


def _keyword_match(query: str, retrieved: list[str]) -> float:
    keywords = {w.lower() for w in query.split() if len(w) > 3}
    if not keywords or not retrieved:
        return 0.0
    top = retrieved[0].lower()
    matched = sum(1 for kw in keywords if kw in top)
    return matched / len(keywords)
