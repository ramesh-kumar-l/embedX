"""Pure-Python BM25 index built over existing SQLite text records."""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from embedx.storage.sqlite_store import Record, SQLiteStore


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25Index:
    """BM25 keyword index. Lazy-built from SQLite; cached in memory.

    Stays in sync with the store via mark_dirty() — call it whenever
    documents are added or removed.
    """

    def __init__(
        self,
        store: "SQLiteStore",
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._store = store
        self._k1 = k1
        self._b = b
        self._dirty = True

        self._records: list["Record"] = []
        self._doc_tokens: list[list[str]] = []
        self._df: dict[str, int] = {}
        self._avgdl: float = 1.0
        self._n: int = 0

    def mark_dirty(self) -> None:
        """Signal that the underlying store has changed."""
        self._dirty = True

    def rebuild(self) -> None:
        """Load all records from SQLite and recompute BM25 structures."""
        self._records = self._store.all_records()
        self._n = len(self._records)
        self._doc_tokens = [_tokenize(r.text) for r in self._records]

        total_len = sum(len(toks) for toks in self._doc_tokens)
        self._avgdl = total_len / self._n if self._n > 0 else 1.0

        self._df = {}
        for tokens in self._doc_tokens:
            for term in set(tokens):
                self._df[term] = self._df.get(term, 0) + 1

        self._dirty = False

    def search(self, query: str, top_k: int) -> list[tuple["Record", float]]:
        """Return up to top_k (record, bm25_score) pairs sorted descending."""
        if self._dirty:
            self.rebuild()
        if self._n == 0:
            return []

        query_terms = _tokenize(query)
        if not query_terms:
            return []

        k1 = self._k1
        b = self._b
        avgdl = self._avgdl
        n = self._n

        scores = [0.0] * n
        for term in query_terms:
            if term not in self._df:
                continue
            df_t = self._df[term]
            idf = math.log((n - df_t + 0.5) / (df_t + 0.5) + 1.0)
            for i, tokens in enumerate(self._doc_tokens):
                tf = tokens.count(term)
                if tf == 0:
                    continue
                dl = len(tokens)
                scores[i] += idf * (tf * (k1 + 1)) / (
                    tf + k1 * (1.0 - b + b * dl / avgdl)
                )

        paired = [
            (self._records[i], scores[i])
            for i in range(n)
            if scores[i] > 0.0
        ]
        paired.sort(key=lambda x: x[1], reverse=True)
        return paired[:top_k]
