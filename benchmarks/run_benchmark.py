"""
EmbedX with-cache vs without-cache benchmark.

Run: python benchmarks/run_benchmark.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CORPUS = [
    "Machine learning enables computers to learn from data without explicit programming.",
    "Natural language processing allows machines to understand and generate human language.",
    "Deep learning uses artificial neural networks with multiple layers to learn features.",
    "Transformer models use self-attention to process sequential data in parallel.",
    "Vector databases store high-dimensional embeddings and support similarity search.",
    "Semantic search retrieves documents by meaning rather than keyword overlap.",
    "Embeddings map text to dense numerical vectors capturing semantic relationships.",
    "Retrieval-augmented generation grounds language model responses in real documents.",
    "Cosine similarity is the standard metric for comparing embedding vectors.",
    "Large language models emerge surprisingly capable behaviors from scale and data.",
]

DB_PATH = "_run_bench.db"


def _make_db():
    from embedx.api.public import EmbedX
    try:
        from embedx.providers.local_model import LocalModelProvider
        return EmbedX(db_path=DB_PATH, provider=LocalModelProvider())
    except ImportError:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from tests.conftest import DummyProvider
        print("  (sentence-transformers not installed — using DummyProvider)")
        return EmbedX(db_path=DB_PATH, provider=DummyProvider())


def run() -> None:
    print("\n  EmbedX — With Cache vs Without Cache")
    print("  " + "=" * 44)

    # ── Without cache: fresh adds on every run ────────────────────────
    print("\n  [Without cache]  cold embed every document")
    with _make_db() as db:
        db.clear()
        t0 = time.perf_counter()
        for text in CORPUS:
            db.add(text)
        no_cache_ms = (time.perf_counter() - t0) * 1000
    print(f"  Time : {no_cache_ms:.0f}ms  ({no_cache_ms / len(CORPUS):.1f}ms/doc)")

    # ── With cache: same docs → 100% exact hits ───────────────────────
    print("\n  [With EmbedX]    exact cache on repeated content")
    with _make_db() as db:
        t0 = time.perf_counter()
        for text in CORPUS:
            db.add(text)
        cache_ms = (time.perf_counter() - t0) * 1000
        stats = db.get_stats()
    print(f"  Time : {cache_ms:.0f}ms  ({cache_ms / len(CORPUS):.1f}ms/doc)")
    print(f"  Hit rate : {stats['hit_rate'] * 100:.0f}%")

    # ── Summary ───────────────────────────────────────────────────────
    saved_pct = max(0.0, (1 - cache_ms / no_cache_ms) * 100) if no_cache_ms > 0 else 0.0
    speedup = no_cache_ms / cache_ms if cache_ms > 0 else float("inf")

    print(f"\n  Without cache : {no_cache_ms:.0f}ms")
    print(f"  With EmbedX   : {cache_ms:.0f}ms")
    print(f"  Cost saved    : {saved_pct:.0f}%")
    print(f"  Speedup       : {speedup:.1f}x\n")

    try:
        os.remove(DB_PATH)
    except OSError:
        pass


if __name__ == "__main__":
    run()
