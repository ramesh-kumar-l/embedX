"""
EmbedX latency and cost-savings benchmark.

Run: python benchmarks/latency_benchmark.py
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
    "Embeddings map text to dense numerical vectors that capture semantic relationships.",
    "Retrieval-augmented generation grounds language model responses in real documents.",
    "Cosine similarity is the standard metric for comparing embedding vectors.",
    "Large language models emerge surprisingly capable behaviors from scale and data.",
    "Fine-tuning adapts a pre-trained model to a specific downstream task.",
    "Prompt engineering crafts input sequences to elicit desired model behaviors.",
    "Chain-of-thought prompting improves reasoning by requesting intermediate steps.",
    "Zero-shot learning generalizes to unseen tasks without task-specific training data.",
    "Few-shot learning adapts quickly to new tasks given only a handful of examples.",
    "Reinforcement learning from human feedback aligns models with human preferences.",
    "Attention mechanisms let models focus on relevant parts of the input context.",
    "Tokenization converts raw text into discrete units processed by neural networks.",
    "Byte-pair encoding is a subword tokenization strategy used in modern LLMs.",
    "Perplexity measures how well a language model predicts a held-out test corpus.",
]

QUERIES = CORPUS[:10]  # first half as queries

DB_PATH = "_bench_result.db"


def _make_provider():
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
        from embedx.providers.local_model import LocalModelProvider
        return LocalModelProvider()
    except ImportError:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from tests.conftest import DummyProvider
        print("  (sentence-transformers not installed — using DummyProvider for benchmark)")
        return DummyProvider()


def _make_db(db_path: str):
    from embedx.api.public import EmbedX
    return EmbedX(db_path=db_path, provider=_make_provider())


def run_benchmark() -> None:
    print("\n  EmbedX Latency & Cost Benchmark")
    print("  " + "=" * 50)

    # ── Phase 1: Cold adds ────────────────────────────────────────────
    print("\n  Phase 1: Cold adds (first time, no cache)")
    with _make_db(DB_PATH) as db:
        db.clear()
        latencies = []
        for text in CORPUS:
            t0 = time.perf_counter()
            db.add(text)
            latencies.append((time.perf_counter() - t0) * 1000)
        cold_total = sum(latencies)
        cold_avg = cold_total / len(latencies)
        print(f"    Documents    : {len(CORPUS)}")
        print(f"    Total time   : {cold_total:.0f}ms")
        print(f"    Avg/doc      : {cold_avg:.1f}ms")

    # ── Phase 2: Warm adds (exact cache) ─────────────────────────────
    print("\n  Phase 2: Warm adds (exact cache hits)")
    with _make_db(DB_PATH) as db:
        latencies = []
        hits = 0
        for text in CORPUS:
            t0 = time.perf_counter()
            result = db.add(text)
            latencies.append((time.perf_counter() - t0) * 1000)
            if result["status"] != "added":
                hits += 1
        warm_total = sum(latencies)
        warm_avg = warm_total / len(latencies)
        print(f"    Cache hits   : {hits}/{len(CORPUS)}")
        print(f"    Total time   : {warm_total:.0f}ms")
        print(f"    Avg/doc      : {warm_avg:.1f}ms")
        speedup = cold_total / warm_total if warm_total > 0 else float("inf")
        print(f"    Speedup      : {speedup:.1f}x")

    # ── Phase 3: Search latency ───────────────────────────────────────
    print("\n  Phase 3: Search latency")
    with _make_db(DB_PATH) as db:
        latencies = []
        for query in QUERIES:
            t0 = time.perf_counter()
            db.search(query, top_k=5)
            latencies.append((time.perf_counter() - t0) * 1000)
        search_avg = sum(latencies) / len(latencies)
        print(f"    Queries      : {len(QUERIES)}")
        print(f"    Avg latency  : {search_avg:.1f}ms")
        print(f"    Min latency  : {min(latencies):.1f}ms")
        print(f"    Max latency  : {max(latencies):.1f}ms")

    # ── Phase 4: Stats ────────────────────────────────────────────────
    print("\n  Phase 4: Cost & cache summary")
    with _make_db(DB_PATH) as db:
        for text in CORPUS:
            db.add(text)
        for query in QUERIES:
            db.search(query, top_k=3)
        stats = db.get_stats()
        print(f"    Hit rate     : {stats['hit_rate'] * 100:.1f}%")
        print(f"    Exact hits   : {stats['exact_hits']}")
        print(f"    Semantic hits: {stats['semantic_hits']}")
        print(f"    Cost saved   : ${stats['cost_saved_usd']:.6f}")
        print(f"    Tokens saved : {stats['tokens_saved']:,}")

    # ── Summary ───────────────────────────────────────────────────────
    saving_pct = max(0.0, (1 - warm_avg / cold_avg) * 100) if cold_avg > 0 else 0
    print(f"\n  Summary")
    print("  " + "-" * 40)
    print(f"    Without cache : {cold_avg:.1f}ms avg")
    print(f"    With EmbedX   : {warm_avg:.1f}ms avg")
    print(f"    Savings       : {saving_pct:.0f}%")
    print()

    try:
        os.remove(DB_PATH)
    except OSError:
        pass


if __name__ == "__main__":
    run_benchmark()
