"""
EmbedX cache benefit demo.

Shows how repeated queries hit the cache instead of re-embedding,
cutting latency and cost on every duplicate or near-duplicate call.
"""

import time
from embedx import EmbedX

TEXTS = [
    "Transformers use self-attention to process sequences in parallel.",
    "Retrieval-augmented generation grounds LLM answers in real documents.",
    "Vector databases store embeddings and support similarity search.",
    "Semantic search finds documents by meaning, not just keywords.",
    "Fine-tuning adapts a pre-trained model to a downstream task.",
]

db = EmbedX(db_path="cost_demo.db")
db.clear()

# ── Round 1: cold adds (embeddings computed) ──────────────────────────────
print("Round 1 — cold adds (no cache):")
t0 = time.perf_counter()
for text in TEXTS:
    result = db.add(text)
    print(f"  [{result['status']:12}] {text[:55]}...")
cold_ms = (time.perf_counter() - t0) * 1000
print(f"  Total: {cold_ms:.0f}ms\n")

# ── Round 2: same texts again (exact cache hits) ──────────────────────────
print("Round 2 — identical texts (exact cache hits):")
t0 = time.perf_counter()
for text in TEXTS:
    result = db.add(text)
    print(f"  [{result['status']:12}] {text[:55]}...")
cached_ms = (time.perf_counter() - t0) * 1000
print(f"  Total: {cached_ms:.0f}ms\n")

# ── Round 3: paraphrased queries (semantic cache hits) ────────────────────
paraphrases = [
    "Self-attention lets transformers handle long-range dependencies.",
    "RAG combines retrieval with language model generation.",
]
print("Round 3 — paraphrased queries (semantic cache):")
for q in paraphrases:
    results = db.search(q, top_k=1)
    if results:
        print(f"  query : {q[:55]}")
        print(f"  match : [{results[0]['score']:.3f}] {results[0]['text'][:55]}...")

# ── Summary ───────────────────────────────────────────────────────────────
stats = db.get_stats()
speedup = cold_ms / cached_ms if cached_ms > 0 else float("inf")
print(f"\n  Cache hit rate : {stats['hit_rate'] * 100:.0f}%")
print(f"  Exact hits     : {stats['exact_hits']}")
print(f"  Cost saved     : ${stats['cost_saved_usd']:.6f}")
print(f"  Speedup        : {speedup:.1f}x faster on cache hits")

db.close()

import os
try:
    os.remove("cost_demo.db")
except OSError:
    pass
