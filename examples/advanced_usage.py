"""Advanced EmbedX usage: custom provider, ranking weights, metadata."""

from __future__ import annotations

import json

from embedx import EmbedX
from embedx.providers.local_model import LocalModelProvider
from embedx.ranking.scorer import RankingScorer

# Custom ranking weights (recall-focused)
db = EmbedX(
    db_path="advanced_demo.db",
    semantic_threshold=0.90,     # slightly more permissive cache
    semantic_weight=0.7,
    recency_weight=0.2,          # boost recently accessed docs
    frequency_weight=0.1,
)

# Add a knowledge base with metadata
knowledge = [
    ("Python is a high-level, dynamically typed programming language.", {"domain": "pl", "year": 1991}),
    ("JavaScript runs in the browser and enables interactive web pages.", {"domain": "web", "year": 1995}),
    ("Rust provides memory safety without a garbage collector.", {"domain": "pl", "year": 2010}),
    ("React is a declarative UI library maintained by Meta.", {"domain": "web", "year": 2013}),
    ("PostgreSQL is a powerful open-source relational database.", {"domain": "db", "year": 1996}),
    ("Redis is an in-memory key-value store often used for caching.", {"domain": "db", "year": 2009}),
    ("Docker packages applications into portable containers.", {"domain": "devops", "year": 2013}),
    ("Kubernetes orchestrates containerized workloads at scale.", {"domain": "devops", "year": 2014}),
]

print("Building knowledge base...")
for text, meta in knowledge:
    db.add(text, metadata=meta)

# Filter by metadata (client-side)
def search_by_domain(query: str, domain: str, top_k: int = 5) -> list[dict]:
    results = db.search(query, top_k=top_k * 2)
    return [r for r in results if r["metadata"].get("domain") == domain][:top_k]

print("\nSearching 'database performance' in domain=db:")
for r in search_by_domain("database performance", domain="db"):
    print(f"  [{r['score']:.3f}] {r['text'][:70]}")

print("\nBatch add with deduplication:")
new_docs = [
    "Python is a high-level, dynamically typed programming language.",  # duplicate
    "Go is a statically typed compiled language designed at Google.",
    "Elixir is a functional language built on the Erlang VM.",
]
results = db.add_batch(new_docs)
for text, res in zip(new_docs, results):
    print(f"  [{res['status']:12}] {text[:60]}...")

print("\nFull stats (JSON):")
stats = db.get_stats()
print(json.dumps(stats, indent=2))

db.close()

import os
try:
    os.remove("advanced_demo.db")
except OSError:
    pass
