"""
EmbedX semantic search demo with a realistic knowledge base.

Demonstrates that search understands meaning — not just keyword overlap.
"""

from embedx import EmbedX

KNOWLEDGE_BASE = [
    ("Python is a dynamically typed, high-level programming language.", {"domain": "languages"}),
    ("Rust provides memory safety through ownership without a garbage collector.", {"domain": "languages"}),
    ("Go is a statically typed compiled language designed for simplicity and concurrency.", {"domain": "languages"}),
    ("PostgreSQL is a powerful open-source relational database system.", {"domain": "databases"}),
    ("Redis is an in-memory key-value store used for caching and pub/sub.", {"domain": "databases"}),
    ("MongoDB stores data as flexible JSON-like documents.", {"domain": "databases"}),
    ("Docker packages applications into lightweight portable containers.", {"domain": "devops"}),
    ("Kubernetes orchestrates containerized workloads across clusters.", {"domain": "devops"}),
    ("GitHub Actions automates CI/CD workflows directly from a repository.", {"domain": "devops"}),
    ("React is a declarative component-based UI library maintained by Meta.", {"domain": "frontend"}),
    ("Vue.js is a progressive JavaScript framework for building user interfaces.", {"domain": "frontend"}),
    ("Next.js is a React framework that adds server-side rendering and routing.", {"domain": "frontend"}),
]

db = EmbedX(db_path="semantic_demo.db")
db.clear()

print("Building knowledge base...")
for text, meta in KNOWLEDGE_BASE:
    db.add(text, metadata=meta)
print(f"  {len(KNOWLEDGE_BASE)} documents indexed\n")

QUERIES = [
    ("safe systems programming language", "languages"),
    ("store data without fixed schema", "databases"),
    ("deploy apps with containers", "devops"),
    ("build interactive web UI", "frontend"),
]

for query, expected_domain in QUERIES:
    results = db.search(query, top_k=3)
    print(f"Query : \"{query}\"")
    for r in results:
        domain = (r["metadata"] or {}).get("domain", "?")
        match = "✓" if domain == expected_domain else " "
        print(f"  {match} [{r['score']:.3f}] [{domain:9}] {r['text'][:65]}")
    print()

db.close()

import os
try:
    os.remove("semantic_demo.db")
except OSError:
    pass
