"""EmbedX Quickstart — works offline, zero config."""

from embedx import EmbedX

db = EmbedX()

# Add documents
texts = [
    "Machine learning enables systems to learn from data automatically.",
    "Natural language processing allows computers to understand human text.",
    "Deep learning uses neural networks with many layers to learn representations.",
    "Transformers are attention-based models that changed NLP benchmarks.",
    "Vector databases enable fast approximate nearest-neighbor search.",
    "Semantic search understands the meaning behind a query, not just keywords.",
]

print("Adding documents...")
for text in texts:
    result = db.add(text)
    print(f"  [{result['status']:12}] {text[:60]}...")

print("\nSearching for 'how do neural networks learn?'")
results = db.search("how do neural networks learn?", top_k=3)
for i, r in enumerate(results, 1):
    print(f"  {i}. [{r['score']:.3f}] {r['text'][:70]}...")

print("\nCache hit demonstration (second add is free):")
result = db.add("Machine learning enables systems to learn from data automatically.")
print(f"  Status: {result['status']} (no embedding computed)")

print("\nStats:")
stats = db.get_stats()
print(f"  Documents      : {stats['document_count']}")
print(f"  Cache hit rate : {stats['hit_rate'] * 100:.0f}%")
print(f"  Avg latency    : {stats['latency']['avg_ms']:.1f}ms")

db.close()
