"""EmbedX basic usage — copy-paste ready, zero config."""

from embedx import EmbedX

db = EmbedX()

db.add("AI is intelligence exhibited by machines")
db.add("Machine learning is a subset of artificial intelligence")
db.add("Python is a popular language for data science")

results = db.search("AI", top_k=2)
for r in results:
    print(f"[{r['score']:.3f}] {r['text']}")

db.close()
