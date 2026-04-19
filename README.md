# EmbedX

**Stop paying to embed the same text twice.**

EmbedX is an **offline-first semantic memory and retrieval cache for LLM applications**. It helps AI apps **cut embedding cost**, **reuse semantically similar content**, and **ship persistent memory** with a simple Python API, CLI, and HTTP server.

**Why developers star it:**

- Exact cache for repeated text
- Semantic cache for near-duplicates
- Persistent SQLite-backed memory
- Hybrid retrieval with semantic + BM25 search
- FastAPI server, CLI, benchmarks, and VS Code integration

```bash
pip install embedx[embeddings]
```

```python
from embedx import EmbedX

db = EmbedX()
db.add("AI is intelligence exhibited by machines")
results = db.search("What is artificial intelligence?")
```

**Typical repeated-workload result**

```text
Without cache : 92ms avg/doc
With EmbedX   : 0.6ms avg/doc
Savings       : 99%
```

---

## What is EmbedX?

EmbedX is a production-grade Python library that gives LLM applications **persistent semantic memory** with **automatic cost optimization**.

Every time your app embeds the same (or similar) text, it wastes tokens and money. EmbedX intercepts those calls with a two-layer cache:

- **L1 Exact cache** — hash lookup, zero compute, sub-millisecond
- **L2 Semantic cache** — similarity threshold, skips embedding near-duplicates

Results are ranked by a composite scorer (semantic similarity + recency + frequency) so the most relevant and recently-used documents surface first.

---

## Why EmbedX?

| Problem | EmbedX solution |
|---|---|
| Re-embedding identical text | L1 exact cache — O(1) hash lookup |
| Re-embedding paraphrased text | L2 semantic cache — configurable threshold |
| No persistent memory across sessions | SQLite store — survives restarts |
| Expensive OpenAI embedding calls | Local model by default — free |
| Keyword-only search | Dense embedding retrieval |
| Unknown cost impact | Built-in cost tracking and stats |

Typical result with `all-MiniLM-L6-v2`:

```
Without cache : 92ms avg/doc
With EmbedX   : 0.6ms avg/doc
Savings       : 99%
```

---

## Quick Start (< 30 seconds)

### Install

```bash
# Offline-first (recommended)
pip install embedx[embeddings]

# With HTTP server
pip install embedx[server]

# OpenAI provider
pip install embedx[openai]

# Large-scale index
pip install embedx[faiss]

# Everything
pip install embedx[all]
```

### Python API

```python
from embedx import EmbedX

db = EmbedX()                          # auto-creates embedx.db

# Add documents
db.add("Python is a dynamically typed programming language.")
db.add("Rust provides memory safety without a garbage collector.")

# Search
results = db.search("safe systems programming", top_k=3)
for r in results:
    print(f"[{r['score']:.3f}] {r['text']}")

# Batch add
db.add_batch(["doc one", "doc two", "doc three"])

# Stats
print(db.get_stats())
```

### CLI

```bash
embedx add "The capital of France is Paris"
embedx search "What is the capital of France?"
embedx stats
embedx benchmark
embedx eval examples/eval_dataset.yaml
embedx serve                           # start HTTP API server
```

---

## HTTP API Server

EmbedX ships a FastAPI server for integrations, VS Code, and multi-process use.

### Start

```bash
pip install embedx[server]
embedx serve                           # http://127.0.0.1:8000
embedx serve --host 0.0.0.0 --port 9000
```

Set the database path via environment variable:

```bash
EMBEDX_DB=/data/my.db embedx serve
```

### Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/add` | Add a document |
| `POST` | `/search` | Semantic search |
| `GET` | `/stats` | Cache and cost statistics |
| `GET` | `/health` | Liveness probe |

### Examples

```bash
# Add
curl -s -X POST http://localhost:8000/add \
  -H "Content-Type: application/json" \
  -d '{"text": "Transformers use self-attention"}'

# Search
curl -s -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "attention mechanisms", "top_k": 3}'

# Stats
curl -s http://localhost:8000/stats
```

### Response shapes

```json
// POST /add
{ "status": "added", "id": 1, "latency_ms": 45.2 }

// POST /search
[{ "text": "...", "score": 0.92, "semantic_score": 0.91, "metadata": {}, "use_count": 1 }]

// GET /stats
{ "document_count": 42, "hit_rate": 0.73, "cost_saved_usd": 0.0021, ... }
```

---

## Hybrid Retrieval

By default EmbedX uses **semantic (vector) search** alone. Hybrid retrieval adds a **BM25 keyword index** alongside it, fuses the scores, and optionally applies a lightweight heuristic reranker — improving recall for exact-term queries and mixed natural-language/keyword searches.

No extra dependencies are required. The BM25 index is built in pure Python over the same SQLite store.

### When to use

| Situation | Recommendation |
|---|---|
| Queries are natural-language only | Default semantic search is sufficient |
| Queries contain specific terms, names, or codes | Enable hybrid retrieval |
| Mixed corpus (prose + technical docs) | Enable hybrid retrieval |

### Enable per query

```python
results = db.search("transformer self-attention", top_k=5, hybrid=True)
```

### Enable for all searches on an instance

```python
db = EmbedX(hybrid=True)
results = db.search("attention mechanism")   # hybrid by default
```

### Tune fusion weights

```python
db = EmbedX(
    hybrid=True,
    hybrid_semantic_weight=0.6,   # default 0.7
    hybrid_keyword_weight=0.4,    # default 0.3
)
```

### CLI

```bash
embedx search "transformer attention" --hybrid
```

### HTTP API

```bash
curl -s -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "transformer attention", "top_k": 5, "hybrid": true}'
```

### How it works

```
query
  ├─ Semantic search  →  vector candidates  (cosine similarity)
  └─ BM25 index       →  keyword candidates (term frequency / IDF)
         │
         ▼
   Score fusion  (max-normalized weighted sum)
         │
         ▼
   Heuristic reranker  (exact-phrase bonus + token coverage + length)
         │
         ▼
   RankingScorer  (semantic + recency + frequency composite)
         │
         ▼
   Results  (same format as standard search)
```

The BM25 index is lazy-built on first hybrid search and cached in memory. It is automatically invalidated when documents are added.

---

## Observability / Trace

EmbedX includes a zero-overhead trace layer that explains exactly how a query was processed — which cache layer was hit, how long each stage took, how many candidates each retriever returned, and how each result was scored.

Tracing is **opt-in and disabled by default**. When `trace=False` there is no measurable overhead.

### Python API

```python
results, trace = db.search("What is AI?", trace=True)

# trace structure:
# {
#   "query": "What is AI?",
#   "mode": "semantic",           # or "hybrid"
#   "namespace": None,
#   "cache": { "l1_hit": false, "l2_hit": false },
#   "timings": {
#     "embedding_ms": 8.3,
#     "vector_ms": 1.2,
#     "bm25_ms": 0.0,
#     "fusion_ms": 0.0,
#     "ranking_ms": 0.1,
#     "total_ms": 10.4
#   },
#   "retrieval": {
#     "semantic_candidates": 5,
#     "bm25_candidates": 0,
#     "final_candidates": 5
#   },
#   "ranking": [
#     { "doc_id": 1, "semantic": 0.82, "bm25": 0.0,
#       "recency": 0.99, "importance": 0.5, "final_score": 0.77 },
#     ...
#   ]
# }
```

When `trace=False` (the default), `search()` returns `list[dict]` as usual — the return type is unchanged.

### CLI

```bash
embedx trace "What is AI?"
embedx trace "transformer attention" --hybrid
```

Example output:

```
Query: "What is AI?"
Mode : semantic

Cache:
  L1 : miss
  L2 : miss

Timings:
  embedding : 8.3ms
  vector    : 1.2ms
  ranking   : 0.1ms
  total     : 10.4ms

Retrieval:
  semantic candidates : 5
  final candidates    : 5

Top 5 result(s):
  1. doc_id=3  score=0.7734
       semantic=0.8200  bm25=0.0000  recency=0.9900  importance=0.5000
       text: AI is intelligence exhibited by machines...
```

### HTTP API

Add `"trace": true` to a `/search` request:

```bash
curl -s -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What is AI?", "top_k": 3, "trace": true}'
```

Response:

```json
{
  "results": [{ "text": "...", "score": 0.77, ... }],
  "trace": { "query": "What is AI?", "mode": "semantic", ... }
}
```

Without `"trace"`, the existing `[{...}, ...]` array response is preserved unchanged.

---

## Memory and Namespaces

EmbedX upgrades from a search tool to a **memory system** by giving every stored document a **namespace** (logical group) and an **importance** score (priority weight).

### What are namespaces?

A namespace is a string label that groups documents together. Documents in different namespaces are invisible to each other during search. Use namespaces to:

- Isolate per-user or per-project memory
- Separate domain-specific knowledge bases
- Tag documents by context (`"work"`, `"personal"`, `"project-x"`)

### When to use importance

Importance (0.0–1.0) lets you signal that some memories matter more. When `importance_weight > 0` on the EmbedX instance, higher-importance documents are boosted in ranking even if their semantic similarity is equal.

Use cases:
- Pin critical facts (`importance=0.9`) so they appear first
- Demote boilerplate or noise (`importance=0.1`)
- Leave at default (`0.5`) when all documents are equal

### Python API

```python
from embedx import EmbedX

# Enable importance-aware ranking
db = EmbedX(importance_weight=0.1)

# Add to a namespace with importance
db.add("Critical system alert: database is down", namespace="ops", importance=0.95)
db.add("Weekly standup notes", namespace="ops", importance=0.3)
db.add("Python tutorial: list comprehensions", namespace="learning")

# Search within a namespace
results = db.search("database issue", namespace="ops")

# Search all namespaces (default behavior — unchanged)
results = db.search("Python")
```

### CLI

```bash
# Add with namespace and importance
embedx add "critical info" --namespace project-x --importance 0.9

# Search within a namespace
embedx search "query" --namespace project-x

# No flags — existing behavior, all namespaces
embedx add "regular text"
embedx search "query"
```

### HTTP API

```bash
# POST /add
curl -s -X POST http://localhost:8000/add \
  -H "Content-Type: application/json" \
  -d '{"text": "critical info", "namespace": "project-x", "importance": 0.9}'

# POST /search (namespace-restricted)
curl -s -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "info", "top_k": 5, "namespace": "project-x"}'
```

### Ranking with importance

When `importance_weight > 0`, the ranking formula becomes:

```
score = semantic_score  × w_semantic
      + recency_score   × w_recency
      + frequency_score × w_frequency
      + importance_score × w_importance
```

All weights are normalized to sum to 1. Recommended starting point:

```python
db = EmbedX(
    semantic_weight=0.7,
    recency_weight=0.2,
    importance_weight=0.1,
)
```

### Backward compatibility

- **No `namespace` or `importance` provided** → defaults to `namespace="default"`, `importance=0.5`
- **`importance_weight=0.0`** (the default) → importance has zero effect on scores; identical to pre-memory behavior
- **No `namespace` in search** → searches all namespaces, same as before

Existing databases are automatically migrated on next startup. No data is lost.

---

## VS Code Extension

The extension connects to the running EmbedX server and exposes three commands.

### Setup

1. Start the server: `embedx serve`
2. Open VS Code → Extensions → install from `vscode-extension/`
3. Set `embedx.serverUrl` in settings if the server is not on `127.0.0.1:8000`

### Commands

| Command | What it does |
|---|---|
| `EmbedX: Search` | Input box → quick-pick results, copy to clipboard |
| `EmbedX: Add Selection` | Adds highlighted text to the running EmbedX instance |
| `EmbedX: Show Stats` | Modal with cache hit rate, cost saved, latency |

### Settings

```json
{
  "embedx.serverUrl": "http://127.0.0.1:8000",
  "embedx.defaultTopK": 5
}
```

### Build

```bash
cd vscode-extension
npm install
npm run compile
```

---

## Examples

All examples run without configuration (offline, local model).

```bash
pip install embedx[embeddings]
python examples/basic_usage.py
python examples/cost_savings_demo.py
python examples/semantic_search_demo.py
```

**`basic_usage.py`** — minimal add + search in 10 lines.

**`cost_savings_demo.py`** — cold adds vs cached adds, prints speedup and cost saved.

**`semantic_search_demo.py`** — 12-document knowledge base across four domains, shows meaning-based retrieval.

---

## Benchmarks

```bash
python benchmarks/run_benchmark.py    # with cache vs without cache
python benchmarks/latency_benchmark.py  # full phase breakdown
```

Typical output (`run_benchmark.py`):

```
  [Without cache]  cold embed every document
  Time : 1840ms  (92.0ms/doc)

  [With EmbedX]    exact cache on repeated content
  Time : 12ms  (0.6ms/doc)
  Hit rate : 100%

  Without cache : 1840ms
  With EmbedX   : 12ms
  Cost saved    : 99%
  Speedup       : 153.3x
```

---

## CLI Reference

| Command | Description |
|---|---|
| `embedx add "text"` | Add a document |
| `embedx search "query"` | Semantic search |
| `embedx stats` | Cache and cost statistics |
| `embedx eval dataset.yaml` | Run evaluation dataset |
| `embedx benchmark` | Built-in latency benchmark |
| `embedx serve` | Start the FastAPI HTTP server |

### Options

```bash
embedx --db /path/to/custom.db search "query"   # custom DB path
embedx search "query" -k 10                     # top-10 results
embedx search "query" --verbose                 # show metadata
embedx search "query" --hybrid                  # hybrid BM25 + semantic
embedx stats --json                             # JSON output
embedx eval dataset.yaml --verbose              # per-case results
embedx serve --host 0.0.0.0 --port 9000 --reload
```

---

## Python API Reference

### `EmbedX(db_path, provider, semantic_threshold, ...)`

| Parameter | Default | Description |
|---|---|---|
| `db_path` | `"embedx.db"` | SQLite file path |
| `provider` | `LocalModelProvider()` | Embedding provider |
| `semantic_threshold` | `0.92` | L2 cache similarity cutoff |
| `semantic_weight` | `0.8` | Ranking: semantic score weight |
| `recency_weight` | `0.1` | Ranking: recency weight |
| `frequency_weight` | `0.1` | Ranking: frequency weight |
| `importance_weight` | `0.0` | Ranking: importance score weight (0 = disabled) |
| `use_faiss` | `False` | Use FAISS index (requires `pip install embedx[faiss]`) |
| `hybrid` | `False` | Enable hybrid retrieval (BM25 + semantic) for all searches |
| `hybrid_semantic_weight` | `0.7` | Semantic score weight in hybrid fusion |
| `hybrid_keyword_weight` | `0.3` | BM25 keyword score weight in hybrid fusion |

### Methods

```python
db.add(text, metadata=None, namespace="default", importance=0.5)   # → {"status", "id", "latency_ms"}
db.search(query, top_k=5, hybrid=False, namespace=None)            # → [{"text", "score", "metadata", ...}]
db.add_batch(texts, metadata=None)                                 # → [{"status", "id"}]
db.get_stats()                                                     # → dict (JSON-serializable)
db.rebuild_index()                                                 # Force index rebuild from DB
db.clear()                                                         # Delete all documents
db.close()                                                         # Close DB connection
```

---

## Embedding Providers

### Local (default, offline, free)

```python
from embedx import EmbedX
from embedx.providers.local_model import LocalModelProvider

db = EmbedX(provider=LocalModelProvider("all-MiniLM-L6-v2"))
```

Requires `pip install embedx[embeddings]`.

### OpenAI

```python
import os
from embedx import EmbedX
from embedx.providers.openai_provider import OpenAIProvider

db = EmbedX(provider=OpenAIProvider(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY"),
))
```

Requires `pip install embedx[openai]`.

### Custom Provider

```python
from embedx.providers.base import BaseProvider, EmbeddingResult

class MyProvider(BaseProvider):
    name = "my_provider"

    def embed(self, text: str) -> EmbeddingResult:
        vec = my_embedding_function(text)
        return EmbeddingResult(embedding=vec, token_count=len(text.split()), cost_usd=0.0, provider=self.name)

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]

    @property
    def dimension(self):
        return 768

db = EmbedX(provider=MyProvider())
```

---

## Evaluation

Create a dataset (YAML or JSON):

```yaml
# dataset.yaml
name: "My Eval"
corpus:
  - "Document one about machine learning."
  - "Document two about databases."
cases:
  - query: "What is ML?"
    expected: ["machine learning"]
  - query: "Where is data stored?"
    expected: ["databases"]
```

Run:

```bash
embedx eval dataset.yaml
```

```
  Eval Results: My Eval
  ------------------------------------
  Cases        : 2
  Recall@1     : 100.0%
  Recall@3     : 100.0%
  Recall@5     : 100.0%
  Keyword match: 75.0%
```

---

## Architecture

```
Client (Python API / CLI / HTTP)
  └─ Smart Cache Layer
       ├─ L1: Exact Cache     (hash lookup, O(1))
       └─ L2: Semantic Cache  (cosine similarity, threshold=0.92)
            └─ Embedding Provider (local / OpenAI / custom)
                 └─ Storage Layer (SQLite WAL)
                      └─ Index Layer (NumPy / optional FAISS)
                           └─ Ranking Engine (semantic + recency + frequency)
                                └─ Metrics (cost, latency, hit rate)

FastAPI Server (server/main.py)
  └─ Thin wrapper → EmbedX instance

VS Code Extension (vscode-extension/)
  └─ HTTP → FastAPI Server → EmbedX
```

### Ranking Formula

```
score = semantic_score × 0.8
      + recency_score  × 0.1
      + frequency_score × 0.1
```

Weights are configurable per-instance.

---

## Limitations

- **Index size**: The default NumPy index is linear scan — suitable for ~50k records. Use `use_faiss=True` for larger corpora.
- **Concurrency**: Single-writer SQLite. Safe for multi-read, single-writer workloads. Not designed for high-concurrency servers.
- **Embedding quality**: Default `all-MiniLM-L6-v2` is fast and good but not state-of-the-art. Swap the provider for better accuracy.
- **Local model download**: First run downloads ~90MB model. Subsequent runs are offline.

---

## Roadmap

- [x] HTTP service mode (FastAPI)
- [x] VS Code extension
- [x] Namespaces and memory importance layer
- [ ] Async API (`asyncio` support)
- [ ] HNSW index (pure Python, no FAISS dependency)
- [ ] Streaming batch ingestion
- [ ] LangChain / LlamaIndex integration

---

## License

Apache 2.0 — see [LICENSE](LICENSE)
