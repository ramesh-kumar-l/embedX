"""EmbedX CLI — embedx add / search / stats / eval / benchmark."""

from __future__ import annotations

import argparse
import json
import sys
import time


def _get_db(args: argparse.Namespace):
    from embedx.api.public import EmbedX

    return EmbedX(db_path=getattr(args, "db", "embedx.db"))


# ------------------------------------------------------------------
# Sub-command handlers
# ------------------------------------------------------------------

def cmd_add(args: argparse.Namespace) -> None:
    with _get_db(args) as db:
        result = db.add(
            args.text,
            metadata={"source": "cli"},
            namespace=args.namespace,
            importance=args.importance,
        )
    status = result["status"]
    icon = {"added": "+", "exact_hit": "=", "semantic_hit": "~"}.get(status, "?")
    print(f"[{icon}] {status}  id={result['id']}  ({result['latency_ms']:.1f}ms)")


def cmd_search(args: argparse.Namespace) -> None:
    with _get_db(args) as db:
        results = db.search(args.query, top_k=args.top_k, hybrid=args.hybrid, namespace=args.namespace)
    if not results:
        print("No results found.")
        return
    for i, r in enumerate(results, 1):
        score = r["score"]
        text = r["text"]
        print(f"  {i}. [{score:.3f}] {text}")
        if args.verbose and r["metadata"]:
            print(f"       metadata: {r['metadata']}")


def cmd_stats(args: argparse.Namespace) -> None:
    with _get_db(args) as db:
        stats = db.get_stats()

    if args.json:
        print(json.dumps(stats, indent=2))
        return

    hit_rate = stats.get("hit_rate", 0.0)
    saved = stats.get("cost_saved_usd", 0.0)
    lat = stats.get("latency", {})

    print("\n  EmbedX Statistics")
    print("  " + "-" * 36)
    print(f"  Documents stored    : {stats.get('document_count', 0)}")
    print(f"  Total requests      : {stats.get('total_requests', 0)}")
    print(f"  Cache hit rate      : {hit_rate * 100:.1f}%")
    print(f"    Exact hits        : {stats.get('exact_hits', 0)}")
    print(f"    Semantic hits     : {stats.get('semantic_hits', 0)}")
    print(f"  Cost saved          : ${saved:.4f}")
    print(f"  Total cost          : ${stats.get('total_cost_usd', 0.0):.6f}")
    print(f"  Tokens saved        : {stats.get('tokens_saved', 0):,}")
    if lat:
        print(f"  Avg latency         : {lat.get('avg_ms', 0.0):.1f}ms")
        print(f"  P95 latency         : {lat.get('p95_ms', 0.0):.1f}ms")
    print()


def cmd_eval(args: argparse.Namespace) -> None:
    from embedx.eval.dataset import load_dataset
    from embedx.eval.evaluator import evaluate

    dataset = load_dataset(args.dataset)
    print(f"  Running eval: {dataset.name} ({len(dataset.cases)} cases)...")

    with _get_db(args) as db:
        metrics = evaluate(db, dataset, top_k=args.top_k)

    if args.json:
        print(json.dumps(
            {
                "dataset": metrics.dataset_name,
                "num_cases": metrics.num_cases,
                "recall@1": metrics.recall_at_1,
                "recall@3": metrics.recall_at_3,
                "recall@5": metrics.recall_at_5,
                "keyword_match": metrics.keyword_match,
            },
            indent=2,
        ))
        return

    print(f"\n  Eval Results: {metrics.dataset_name}")
    print("  " + "-" * 36)
    print(f"  Cases        : {metrics.num_cases}")
    print(f"  Recall@1     : {metrics.recall_at_1 * 100:.1f}%")
    print(f"  Recall@3     : {metrics.recall_at_3 * 100:.1f}%")
    print(f"  Recall@5     : {metrics.recall_at_5 * 100:.1f}%")
    print(f"  Keyword match: {metrics.keyword_match * 100:.1f}%")
    if args.verbose:
        print("\n  Per-case results:")
        for c in metrics.per_case:
            print(f"    Q: {c['query'][:60]}")
            print(f"       R@1={c['recall@1']:.2f}  kw={c['keyword_match']:.2f}")
    print()


def cmd_trace(args: argparse.Namespace) -> None:
    with _get_db(args) as db:
        out = db.search(args.query, top_k=args.top_k, hybrid=args.hybrid, trace=True)
    results, trace = out

    print(f'\nQuery: "{trace["query"]}"')
    print(f'Mode : {trace["mode"]}')
    if trace.get("namespace"):
        print(f'Namespace: {trace["namespace"]}')

    cache = trace["cache"]
    print("\nCache:")
    print(f'  L1 : {"hit" if cache["l1_hit"] else "miss"}')
    print(f'  L2 : {"hit" if cache["l2_hit"] else "miss"}')

    t = trace["timings"]
    print("\nTimings:")
    if t["embedding_ms"] > 0:
        print(f'  embedding : {t["embedding_ms"]:.1f}ms')
    if t["vector_ms"] > 0:
        print(f'  vector    : {t["vector_ms"]:.1f}ms')
    if t["bm25_ms"] > 0:
        print(f'  bm25      : {t["bm25_ms"]:.1f}ms')
    if t["fusion_ms"] > 0:
        print(f'  fusion    : {t["fusion_ms"]:.1f}ms')
    if t["ranking_ms"] > 0:
        print(f'  ranking   : {t["ranking_ms"]:.1f}ms')
    print(f'  total     : {t["total_ms"]:.1f}ms')

    ret = trace["retrieval"]
    print("\nRetrieval:")
    print(f'  semantic candidates : {ret["semantic_candidates"]}')
    if ret["bm25_candidates"] > 0:
        print(f'  bm25 candidates     : {ret["bm25_candidates"]}')
    print(f'  final candidates    : {ret["final_candidates"]}')

    ranking = trace["ranking"]
    if ranking:
        print(f'\nTop {min(len(ranking), args.top_k)} result(s):')
        for i, entry in enumerate(ranking[: args.top_k], 1):
            print(f'  {i}. doc_id={entry["doc_id"]}  score={entry["final_score"]:.4f}')
            print(f'       semantic={entry["semantic"]:.4f}  '
                  f'bm25={entry["bm25"]:.4f}  '
                  f'recency={entry["recency"]:.4f}  '
                  f'importance={entry["importance"]:.4f}')
            if i <= len(results):
                text = results[i - 1]["text"]
                print(f'       text: {text[:80]}{"..." if len(text) > 80 else ""}')
    print()


def cmd_serve(args: argparse.Namespace) -> None:
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn is required. Install with: pip install embedx[server]", file=sys.stderr)
        sys.exit(1)

    import os
    os.environ.setdefault("EMBEDX_DB", getattr(args, "db", "embedx.db"))
    print(f"  Starting EmbedX server on http://{args.host}:{args.port}")
    uvicorn.run(
        "server.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def cmd_benchmark(args: argparse.Namespace) -> None:
    import random
    import string

    from embedx.api.public import EmbedX

    print("\n  EmbedX Benchmark")
    print("  " + "-" * 40)

    sentences = [
        "Machine learning enables computers to learn from data without explicit programming.",
        "Natural language processing allows machines to understand human language.",
        "Deep neural networks can approximate any continuous function.",
        "Transformers revolutionized NLP by using attention mechanisms.",
        "Retrieval-augmented generation combines search with language model generation.",
        "Vector databases enable efficient similarity search over high-dimensional data.",
        "Semantic search understands intent and context, not just keywords.",
        "Embeddings map text to dense numerical representations in semantic space.",
        "Cosine similarity measures the angle between two vectors in n-dimensional space.",
        "Large language models are trained on vast corpora using self-supervised learning.",
    ]

    db_path = "/tmp/_embedx_bench.db" if sys.platform != "win32" else "_embedx_bench.db"

    # Baseline: no cache (simulate fresh adds)
    t_start = time.perf_counter()
    with EmbedX(db_path=db_path) as db:
        db.clear()
        for s in sentences:
            db.add(s)
        t_cold = (time.perf_counter() - t_start) * 1000

    # With cache: repeat same queries
    t_start = time.perf_counter()
    with EmbedX(db_path=db_path) as db:
        for s in sentences:
            db.search(s, top_k=3)
        t_warm = (time.perf_counter() - t_start) * 1000

    # With semantic cache
    t_start = time.perf_counter()
    with EmbedX(db_path=db_path) as db:
        for s in sentences:
            db.add(s)  # all should be cache hits
        stats = db.get_stats()
        t_cached = (time.perf_counter() - t_start) * 1000

    saving_pct = max(0.0, (1 - t_cached / t_cold) * 100) if t_cold > 0 else 0.0

    print(f"  Documents         : {len(sentences)}")
    print(f"  Cold add (no cache): {t_cold:.0f}ms")
    print(f"  Search (warm)     : {t_warm:.0f}ms")
    print(f"  Cache re-add      : {t_cached:.0f}ms")
    print(f"  Cache hit rate    : {stats.get('hit_rate', 0) * 100:.0f}%")
    print(f"  Speedup           : {saving_pct:.0f}% faster\n")

    import os
    try:
        os.remove(db_path)
    except OSError:
        pass


# ------------------------------------------------------------------
# Argument parser
# ------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="embedx",
        description="EmbedX — cost-aware semantic memory for LLM apps",
    )
    parser.add_argument("--db", default="embedx.db", help="SQLite database path")

    sub = parser.add_subparsers(dest="command")

    # add
    p_add = sub.add_parser("add", help="Add a document")
    p_add.add_argument("text", help="Text to add")
    p_add.add_argument("--namespace", default="default", help="Logical namespace (default: 'default')")
    p_add.add_argument("--importance", type=float, default=0.5, help="Importance score 0–1 (default: 0.5)")

    # search
    p_search = sub.add_parser("search", help="Semantic search")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("-k", "--top-k", type=int, default=5)
    p_search.add_argument("-v", "--verbose", action="store_true")
    p_search.add_argument("--hybrid", action="store_true", help="Use hybrid retrieval (BM25 + semantic)")
    p_search.add_argument("--namespace", default=None, help="Restrict search to a namespace")

    # trace
    p_trace = sub.add_parser("trace", help="Search with full execution trace")
    p_trace.add_argument("query", help="Search query")
    p_trace.add_argument("-k", "--top-k", type=int, default=5)
    p_trace.add_argument("--hybrid", action="store_true", help="Use hybrid retrieval")

    # stats
    p_stats = sub.add_parser("stats", help="Show cache statistics")
    p_stats.add_argument("--json", action="store_true")

    # eval
    p_eval = sub.add_parser("eval", help="Run evaluation dataset")
    p_eval.add_argument("dataset", help="Path to .yaml or .json dataset")
    p_eval.add_argument("-k", "--top-k", type=int, default=5)
    p_eval.add_argument("--json", action="store_true")
    p_eval.add_argument("-v", "--verbose", action="store_true")

    # benchmark
    sub.add_parser("benchmark", help="Run built-in latency benchmark")

    # serve
    p_serve = sub.add_parser("serve", help="Start the FastAPI HTTP server")
    p_serve.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    p_serve.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    p_serve.add_argument("--reload", action="store_true", help="Auto-reload on code changes")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    handlers = {
        "add": cmd_add,
        "search": cmd_search,
        "trace": cmd_trace,
        "stats": cmd_stats,
        "eval": cmd_eval,
        "benchmark": cmd_benchmark,
        "serve": cmd_serve,
    }

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    try:
        handler(args)
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
