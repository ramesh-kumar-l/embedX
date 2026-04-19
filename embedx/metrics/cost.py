"""Cost estimation and savings tracking."""

from __future__ import annotations

from embedx.storage.sqlite_store import SQLiteStore

# Cost per 1k tokens for reference (OpenAI text-embedding-3-small default)
_DEFAULT_COST_PER_1K_TOKENS = 0.00002


class CostTracker:
    def __init__(self, store: SQLiteStore, cost_per_1k_tokens: float = _DEFAULT_COST_PER_1K_TOKENS) -> None:
        self._store = store
        self._cost_per_1k = cost_per_1k_tokens

    def record_embedding(self, token_count: int, cost_usd: float) -> None:
        self._store.increment_stat("total_tokens", token_count)
        self._store.increment_stat("total_cost_usd", cost_usd)
        self._store.increment_stat("total_requests")

    def record_cache_hit(self, token_count: int) -> None:
        saved = (token_count / 1000) * self._cost_per_1k
        self._store.increment_stat("saved_tokens", token_count)
        self._store.increment_stat("saved_cost_usd", saved)
        self._store.increment_stat("cache_hits")

    def get_summary(self) -> dict:
        stats = self._store.get_all_stats()
        total = stats.get("total_requests", 0) + stats.get("cache_hits", 0)
        hit_rate = stats.get("cache_hits", 0) / total if total > 0 else 0.0
        return {
            "total_requests": int(total),
            "cache_hits": int(stats.get("cache_hits", 0)),
            "exact_hits": int(stats.get("exact_hits", 0)),
            "semantic_hits": int(stats.get("semantic_hits", 0)),
            "hit_rate": round(hit_rate, 4),
            "total_tokens_used": int(stats.get("total_tokens", 0)),
            "tokens_saved": int(stats.get("saved_tokens", 0)),
            "total_cost_usd": round(stats.get("total_cost_usd", 0.0), 6),
            "cost_saved_usd": round(stats.get("saved_cost_usd", 0.0), 6),
        }
