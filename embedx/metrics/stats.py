"""Latency and throughput statistics."""

from __future__ import annotations

import time
from collections import deque
from typing import Deque


class LatencyTracker:
    """Rolling window latency tracker (last N operations)."""

    def __init__(self, window: int = 1000) -> None:
        self._window = window
        self._latencies: Deque[float] = deque(maxlen=window)

    def record(self, latency_ms: float) -> None:
        self._latencies.append(latency_ms)

    def summary(self) -> dict:
        if not self._latencies:
            return {"count": 0, "avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
        import numpy as np

        arr = np.array(list(self._latencies))
        return {
            "count": len(arr),
            "avg_ms": round(float(arr.mean()), 2),
            "p50_ms": round(float(np.percentile(arr, 50)), 2),
            "p95_ms": round(float(np.percentile(arr, 95)), 2),
            "p99_ms": round(float(np.percentile(arr, 99)), 2),
        }
