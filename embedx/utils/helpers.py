"""Shared utility functions."""

from __future__ import annotations

import hashlib
import time
from typing import Any


def hash_text(text: str) -> str:
    """Stable SHA-256 hash of normalized text."""
    normalized = " ".join(text.strip().lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity (fallback when numpy unavailable)."""
    import numpy as np

    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


def now_ts() -> float:
    """Current Unix timestamp."""
    return time.time()


def serialize_vector(vec: list[float]) -> bytes:
    """Serialize a float list to bytes for SQLite BLOB storage."""
    import numpy as np

    return np.array(vec, dtype=np.float32).tobytes()


def deserialize_vector(data: bytes) -> list[float]:
    """Deserialize bytes back to float list."""
    import numpy as np

    return np.frombuffer(data, dtype=np.float32).tolist()


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def truncate(text: str, max_len: int = 80) -> str:
    return text if len(text) <= max_len else text[: max_len - 3] + "..."
