"""Shared test fixtures."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Generator

import pytest

from embedx.api.public import EmbedX
from embedx.providers.base import BaseProvider, EmbeddingResult
from embedx.storage.sqlite_store import SQLiteStore


class DummyProvider(BaseProvider):
    """Deterministic embeddings using character frequency vectors."""

    name = "dummy"

    def embed(self, text: str) -> EmbeddingResult:
        vec = self._char_vec(text)
        return EmbeddingResult(embedding=vec, token_count=len(text.split()), cost_usd=0.0001, provider="dummy")

    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        return [self.embed(t) for t in texts]

    @property
    def dimension(self) -> int:
        return 128

    @staticmethod
    def _char_vec(text: str) -> list[float]:
        import numpy as np

        vec = np.zeros(128, dtype=np.float32)
        for ch in text.lower():
            idx = ord(ch) % 128
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.tolist()


@pytest.fixture
def tmp_db(tmp_path: Path) -> Generator[str, None, None]:
    yield str(tmp_path / "test.db")


@pytest.fixture
def db(tmp_db: str) -> Generator[EmbedX, None, None]:
    with EmbedX(db_path=tmp_db, provider=DummyProvider()) as embedx:
        yield embedx


@pytest.fixture
def store(tmp_db: str) -> Generator[SQLiteStore, None, None]:
    s = SQLiteStore(tmp_db)
    yield s
    s.close()
