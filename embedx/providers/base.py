"""Abstract base for embedding providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EmbeddingResult:
    embedding: list[float]
    token_count: int
    cost_usd: float
    provider: str
    cached: bool = False


class BaseProvider(ABC):
    name: str = "base"

    @abstractmethod
    def embed(self, text: str) -> EmbeddingResult:
        """Embed a single text string."""
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Embed a list of strings."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimensionality."""
        ...
