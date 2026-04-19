"""OpenAI embedding provider."""

from __future__ import annotations

import os
from typing import Optional

from embedx.providers.base import BaseProvider, EmbeddingResult

_MODEL_COSTS: dict[str, float] = {
    "text-embedding-3-small": 0.00002 / 1000,
    "text-embedding-3-large": 0.00013 / 1000,
    "text-embedding-ada-002": 0.0001 / 1000,
}
_DEFAULT_MODEL = "text-embedding-3-small"
_DIMS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIProvider(BaseProvider):
    """OpenAI embedding via the openai SDK (lazy loaded)."""

    name = "openai"

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        api_key: Optional[str] = None,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client: Optional[object] = None

    def _load(self) -> None:
        if self._client is not None:
            return
        try:
            import openai  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "openai package is required.\n"
                "Install with: pip install embedx[openai]"
            ) from exc
        if not self._api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY env var or pass api_key=."
            )
        self._client = openai.OpenAI(api_key=self._api_key)

    @property
    def dimension(self) -> int:
        return _DIMS.get(self._model, 1536)

    def _count_tokens(self, text: str) -> int:
        try:
            import tiktoken  # type: ignore

            enc = tiktoken.encoding_for_model(self._model)
            return len(enc.encode(text))
        except Exception:
            return len(text.split())

    def embed(self, text: str) -> EmbeddingResult:
        self._load()
        response = self._client.embeddings.create(input=[text], model=self._model)
        vec = response.data[0].embedding
        tokens = response.usage.total_tokens
        cost = tokens * _MODEL_COSTS.get(self._model, 0.00002 / 1000)
        return EmbeddingResult(
            embedding=vec,
            token_count=tokens,
            cost_usd=cost,
            provider=self.name,
        )

    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        self._load()
        response = self._client.embeddings.create(input=texts, model=self._model)
        total_tokens = response.usage.total_tokens
        per_token_cost = _MODEL_COSTS.get(self._model, 0.00002 / 1000)
        results = []
        for i, item in enumerate(response.data):
            tokens = total_tokens // len(texts)
            results.append(
                EmbeddingResult(
                    embedding=item.embedding,
                    token_count=tokens,
                    cost_usd=tokens * per_token_cost,
                    provider=self.name,
                )
            )
        return results
