"""Local embedding provider using sentence-transformers (lazy loaded)."""

from __future__ import annotations

from typing import Optional

from embedx.providers.base import BaseProvider, EmbeddingResult

_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_COST_PER_TOKEN = 0.0  # local = free


class LocalModelProvider(BaseProvider):
    """Offline embedding via sentence-transformers. Lazy-loads on first use."""

    name = "local"

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._model: Optional[object] = None
        self._dim: Optional[int] = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for local embeddings.\n"
                "Install with: pip install embedx[embeddings]"
            ) from exc
        self._model = SentenceTransformer(self._model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        self._load()
        assert self._dim is not None
        return self._dim

    def embed(self, text: str) -> EmbeddingResult:
        self._load()
        vec = self._model.encode(text, normalize_embeddings=True).tolist()
        tokens = len(text.split())
        return EmbeddingResult(
            embedding=vec,
            token_count=tokens,
            cost_usd=0.0,
            provider=self.name,
        )

    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        self._load()
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        results = []
        for text, vec in zip(texts, vecs):
            tokens = len(text.split())
            results.append(
                EmbeddingResult(
                    embedding=vec.tolist(),
                    token_count=tokens,
                    cost_usd=0.0,
                    provider=self.name,
                )
            )
        return results
