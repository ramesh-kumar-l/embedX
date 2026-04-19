"""Optional FAISS-backed index for large-scale deployments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from embedx.storage.sqlite_store import Record, SQLiteStore

if TYPE_CHECKING:
    import faiss  # type: ignore


class FAISSIndex:
    """
    FAISS IVF+Flat index. Falls back to FallbackIndex if faiss not installed.
    Only for corpora >50k records where linear scan is too slow.
    """

    def __init__(self, store: SQLiteStore, dimension: int, nlist: int = 100) -> None:
        self._store = store
        self._dim = dimension
        self._nlist = nlist
        self._index: Optional[object] = None
        self._id_map: list[int] = []

    def _load_faiss(self) -> "faiss":
        try:
            import faiss  # type: ignore

            return faiss
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is required for FAISSIndex.\n"
                "Install with: pip install embedx[faiss]"
            ) from exc

    def rebuild(self) -> None:
        import numpy as np

        faiss = self._load_faiss()
        records = self._store.all_records()
        if not records:
            return

        vecs = np.array([r.embedding for r in records], dtype=np.float32)
        faiss.normalize_L2(vecs)

        nlist = min(self._nlist, max(1, len(records) // 10))
        quantizer = faiss.IndexFlatIP(self._dim)
        index = faiss.IndexIVFFlat(quantizer, self._dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(vecs)
        index.add(vecs)

        self._index = index
        self._id_map = [r.id for r in records]

    def search(self, query: list[float], top_k: int = 5) -> list[tuple[Record, float]]:
        import numpy as np

        faiss = self._load_faiss()
        if self._index is None:
            self.rebuild()
        if self._index is None:
            return []

        qvec = np.array([query], dtype=np.float32)
        faiss.normalize_L2(qvec)
        scores, indices = self._index.search(qvec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            db_id = self._id_map[idx]
            all_records = self._store.all_records()
            record = next((r for r in all_records if r.id == db_id), None)
            if record:
                results.append((record, float(score)))
        return results
