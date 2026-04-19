"""EmbedX FastAPI server — thin HTTP wrapper over the core library."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from embedx.api.public import EmbedX

# ---------------------------------------------------------------------------
# Shared instance — one DB connection for the server lifetime
# ---------------------------------------------------------------------------

_db: Optional[EmbedX] = None


def _get_db() -> EmbedX:
    if _db is None:
        raise HTTPException(status_code=503, detail="EmbedX not initialised")
    return _db


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _db
    db_path = os.environ.get("EMBEDX_DB", "embedx.db")
    _db = EmbedX(db_path=db_path)
    yield
    _db.close()
    _db = None


app = FastAPI(
    title="EmbedX API",
    version="0.1.0",
    description="Cost-aware semantic memory and retrieval for LLM applications.",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class AddRequest(BaseModel):
    text: str
    metadata: Optional[dict] = None
    namespace: str = "default"
    importance: float = 0.5


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    hybrid: bool = False
    namespace: Optional[str] = None
    trace: bool = False


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/add")
def add(req: AddRequest) -> dict:
    """Add a document. Returns cache status and record id."""
    return _get_db().add(req.text, metadata=req.metadata, namespace=req.namespace, importance=req.importance)


@app.post("/search")
def search(req: SearchRequest):
    """Semantic search. Returns ranked results with scores.

    When ``trace`` is ``true`` the response wraps results in
    ``{"results": [...], "trace": {...}}``. Without ``trace`` the existing
    ``[{...}, ...]`` array format is preserved unchanged.
    """
    out = _get_db().search(
        req.query,
        top_k=req.top_k,
        hybrid=req.hybrid,
        namespace=req.namespace,
        trace=req.trace,
    )
    if req.trace:
        results, trace_data = out
        return {"results": results, "trace": trace_data}
    return out


@app.get("/stats")
def stats() -> dict:
    """Return cache, cost, and latency statistics."""
    return _get_db().get_stats()


@app.get("/health")
def health() -> dict:
    """Liveness probe."""
    return {"status": "ok"}
