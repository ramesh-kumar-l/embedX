"""SQLite-backed persistent storage for embeddings and metadata."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from embedx.utils.helpers import deserialize_vector, now_ts, serialize_vector


@dataclass
class Record:
    id: int
    text: str
    text_hash: str
    embedding: list[float]
    metadata: dict
    created_at: float
    last_used: float
    use_count: int
    namespace: str = "default"
    importance: float = 0.5


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS embeddings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    text        TEXT    NOT NULL,
    text_hash   TEXT    NOT NULL UNIQUE,
    embedding   BLOB    NOT NULL,
    metadata    TEXT    NOT NULL DEFAULT '{}',
    created_at  REAL    NOT NULL,
    last_used   REAL    NOT NULL,
    use_count   INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_hash ON embeddings (text_hash);
"""

# Safe migrations — executed on every startup; skipped if column already exists.
_MIGRATIONS = [
    "ALTER TABLE embeddings ADD COLUMN namespace TEXT DEFAULT 'default'",
    "ALTER TABLE embeddings ADD COLUMN importance REAL DEFAULT 0.5",
    "CREATE INDEX IF NOT EXISTS idx_namespace ON embeddings (namespace)",
]

_CREATE_STATS = """
CREATE TABLE IF NOT EXISTS stats (
    key   TEXT PRIMARY KEY,
    value REAL NOT NULL DEFAULT 0
);
"""


class SQLiteStore:
    def __init__(self, db_path: str = "embedx.db") -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_CREATE_TABLE)
        self._conn.executescript(_CREATE_STATS)
        self._conn.commit()
        self._apply_migrations()

    # ------------------------------------------------------------------
    # Record operations
    # ------------------------------------------------------------------

    def upsert(
        self,
        text: str,
        text_hash: str,
        embedding: list[float],
        metadata: Optional[dict] = None,
        namespace: str = "default",
        importance: float = 0.5,
    ) -> int:
        import json

        ts = now_ts()
        blob = serialize_vector(embedding)
        meta_str = json.dumps(metadata or {})
        with self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO embeddings (text, text_hash, embedding, metadata, created_at, last_used, use_count, namespace, importance)
                VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)
                ON CONFLICT(text_hash) DO UPDATE SET
                    last_used = excluded.last_used,
                    use_count = use_count + 1
                """,
                (text, text_hash, blob, meta_str, ts, ts, namespace, importance),
            )
            return cur.lastrowid or self._get_id_by_hash(text_hash)

    def get_by_hash(self, text_hash: str) -> Optional[Record]:
        row = self._conn.execute(
            "SELECT * FROM embeddings WHERE text_hash = ?", (text_hash,)
        ).fetchone()
        return self._row_to_record(row) if row else None

    def touch(self, record_id: int) -> None:
        """Increment use_count and update last_used."""
        with self._conn:
            self._conn.execute(
                "UPDATE embeddings SET use_count = use_count + 1, last_used = ? WHERE id = ?",
                (now_ts(), record_id),
            )

    def all_records(self) -> list[Record]:
        rows = self._conn.execute("SELECT * FROM embeddings").fetchall()
        return [self._row_to_record(r) for r in rows]

    def records_by_namespace(self, namespace: str) -> list[Record]:
        rows = self._conn.execute(
            "SELECT * FROM embeddings WHERE namespace = ?", (namespace,)
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]

    def delete_by_hash(self, text_hash: str) -> bool:
        with self._conn:
            cur = self._conn.execute(
                "DELETE FROM embeddings WHERE text_hash = ?", (text_hash,)
            )
            return cur.rowcount > 0

    def clear(self) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM embeddings")

    # ------------------------------------------------------------------
    # Stats operations
    # ------------------------------------------------------------------

    def increment_stat(self, key: str, amount: float = 1.0) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO stats (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = value + ?",
                (key, amount, amount),
            )

    def get_stat(self, key: str) -> float:
        row = self._conn.execute("SELECT value FROM stats WHERE key = ?", (key,)).fetchone()
        return row[0] if row else 0.0

    def get_all_stats(self) -> dict[str, float]:
        rows = self._conn.execute("SELECT key, value FROM stats").fetchall()
        return {r["key"]: r["value"] for r in rows}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _apply_migrations(self) -> None:
        for sql in _MIGRATIONS:
            try:
                self._conn.execute(sql)
                self._conn.commit()
            except sqlite3.OperationalError:
                pass  # column/index already exists

    def _get_id_by_hash(self, text_hash: str) -> int:
        row = self._conn.execute(
            "SELECT id FROM embeddings WHERE text_hash = ?", (text_hash,)
        ).fetchone()
        return row[0] if row else -1

    def _row_to_record(self, row: sqlite3.Row) -> Record:
        import json

        return Record(
            id=row["id"],
            text=row["text"],
            text_hash=row["text_hash"],
            embedding=deserialize_vector(row["embedding"]),
            metadata=json.loads(row["metadata"]),
            created_at=row["created_at"],
            last_used=row["last_used"],
            use_count=row["use_count"],
            namespace=row["namespace"] if "namespace" in row.keys() else "default",
            importance=row["importance"] if "importance" in row.keys() else 0.5,
        )

    def close(self) -> None:
        self._conn.close()
