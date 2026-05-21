from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def hash_sequence(seq: str) -> str:
    return hashlib.sha256(str(seq).encode("utf-8")).hexdigest()


def make_cache_key(seq: str, scorer_name: str, scorer_version: str, config_hash: str) -> str:
    raw = "|".join([hash_sequence(seq), scorer_name, scorer_version, config_hash])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class OracleSQLiteCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS oracle_cache (
                key TEXT PRIMARY KEY,
                sequence_hash TEXT,
                scorer_name TEXT,
                scorer_version TEXT,
                result_json TEXT,
                created_at TEXT
            )
            """
        )
        self.conn.commit()

    def get_cached(self, seq: str, scorer_name: str, scorer_version: str, config_hash: str) -> dict[str, Any] | None:
        key = make_cache_key(seq, scorer_name, scorer_version, config_hash)
        row = self.conn.execute("SELECT result_json FROM oracle_cache WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set_cached(self, seq: str, scorer_name: str, scorer_version: str, config_hash: str, result: dict[str, Any]) -> None:
        key = make_cache_key(seq, scorer_name, scorer_version, config_hash)
        self.conn.execute(
            """
            INSERT OR REPLACE INTO oracle_cache
            (key, sequence_hash, scorer_name, scorer_version, result_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                key,
                hash_sequence(seq),
                scorer_name,
                scorer_version,
                json.dumps(result, ensure_ascii=True, sort_keys=True),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

