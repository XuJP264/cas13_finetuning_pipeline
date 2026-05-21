from __future__ import annotations

import json
import sqlite3
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict


def sequence_sha256(sequence: str) -> str:
    return sha256(str(sequence).encode("utf-8")).hexdigest()


class OracleCache:
    """SQLite cache for expensive per-sequence oracle calls."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS oracle_cache (
                seq_hash TEXT PRIMARY KEY,
                sequence TEXT NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def get(self, sequence: str) -> Dict[str, Any] | None:
        seq_hash = sequence_sha256(sequence)
        row = self.conn.execute("SELECT payload FROM oracle_cache WHERE seq_hash = ?", (seq_hash,)).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def set(self, sequence: str, payload: Dict[str, Any]) -> None:
        seq_hash = sequence_sha256(sequence)
        self.conn.execute(
            "INSERT OR REPLACE INTO oracle_cache(seq_hash, sequence, payload) VALUES (?, ?, ?)",
            (seq_hash, sequence, json.dumps(payload, ensure_ascii=True, sort_keys=True)),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

