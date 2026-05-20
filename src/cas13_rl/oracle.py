from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List

from cas13_ft.sequence import sequence_hash, validity_score
from .oracles import ESMFoldOracle, ProGen3Oracle
from .reward import cas13_motif_score


class OracleCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS scores (seq_hash TEXT PRIMARY KEY, sequence TEXT NOT NULL, payload TEXT NOT NULL)"
        )
        self.conn.commit()

    def get(self, sequence: str) -> Dict | None:
        digest = sequence_hash(sequence)
        row = self.conn.execute("SELECT payload FROM scores WHERE seq_hash = ?", (digest,)).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def set(self, sequence: str, payload: Dict) -> None:
        digest = sequence_hash(sequence)
        self.conn.execute(
            "INSERT OR REPLACE INTO scores(seq_hash, sequence, payload) VALUES (?, ?, ?)",
            (digest, sequence, json.dumps(payload, ensure_ascii=True)),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


class MockOracle:
    def __init__(self, min_len: int = 200, max_len: int = 1500):
        self.min_len = min_len
        self.max_len = max_len

    def score_one(self, sequence: str) -> Dict[str, float | str]:
        valid = validity_score(sequence, self.min_len, self.max_len)
        motif = cas13_motif_score(sequence)
        return {
            "sequence": sequence,
            "progen3_nll": float(max(1.0, len(sequence) / 100.0) - motif),
            "mean_plddt": float(50.0 + 30.0 * valid + 10.0 * motif),
            "validity_score": valid,
            "cas13_motif_score": motif,
        }

    def score_many(self, sequences: Iterable[str]) -> List[Dict]:
        return [self.score_one(seq) for seq in sequences]


class ProGen3EsmFoldOracle:
    def __init__(self, config: Dict):
        oracle_cfg = config.get("oracle", {})
        pg3_cfg = oracle_cfg.get("progen3", {})
        esm_cfg = oracle_cfg.get("esmfold", {})
        self.progen3 = ProGen3Oracle(
            model_name=pg3_cfg.get("model_name", oracle_cfg.get("progen3_model", "Profluent-Bio/progen3-219m")),
            device=pg3_cfg.get("device", "auto"),
            max_length=pg3_cfg.get("max_length", 1024),
            dtype=pg3_cfg.get("dtype", "auto"),
        )
        self.esmfold = ESMFoldOracle(
            enabled=esm_cfg.get("enabled", False),
            device=esm_cfg.get("device", "auto"),
            max_length=esm_cfg.get("max_length", 800),
            output_dir=esm_cfg.get("pdb_dir", "outputs/rl/pdb"),
        )

    def score_one(self, sequence: str) -> Dict:
        pg3 = self.progen3.score_one(sequence)
        esm = self.esmfold.score_one(sequence)
        merged = dict(pg3)
        merged.update({k: v for k, v in esm.items() if k != "sequence"})
        return merged

    def score_many(self, sequences: Iterable[str]) -> List[Dict]:
        return [self.score_one(seq) for seq in sequences]


def build_oracle(config: Dict):
    oracle_cfg = config.get("oracle", {})
    mode = oracle_cfg.get("mode", "mock")
    if mode == "mock":
        return MockOracle(oracle_cfg.get("min_len", 200), oracle_cfg.get("max_len", 1500))
    if mode in {"real_progen3", "real"}:
        if oracle_cfg.get("progen3", {}).get("enabled", True) is not True:
            raise ValueError("oracle.mode=real_progen3 requires oracle.progen3.enabled=true")
        return ProGen3EsmFoldOracle(config)
    raise ValueError(f"Unsupported oracle.mode: {mode}")


def score_with_cache(sequences: Iterable[str], oracle, cache: OracleCache) -> List[Dict]:
    out = []
    misses = []
    for seq in sequences:
        cached = cache.get(seq)
        if cached is None:
            misses.append(seq)
        else:
            out.append(cached)
    if misses:
        for payload in oracle.score_many(misses):
            cache.set(payload["sequence"], payload)
            out.append(payload)
    return out
