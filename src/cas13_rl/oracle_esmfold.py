from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from .cache import OracleCache


def _mock_struct_scores(sequence: str) -> Dict[str, Any]:
    seq = str(sequence or "")
    valid = bool(seq)
    motif_bonus = 8.0 if "R" in seq and "H" in seq else 0.0
    length_penalty = min(20.0, abs(len(seq) - 900) / 50.0)
    mean_plddt = max(1.0, min(95.0, 62.0 + motif_bonus - length_penalty))
    ptm = max(0.05, min(0.9, mean_plddt / 100.0 - 0.08))
    mean_pae = max(2.0, 30.0 - mean_plddt / 4.0)
    return {
        "mean_pLDDT": float(mean_plddt),
        "pTM": float(ptm),
        "mean_PAE": float(mean_pae),
        "valid": valid,
        "error": None if valid else "empty sequence",
    }


@dataclass
class ESMFoldOracle:
    mode: str = "mock"
    model_path: str | None = None
    device: str = "cuda"
    max_length: int = 1500
    cache: OracleCache | None = None

    def __post_init__(self) -> None:
        self._real = None
        if self.mode == "mock":
            return
        from .oracles.esmfold import ESMFoldOracle as RealESMFoldOracle

        self._real = RealESMFoldOracle(enabled=True, device=self.device, max_length=self.max_length, output_dir=None)

    def score_one_uncached(self, sequence: str) -> Dict[str, Any]:
        seq = str(sequence or "")
        try:
            if self.mode == "mock":
                payload = _mock_struct_scores(seq)
            else:
                raw = self._real.score_one(seq)
                payload = {
                    "mean_pLDDT": float(raw.get("mean_pLDDT", raw.get("mean_plddt", 0.0)) or 0.0),
                    "pTM": float(raw.get("pTM", raw.get("ptm", math.nan)) or 0.0),
                    "mean_PAE": float(raw.get("mean_PAE", raw.get("mean_pae", math.nan)) or 0.0),
                    "valid": True,
                    "error": None,
                }
            payload["sequence"] = seq
            return payload
        except Exception as exc:
            return {"sequence": seq, "mean_pLDDT": 0.0, "pTM": 0.0, "mean_PAE": 0.0, "valid": False, "error": f"{type(exc).__name__}: {exc}"}

    def score_one(self, sequence: str) -> Dict[str, Any]:
        if self.cache is None:
            return self.score_one_uncached(sequence)
        cached = self.cache.get(sequence)
        if cached is not None:
            return cached
        payload = self.score_one_uncached(sequence)
        self.cache.set(sequence, payload)
        return payload

    def score_many(self, sequences: Iterable[str]) -> List[Dict[str, Any]]:
        return [self.score_one(seq) for seq in sequences]

