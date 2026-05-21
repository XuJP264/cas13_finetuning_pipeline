from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
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
        "mean_plddt": float(mean_plddt),
        "ptm": float(ptm),
        "mean_pae": float(mean_pae),
        "pdb_path": None,
        "valid": valid,
        "error": None if valid else "empty sequence",
        "backend": "mock",
    }


def _schema(sequence: str, payload: Dict[str, Any], backend: str) -> Dict[str, Any]:
    valid = bool(payload.get("valid", False))
    return {
        "sequence": str(payload.get("sequence", sequence)),
        "valid": valid,
        "mean_plddt": float(payload["mean_plddt"]) if payload.get("mean_plddt") is not None else (float(payload["mean_pLDDT"]) if payload.get("mean_pLDDT") is not None else None),
        "ptm": float(payload["ptm"]) if payload.get("ptm") is not None else (float(payload["pTM"]) if payload.get("pTM") is not None else None),
        "mean_pae": float(payload["mean_pae"]) if payload.get("mean_pae") is not None else (float(payload["mean_PAE"]) if payload.get("mean_PAE") is not None else None),
        "pdb_path": payload.get("pdb_path"),
        "error": payload.get("error"),
        "backend": payload.get("backend", backend),
    }


@dataclass
class ESMFoldOracle:
    mode: str = "mock"
    backend: str = "facebook_esm"
    model_path: str | None = None
    device: str = "cuda"
    max_length: int = 1500
    cache_dir: str | None = None
    cache: OracleCache | None = None

    def __post_init__(self) -> None:
        if self.cache is None and self.cache_dir:
            self.cache = OracleCache(Path(self.cache_dir) / "esmfold_cache.sqlite")
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
                    "mean_plddt": float(raw.get("mean_plddt", raw.get("mean_pLDDT", 0.0)) or 0.0),
                    "ptm": float(raw.get("ptm", raw.get("pTM", math.nan)) or 0.0),
                    "mean_pae": float(raw.get("mean_pae", raw.get("mean_PAE", math.nan)) or 0.0),
                    "pdb_path": raw.get("pdb_path"),
                    "valid": True,
                    "error": None,
                    "backend": "real",
                }
            payload["sequence"] = seq
            return _schema(seq, payload, "mock" if self.mode == "mock" else "real")
        except Exception as exc:
            return {
                "sequence": seq,
                "valid": False,
                "mean_plddt": None,
                "ptm": None,
                "mean_pae": None,
                "pdb_path": None,
                "error": f"{type(exc).__name__}: {exc}",
                "backend": "mock" if self.mode == "mock" else "real",
            }

    def score_one(self, sequence: str) -> Dict[str, Any]:
        if self.cache is None:
            return self.score_one_uncached(sequence)
        cached = self.cache.get(sequence)
        if cached is not None:
            return _schema(sequence, cached, "mock" if self.mode == "mock" else "real")
        payload = self.score_one_uncached(sequence)
        self.cache.set(sequence, payload)
        return payload

    def score_many(self, sequences: Iterable[str]) -> List[Dict[str, Any]]:
        return [self.score_one(seq) for seq in sequences]
