from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from .cache import OracleCache


def _mock_lm_scores(sequence: str) -> Dict[str, Any]:
    seq = str(sequence or "")
    valid = bool(seq)
    rare_penalty = sum(1 for ch in seq if ch in {"W", "C", "M"}) / max(1, len(seq))
    mean_logprob = -1.8 - rare_penalty
    return {
        "mean_logprob": float(mean_logprob),
        "perplexity": float(math.exp(-mean_logprob)),
        "valid": valid,
        "error": None if valid else "empty sequence",
    }


@dataclass
class ProGen3Oracle:
    mode: str = "mock"
    model_path: str | None = None
    device: str = "cuda"
    max_length: int = 1024
    dtype: str = "auto"
    cache: OracleCache | None = None

    def __post_init__(self) -> None:
        self._real = None
        if self.mode == "mock":
            return
        from .oracles.progen3 import ProGen3Oracle as RealProGen3Oracle

        self._real = RealProGen3Oracle(
            model_name=self.model_path or "Profluent-Bio/progen3-219m",
            device=self.device,
            max_length=self.max_length,
            dtype=self.dtype,
        )

    def score_one_uncached(self, sequence: str) -> Dict[str, Any]:
        seq = str(sequence or "")
        try:
            if self.mode == "mock":
                payload = _mock_lm_scores(seq)
            else:
                raw = self._real.score_one(seq)
                mean_logprob = float(raw.get("mean_logprob", raw.get("progen3_mean_logprob", 0.0)) or 0.0)
                payload = {
                    "mean_logprob": mean_logprob,
                    "perplexity": float(raw.get("perplexity", raw.get("progen3_perplexity", math.exp(-mean_logprob))) or 0.0),
                    "valid": True,
                    "error": None,
                }
            payload["sequence"] = seq
            return payload
        except Exception as exc:
            return {"sequence": seq, "mean_logprob": 0.0, "perplexity": float("inf"), "valid": False, "error": f"{type(exc).__name__}: {exc}"}

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

