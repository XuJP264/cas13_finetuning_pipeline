from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
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
        "backend": "mock",
    }


def _schema(sequence: str, payload: Dict[str, Any], backend: str) -> Dict[str, Any]:
    mean_logprob = payload.get("mean_logprob", payload.get("progen3_mean_logprob"))
    perplexity = payload.get("perplexity", payload.get("progen3_perplexity"))
    return {
        "sequence": str(payload.get("sequence", sequence)),
        "valid": bool(payload.get("valid", False)),
        "mean_logprob": float(mean_logprob) if mean_logprob is not None else None,
        "perplexity": float(perplexity) if perplexity is not None else None,
        "error": payload.get("error"),
        "backend": payload.get("backend", backend),
    }


@dataclass
class ProGen3Oracle:
    mode: str = "disabled"  # Default disabled per requirements
    model_path: str | None = None
    model_name_or_path: str | None = None
    code_path: str | None = None
    device: str = "cuda"
    max_length: int = 1024
    dtype: str = "auto"
    batch_size: int = 1
    cache_dir: str | None = None
    cache: OracleCache | None = None

    def __post_init__(self) -> None:
        if self.cache is None and self.cache_dir:
            self.cache = OracleCache(Path(self.cache_dir) / "progen3_cache.sqlite")
        self._real = None
        # Only enable if explicitly set to "real" - all other modes (mock, disabled) return mock/disabled scores
        if self.mode == "real":
            model_name = self.model_name_or_path or self.model_path
            if not model_name:
                raise ValueError(
                    "ProGen3 real backend requires oracle.progen3.model_name_or_path "
                    "or oracle.progen3.model_path"
                )
            from .oracles.progen3 import ProGen3Oracle as RealProGen3Oracle

            self._real = RealProGen3Oracle(
                model_name=model_name,
                device=self.device,
                max_length=self.max_length,
                dtype=self.dtype,
                code_path=self.code_path,
            )

    def score_one_uncached(self, sequence: str) -> Dict[str, Any]:
        seq = str(sequence or "")
        try:
            if self.mode == "disabled":
                # ProGen3 is disabled - return explicit error
                return {
                    "sequence": seq,
                    "valid": False,
                    "mean_logprob": None,
                    "perplexity": None,
                    "error": "ProGen3 is disabled in this configuration",
                    "backend": "disabled",
                }
            elif self.mode == "mock":
                payload = _mock_lm_scores(seq)
                return _schema(seq, payload, "mock")
            else:
                # Only run real mode if explicitly enabled
                if self._real is None:
                    raise RuntimeError("ProGen3 real instance not initialized - check configuration")
                raw = self._real.score_one(seq)
                mean_logprob = float(raw.get("mean_logprob", raw.get("progen3_mean_logprob", 0.0)) or 0.0)
                payload = {
                    "mean_logprob": mean_logprob,
                    "perplexity": float(raw.get("perplexity", raw.get("progen3_perplexity", math.exp(-mean_logprob))) or 0.0),
                    "valid": True,
                    "error": None,
                    "backend": "real",
                }
                payload["sequence"] = seq
                return _schema(seq, payload, "real")
        except Exception as exc:
            backend = "disabled" if self.mode == "disabled" else ("mock" if self.mode == "mock" else "real")
            return {
                "sequence": seq,
                "valid": False,
                "mean_logprob": None,
                "perplexity": None,
                "error": f"{type(exc).__name__}: {exc}",
                "backend": backend,
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