from __future__ import annotations

import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List

from .cache import sequence_sha256
from .cas13_rules import hepn_motif_score

HEPN_RE = re.compile(r"R[A-Z]{4}H")


DEFAULT_CALIBRATION = {
    "lm": {"low": -5.0, "high": -0.5},
    "plddt": {"low": 50.0, "high": 90.0},
    "ptm": {"low": 0.2, "high": 0.8},
    "pae": {"low": 5.0, "high": 25.0},
}


def cas13_motif_score(sequence: str) -> float:
    return hepn_motif_score(sequence)


def clamp(value: float, eps: float = 1e-6) -> float:
    return max(float(eps), min(1.0, float(value)))


def phi_up(x: float | None, low: float, high: float, eps: float = 1e-6) -> float:
    if x is None or high <= low:
        return float(eps)
    return clamp((float(x) - low) / (high - low), eps=eps)


def phi_down(x: float | None, low: float, high: float, eps: float = 1e-6) -> float:
    if x is None or high <= low:
        return float(eps)
    return clamp((high - float(x)) / (high - low), eps=eps)


def phi_centered_length(length: int, center: int, tolerance: int, eps: float = 1e-6) -> float:
    if tolerance <= 0:
        return 1.0 if int(length) == int(center) else float(eps)
    return clamp(1.0 - abs(int(length) - int(center)) / float(tolerance), eps=eps)


def normalize(values: Iterable[float]) -> List[float]:
    vals = [float(v) for v in values]
    if not vals:
        return []
    mu = mean(vals)
    var = mean([(v - mu) ** 2 for v in vals])
    sigma = math.sqrt(var)
    if sigma < 1e-8:
        return [0.0 for _ in vals]
    return [(v - mu) / sigma for v in vals]


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_float(row: Dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = _float_or_none(row.get(name))
        if value is not None:
            return value
    return None


def _hepn_count(sequence: str) -> int:
    return len(HEPN_RE.findall(str(sequence or "").upper()))


def _sequence_diversity(sequence: str, batch: List[str]) -> float:
    seq = str(sequence or "")
    others = [str(item or "") for item in batch if item != sequence]
    if not seq or not others:
        return 1.0
    scores = []
    for other in others:
        width = max(len(seq), len(other), 1)
        matches = sum(1 for a, b in zip(seq, other) if a == b)
        scores.append(1.0 - matches / float(width))
    return clamp(mean(scores), eps=1e-6)


def _weighted_geomean(scores: Dict[str, float], weights: Dict[str, float], eps: float) -> float:
    active = [(name, float(weight)) for name, weight in weights.items() if float(weight) > 0 and name in scores]
    if not active:
        return 1.0
    total = sum(weight for _, weight in active)
    log_sum = sum((weight / total) * math.log(clamp(scores[name], eps=eps)) for name, weight in active)
    return clamp(math.exp(log_sum), eps=eps)


def _calibration(cfg: Dict[str, Any], name: str) -> Dict[str, float]:
    merged = dict(DEFAULT_CALIBRATION[name])
    merged.update((cfg.get("calibration") or {}).get(name, {}))
    return merged


def _basic_invalid_reason(sequence: str, min_len: int, max_len: int) -> str | None:
    seq = str(sequence or "").upper()
    canonical = set("ACDEFGHIKLMNPQRSTVWY")
    if not seq:
        return "empty sequence"
    if any(ch not in canonical for ch in seq):
        return "non-canonical amino acid"
    if len(seq) < min_len or len(seq) > max_len:
        return f"length {len(seq)} outside [{min_len}, {max_len}]"
    return None


def compute_cas13_rewards(
    sequences: List[str],
    esmfold_rows: List[Dict[str, Any]],
    progen3_rows: List[Dict[str, Any]],
    config: Dict[str, Any] | None = None,
    log_path: str | Path | None = None,
    kl_values: Iterable[float] | None = None,
) -> List[Dict[str, Any]]:
    cfg = config or {}
    eps = float(cfg.get("eps", 1e-6))
    reward_scale = float(cfg.get("reward_scale", 1.0))
    invalid_reward = float(cfg.get("invalid_reward", -1.0))
    min_len = int(cfg.get("min_len", 200))
    max_len = int(cfg.get("max_len", 1500))
    length_center = int(cfg.get("length_center", cfg.get("target_len", 900)))
    length_tolerance = int(cfg.get("length_tolerance", 700))
    weights = {
        "lm": 1.0,
        "plddt": 1.0,
        "ptm": 1.0,
        "pae": 0.0,
        "hepn": 0.5,
        "length": 0.5,
        "diversity": 0.0,
    }
    weights.update(cfg.get("weights", {}))
    all_sequences = [str(seq or "").upper() for seq in sequences]
    kl_list = [float(x) for x in (kl_values if kl_values is not None else [0.0] * len(all_sequences))]
    rows: List[Dict[str, Any]] = []

    for i, sequence in enumerate(all_sequences):
        esm = esmfold_rows[i] if i < len(esmfold_rows) else {}
        lm = progen3_rows[i] if i < len(progen3_rows) else {}
        kl = kl_list[i] if i < len(kl_list) else 0.0
        invalid_reason = _basic_invalid_reason(sequence, min_len=min_len, max_len=max_len)
        valid_basic = invalid_reason is None
        hepn_count = _hepn_count(sequence)
        raw_lm = _get_float(lm, "mean_logprob", "progen3_mean_logprob", "progen3_normalized_score")
        raw_ppl = _get_float(lm, "perplexity", "progen3_perplexity")
        raw_plddt = _get_float(esm, "mean_plddt", "mean_pLDDT", "mean_plddt")
        raw_ptm = _get_float(esm, "ptm", "pTM")
        raw_pae = _get_float(esm, "mean_pae", "mean_PAE")
        oracle_errors = [str(x) for x in [esm.get("error"), lm.get("error")] if x]

        lm_cal = _calibration(cfg, "lm")
        plddt_cal = _calibration(cfg, "plddt")
        ptm_cal = _calibration(cfg, "ptm")
        pae_cal = _calibration(cfg, "pae")
        scores = {
            "lm": phi_up(raw_lm, lm_cal["low"], lm_cal["high"], eps=eps),
            "plddt": phi_up(raw_plddt, plddt_cal["low"], plddt_cal["high"], eps=eps),
            "ptm": phi_up(raw_ptm, ptm_cal["low"], ptm_cal["high"], eps=eps),
            "pae": phi_down(raw_pae, pae_cal["low"], pae_cal["high"], eps=eps),
            "hepn": clamp(hepn_motif_score(sequence), eps=eps),
            "length": phi_centered_length(len(sequence), length_center, length_tolerance, eps=eps),
            "diversity": _sequence_diversity(sequence, all_sequences),
        }
        if valid_basic and not oracle_errors:
            r_prop = _weighted_geomean(scores, weights, eps=eps)
            reward_for_rl = reward_scale * (2.0 * r_prop - 1.0)
            valid = bool(esm.get("valid", True) and lm.get("valid", True))
        else:
            r_prop = 0.0
            reward_for_rl = invalid_reward
            valid = False

        breakdown = {
            "sequence": sequence,
            "sequence_sha256": sequence_sha256(sequence),
            "sequence_length": len(sequence),
            "valid_basic": valid_basic,
            "raw_lm_mean_logprob": raw_lm,
            "raw_lm_perplexity": raw_ppl,
            "raw_mean_plddt": raw_plddt,
            "raw_ptm": raw_ptm,
            "raw_mean_pae": raw_pae,
            "raw_hepn_count": hepn_count,
            "score_lm": scores["lm"],
            "score_plddt": scores["plddt"],
            "score_ptm": scores["ptm"],
            "score_pae": scores["pae"],
            "score_hepn": scores["hepn"],
            "score_length": scores["length"],
            "score_diversity": scores["diversity"],
            "r_prop": r_prop,
            "reward_for_rl": reward_for_rl,
            "invalid_reason": invalid_reason,
            "oracle_errors": oracle_errors,
        }
        row = {
            **breakdown,
            "reward": reward_for_rl,
            "valid": valid,
            "error": "; ".join([x for x in [invalid_reason, *oracle_errors] if x]) or None,
            "mean_plddt": raw_plddt,
            "ptm": raw_ptm,
            "mean_pae": raw_pae,
            "mean_pLDDT": raw_plddt,
            "pTM": raw_ptm,
            "mean_PAE": raw_pae,
            "mean_logprob": raw_lm,
            "perplexity": raw_ppl,
            "R_struct": scores["plddt"],
            "R_lm": scores["lm"],
            "R_motif": scores["hepn"],
            "R_len": scores["length"],
            "R_div": scores["diversity"],
            "KL": kl,
            "reward_components": {
                "struct": scores["plddt"],
                "lm": scores["lm"],
                "plddt": scores["plddt"],
                "ptm": scores["ptm"],
                "pae": scores["pae"],
                "hepn": scores["hepn"],
                "length": scores["length"],
                "diversity": scores["diversity"],
            },
            "validity_score": 1.0 if valid else 0.0,
            "cas13_motif_score": scores["hepn"],
            "pg3_norm": scores["lm"],
            "plddt_norm": scores["plddt"],
        }
        rows.append(row)

    if log_path is not None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    return rows


def compute_rewards(
    oracle_rows: List[Dict[str, Any]],
    weights: Dict[str, float],
    min_len: int = 200,
    max_len: int = 1500,
) -> List[Dict[str, Any]]:
    sequences = [str(row.get("sequence", "")) for row in oracle_rows]
    esm_rows = [
        {
            "sequence": row.get("sequence", ""),
            "mean_plddt": row.get("mean_plddt", row.get("mean_pLDDT", 0.0)),
            "ptm": row.get("ptm", row.get("pTM", 0.0)),
            "mean_pae": row.get("mean_pae", row.get("mean_PAE", 0.0)),
            "valid": row.get("valid", True),
            "error": row.get("error"),
        }
        for row in oracle_rows
    ]
    lm_rows = [
        {
            "sequence": row.get("sequence", ""),
            "mean_logprob": row.get("mean_logprob", row.get("progen3_mean_logprob", -float(row.get("progen3_nll", 0.0)))),
            "perplexity": row.get("perplexity", row.get("progen3_perplexity", 0.0)),
            "valid": row.get("valid", True),
            "error": row.get("error"),
        }
        for row in oracle_rows
    ]
    mapped = {
        "lm": weights.get("w_lm", weights.get("w_pg3", 1.0)),
        "plddt": weights.get("w_plddt", weights.get("w_struct", 1.0)),
        "ptm": weights.get("w_ptm", 0.0),
        "hepn": weights.get("w_motif", 1.0),
        "length": weights.get("w_len", weights.get("w_valid", 1.0)),
        "diversity": weights.get("w_div", 0.0),
        "pae": weights.get("w_pae", 0.0),
    }
    return compute_cas13_rewards(sequences, esm_rows, lm_rows, {"weights": mapped, "min_len": min_len, "max_len": max_len})
