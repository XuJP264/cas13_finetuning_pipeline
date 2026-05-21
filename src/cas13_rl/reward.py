from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List

from .cas13_rules import hepn_motif_score, length_distribution_score


def cas13_motif_score(sequence: str) -> float:
    """Backward-compatible alias for the HEPN RxxxxH motif score."""
    return hepn_motif_score(sequence)


def normalize(values: Iterable[float]) -> List[float]:
    vals = [float(v) for v in values]
    if not vals:
        return []
    mu = mean(vals)
    sigma = pstdev(vals)
    if sigma < 1e-8:
        return [0.0 for _ in vals]
    return [(v - mu) / sigma for v in vals]


def _field(row: Dict[str, Any], *names: str, default: float = 0.0) -> float:
    for name in names:
        value = row.get(name)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default
    return default


def _sequence_diversity(sequence: str, batch: List[str]) -> float:
    seq = str(sequence or "")
    others = [str(item or "") for item in batch if item != sequence]
    if not seq or not others:
        return 0.0
    scores = []
    for other in others:
        width = max(len(seq), len(other), 1)
        shared = sum(1 for a, b in zip(seq, other) if a == b)
        shared += max(0, width - max(len(seq), len(other)))
        scores.append(1.0 - shared / float(width))
    return float(mean(scores)) if scores else 0.0


def compute_cas13_rewards(
    sequences: List[str],
    esmfold_rows: List[Dict[str, Any]],
    progen3_rows: List[Dict[str, Any]],
    config: Dict[str, Any] | None = None,
    log_path: str | Path | None = None,
    kl_values: Iterable[float] | None = None,
) -> List[Dict[str, Any]]:
    cfg = config or {}
    weights = cfg.get("weights", cfg)
    alpha = float(cfg.get("alpha", 0.5))
    beta = float(cfg.get("beta", 0.25))
    min_len = int(cfg.get("min_len", 200))
    max_len = int(cfg.get("max_len", 1500))
    target_len = int(cfg.get("target_len", 900))
    length_tolerance = int(cfg.get("length_tolerance", max(1, max_len - min_len)))
    kl_list = [float(x) for x in (kl_values if kl_values is not None else [0.0] * len(sequences))]

    plddt_z = normalize([_field(row, "mean_pLDDT", "mean_plddt") for row in esmfold_rows])
    ptm_z = normalize([_field(row, "pTM", "ptm") for row in esmfold_rows])
    pae_z = normalize([_field(row, "mean_PAE", "mean_pae") for row in esmfold_rows])
    all_sequences = [str(seq or "") for seq in sequences]

    rows: List[Dict[str, Any]] = []
    for i, sequence in enumerate(all_sequences):
        esm = esmfold_rows[i] if i < len(esmfold_rows) else {}
        lm = progen3_rows[i] if i < len(progen3_rows) else {}
        kl = kl_list[i] if i < len(kl_list) else 0.0
        struct_valid = bool(esm.get("valid", True))
        lm_valid = bool(lm.get("valid", True))
        error = "; ".join(str(x) for x in [esm.get("error"), lm.get("error")] if x)
        r_struct = float(plddt_z[i] + alpha * ptm_z[i] - beta * pae_z[i])
        r_lm = _field(lm, "mean_logprob", "progen3_mean_logprob", "progen3_normalized_score", default=0.0)
        r_motif = hepn_motif_score(sequence)
        r_len = length_distribution_score(sequence, target_len=target_len, tolerance=length_tolerance)
        r_div = _sequence_diversity(sequence, all_sequences)
        reward = (
            float(weights.get("w_struct", weights.get("w_plddt", 0.4))) * r_struct
            + float(weights.get("w_lm", weights.get("w_pg3", 0.3))) * r_lm
            + float(weights.get("w_motif", 0.1)) * r_motif
            + float(weights.get("w_len", weights.get("w_valid", 0.2))) * r_len
            + float(weights.get("w_div", 0.0)) * r_div
            - float(weights.get("w_kl", 0.0)) * kl
        )
        row = {
            "sequence": sequence,
            "reward": float(reward),
            "R_struct": r_struct,
            "R_lm": r_lm,
            "R_motif": r_motif,
            "R_len": r_len,
            "R_div": r_div,
            "KL": kl,
            "valid": bool(struct_valid and lm_valid and not error),
            "error": error or None,
            "mean_pLDDT": _field(esm, "mean_pLDDT", "mean_plddt"),
            "pTM": _field(esm, "pTM", "ptm"),
            "mean_PAE": _field(esm, "mean_PAE", "mean_pae"),
            "mean_logprob": r_lm,
            "perplexity": _field(lm, "perplexity", "progen3_perplexity"),
            "reward_components": {
                "struct": r_struct,
                "lm": r_lm,
                "motif": r_motif,
                "length": r_len,
                "diversity": r_div,
                "kl": kl,
            },
            "validity_score": 1.0 if struct_valid and lm_valid else 0.0,
            "cas13_motif_score": r_motif,
            "pg3_norm": r_lm,
            "plddt_norm": plddt_z[i],
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
    """Compatibility wrapper for the original RL smoke tests."""
    sequences = [str(row.get("sequence", "")) for row in oracle_rows]
    esm_rows = [
        {
            "sequence": row.get("sequence", ""),
            "mean_pLDDT": row.get("mean_pLDDT", row.get("mean_plddt", 0.0)),
            "pTM": row.get("pTM", 0.0),
            "mean_PAE": row.get("mean_PAE", 0.0),
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
    cfg = {"weights": weights, "min_len": min_len, "max_len": max_len, "target_len": (min_len + max_len) // 2}
    rewarded = compute_cas13_rewards(sequences, esm_rows, lm_rows, cfg)
    for dst, src in zip(rewarded, oracle_rows):
        dst.update({k: v for k, v in src.items() if k not in dst})
    return rewarded

