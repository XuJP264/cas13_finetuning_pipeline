from __future__ import annotations

import re
from statistics import mean, pstdev
from typing import Dict, Iterable, List

from cas13_ft.sequence import validity_score

HEPN_RE = re.compile(r"R[A-Z]{4}H")


def cas13_motif_score(sequence: str) -> float:
    """Heuristic HEPN-like motif score; not evidence of biological function."""
    matches = HEPN_RE.findall(str(sequence or "").upper())
    if len(matches) >= 2:
        return 1.0
    if len(matches) == 1:
        return 0.5
    return 0.0


def normalize(values: Iterable[float]) -> List[float]:
    vals = [float(v) for v in values]
    if not vals:
        return []
    mu = mean(vals)
    sigma = pstdev(vals)
    if sigma < 1e-8:
        return [0.0 for _ in vals]
    return [(v - mu) / sigma for v in vals]


def compute_rewards(
    oracle_rows: List[Dict[str, float]],
    weights: Dict[str, float],
    min_len: int = 200,
    max_len: int = 1500,
) -> List[Dict[str, float]]:
    pg3_norm = normalize([-row.get("progen3_nll", 0.0) for row in oracle_rows])
    plddt_norm = normalize([row.get("mean_plddt", 0.0) for row in oracle_rows])
    out = []
    for i, row in enumerate(oracle_rows):
        seq = row["sequence"]
        valid = validity_score(seq, min_len=min_len, max_len=max_len)
        motif = cas13_motif_score(seq)
        kl = float(row.get("kl_to_reference", 0.0))
        reward = (
            weights.get("w_pg3", 0.3) * pg3_norm[i]
            + weights.get("w_plddt", 0.4) * plddt_norm[i]
            + weights.get("w_valid", 0.2) * valid
            + weights.get("w_motif", 0.1) * motif
            - weights.get("w_kl", 0.0) * kl
        )
        merged = dict(row)
        merged.update(
            {
                "reward": float(reward),
                "validity_score": float(valid),
                "cas13_motif_score": float(motif),
                "pg3_norm": float(pg3_norm[i]),
                "plddt_norm": float(plddt_norm[i]),
            }
        )
        out.append(merged)
    return out
