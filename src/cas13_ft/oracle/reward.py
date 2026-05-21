from __future__ import annotations

from statistics import mean
from typing import Any

from .types import HardFilterResult


def z_normalize(value: float | None, mean: float | None, std: float | None, eps: float = 1e-8, clip: list[float] | tuple[float, float] | None = None) -> float:
    if value is None or mean is None or std is None or abs(float(std)) < eps:
        z = 0.0
    else:
        z = (float(value) - float(mean)) / max(abs(float(std)), eps)
    if clip is not None:
        z = max(float(clip[0]), min(float(clip[1]), z))
    return float(z)


def hepn_position_score(hepn_count: int) -> float:
    if hepn_count == 2:
        return 1.0
    if hepn_count == 1:
        return -0.5
    return -1.0


def _clip(value: float, bounds: list[float] | tuple[float, float] | None) -> float:
    if bounds is None:
        return value
    return max(float(bounds[0]), min(float(bounds[1]), value))


def structure_score(
    plddt_mean: float | None,
    ptm: float | None,
    tm_to_reference: float | None,
    config: dict[str, Any],
    metadata: dict[str, Any],
) -> float:
    clip = config.get("clip", [-3, 3])
    terms = []
    if plddt_mean is not None:
        terms.append(z_normalize(plddt_mean, config.get("plddt_mean", 70.0), config.get("plddt_std", 10.0), clip=clip))
    if ptm is not None:
        terms.append(z_normalize(ptm, config.get("ptm_mean", 0.5), config.get("ptm_std", 0.15), clip=clip))
    if tm_to_reference is not None:
        terms.append(z_normalize(tm_to_reference, config.get("tm_mean", 0.5), config.get("tm_std", 0.15), clip=clip))
    if not terms:
        metadata["structure_missing"] = True
        return 0.0
    return float(mean(terms))


def compute_penalty(hard_filter: HardFilterResult) -> float:
    penalty = 0.0
    if hard_filter.low_complexity.get("is_low_complexity"):
        penalty += 0.25
    return penalty


def combine_oracle_scores(
    *,
    hard_filter: HardFilterResult,
    naturalness_score: float | None,
    cas13_identity_score: float | None,
    plddt_mean: float | None,
    ptm: float | None,
    tm_to_reference: float | None = None,
    diversity_score: float | None = None,
    reward_config: dict[str, Any] | None = None,
    naturalness_config: dict[str, Any] | None = None,
    cas13_identity_config: dict[str, Any] | None = None,
    structure_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reward_cfg = reward_config or {}
    hard_fail_reward = float(reward_cfg.get("hard_fail_reward", -10.0))
    if not hard_filter.passed:
        return {
            "naturalness_z": None,
            "cas13_identity_z": None,
            "structure_z": None,
            "hepn_position_score": hepn_position_score(len(hard_filter.hepn_positions)),
            "penalty": 0.0,
            "final_reward": hard_fail_reward,
            "metadata": {"hard_fail_reward": True},
        }

    nat_cfg = naturalness_config or {}
    id_cfg = cas13_identity_config or {}
    struct_cfg = structure_config or {}
    metadata: dict[str, Any] = {}
    naturalness_z = z_normalize(
        naturalness_score,
        nat_cfg.get("mean"),
        nat_cfg.get("std"),
        clip=nat_cfg.get("clip", [-3, 3]),
    )
    cas13_identity_z = z_normalize(
        cas13_identity_score,
        id_cfg.get("mean", 0.5),
        id_cfg.get("std", 0.25),
        clip=id_cfg.get("clip", [-3, 3]),
    )
    structure_z = structure_score(plddt_mean, ptm, tm_to_reference, struct_cfg, metadata)
    hepn_z = hepn_position_score(len(hard_filter.hepn_positions))
    diversity_z = float(diversity_score) if diversity_score is not None else 0.0
    weights = {
        "naturalness": 0.20,
        "cas13_identity": 0.25,
        "structure": 0.25,
        "hepn_position": 0.15,
        "diversity": 0.15,
    }
    weights.update(reward_cfg.get("weights", {}))
    penalty = compute_penalty(hard_filter)
    final = (
        float(weights.get("naturalness", 0.0)) * naturalness_z
        + float(weights.get("cas13_identity", 0.0)) * cas13_identity_z
        + float(weights.get("structure", 0.0)) * structure_z
        + float(weights.get("hepn_position", 0.0)) * hepn_z
        + float(weights.get("diversity", 0.0)) * diversity_z
        - penalty
    )
    final = _clip(final, reward_cfg.get("clip_final_reward", [-5, 5]))
    return {
        "naturalness_z": naturalness_z,
        "cas13_identity_z": cas13_identity_z,
        "structure_z": structure_z,
        "hepn_position_score": hepn_z,
        "penalty": penalty,
        "final_reward": final,
        "metadata": metadata,
    }

