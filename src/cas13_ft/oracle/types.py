from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OracleScore:
    sequence: str
    sequence_id: str | None
    passed_hard_filters: bool
    hard_fail_reasons: list[str]
    length: int
    hepn_motif_count: int
    naturalness_score: float | None
    naturalness_z: float | None
    cas13_identity_score: float | None
    cas13_identity_z: float | None
    plddt_mean: float | None
    ptm: float | None
    tm_to_reference: float | None
    structure_z: float | None
    diversity_score: float | None
    penalty: float
    final_reward: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OracleConfig:
    hard_filters: dict[str, Any] = field(default_factory=dict)
    naturalness: dict[str, Any] = field(default_factory=dict)
    cas13_identity: dict[str, Any] = field(default_factory=dict)
    structure: dict[str, Any] = field(default_factory=dict)
    diversity: dict[str, Any] = field(default_factory=dict)
    reward: dict[str, Any] = field(default_factory=dict)
    cache: dict[str, Any] = field(default_factory=dict)


@dataclass
class HardFilterResult:
    passed: bool
    reasons: list[str]
    hepn_positions: list[int]
    length: int
    low_complexity: dict[str, Any]

