from __future__ import annotations

import re
from dataclasses import dataclass

from cas13_ft.sequence import VALID_AA

HEPN_MOTIF_RE = re.compile(r"R[A-Z]{4}H")


@dataclass(frozen=True)
class Cas13RuleResult:
    valid: bool
    motif_score: float
    length_score: float
    motif_count: int
    length: int
    error: str | None = None


def hepn_motif_score(sequence: str) -> float:
    seq = str(sequence or "").upper()
    count = len(HEPN_MOTIF_RE.findall(seq))
    if count >= 2:
        return 1.0
    if count == 1:
        return 0.5
    return 0.0


def length_distribution_score(sequence: str, target_len: int = 900, tolerance: int = 700) -> float:
    length = len(str(sequence or ""))
    if tolerance <= 0:
        return 1.0 if length == target_len else 0.0
    return max(0.0, 1.0 - abs(length - target_len) / float(tolerance))


def validate_cas13_sequence(
    sequence: str,
    min_len: int = 200,
    max_len: int = 1500,
    target_len: int = 900,
    length_tolerance: int = 700,
) -> Cas13RuleResult:
    seq = str(sequence or "").upper()
    motif_count = len(HEPN_MOTIF_RE.findall(seq))
    length = len(seq)
    errors = []
    if not seq:
        errors.append("empty sequence")
    if any(ch not in VALID_AA for ch in seq):
        errors.append("non-canonical amino acid")
    if length < min_len or length > max_len:
        errors.append(f"length {length} outside [{min_len}, {max_len}]")
    return Cas13RuleResult(
        valid=not errors,
        motif_score=hepn_motif_score(seq),
        length_score=length_distribution_score(seq, target_len=target_len, tolerance=length_tolerance),
        motif_count=motif_count,
        length=length,
        error="; ".join(errors) if errors else None,
    )

