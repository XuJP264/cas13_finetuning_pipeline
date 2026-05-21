from __future__ import annotations

import re
from collections import Counter
from typing import Any

from .types import HardFilterResult

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
HEPN_RE = re.compile(r"R[A-Z]{4}H")


def validate_amino_acid_sequence(seq: str) -> tuple[bool, list[str]]:
    text = str(seq or "")
    reasons: list[str] = []
    if not text:
        reasons.append("empty_sequence")
    if text != text.strip() or any(ch.isspace() for ch in text):
        reasons.append("contains_whitespace")
    invalid = sorted({ch for ch in text.upper() if ch not in VALID_AA})
    if invalid:
        reasons.append(f"invalid_alphabet:{''.join(invalid)}")
    return not reasons, reasons


def count_hepn_rx4h(seq: str) -> list[int]:
    valid, reasons = validate_amino_acid_sequence(seq)
    if not valid:
        raise ValueError(f"count_hepn_rx4h requires valid standard amino acid sequence: {reasons}")
    return [match.start() for match in HEPN_RE.finditer(str(seq).upper())]


def _max_run(seq: str) -> int:
    best = 0
    current = 0
    previous = None
    for ch in seq:
        current = current + 1 if ch == previous else 1
        previous = ch
        best = max(best, current)
    return best


def low_complexity_score(seq: str, max_aa_fraction: float = 0.20, max_run: int = 8) -> dict[str, Any]:
    text = str(seq or "").upper()
    reasons: list[str] = []
    if not text:
        fraction = 0.0
        run = 0
    else:
        counts = Counter(text)
        fraction = max(counts.values()) / float(len(text))
        run = _max_run(text)
    if fraction > max_aa_fraction:
        reasons.append(f"max_aa_fraction>{max_aa_fraction}")
    if run >= max_run:
        reasons.append(f"max_run>={max_run}")
    return {
        "max_aa_fraction": float(fraction),
        "max_run": int(run),
        "is_low_complexity": bool(reasons),
        "reasons": reasons,
    }


def apply_hard_filters(seq: str, config: dict[str, Any] | None = None) -> HardFilterResult:
    cfg = config or {}
    text = str(seq or "").upper()
    reasons: list[str] = []
    valid, alphabet_reasons = validate_amino_acid_sequence(text)
    if cfg.get("valid_amino_acids", True) and not valid:
        reasons.extend(alphabet_reasons)

    length_min, length_max = cfg.get("length_range", [800, 1400])
    if not (int(length_min) <= len(text) <= int(length_max)):
        reasons.append(f"length_out_of_range:{len(text)} not in [{length_min},{length_max}]")

    hepn_positions: list[int] = []
    if valid:
        hepn_positions = count_hepn_rx4h(text)
    if cfg.get("require_two_hepn_rx4h", True) and len(hepn_positions) != 2:
        reasons.append(f"hepn_count:{len(hepn_positions)}")

    lc_cfg = cfg.get("low_complexity", {})
    lc = low_complexity_score(
        text,
        max_aa_fraction=float(lc_cfg.get("max_aa_fraction", 0.20)),
        max_run=int(lc_cfg.get("max_run", 8)),
    )
    if cfg.get("low_complexity_filter", True) and lc["is_low_complexity"]:
        reasons.extend([f"low_complexity:{reason}" for reason in lc["reasons"]])

    return HardFilterResult(
        passed=not reasons,
        reasons=reasons,
        hepn_positions=hepn_positions,
        length=len(text),
        low_complexity=lc,
    )

