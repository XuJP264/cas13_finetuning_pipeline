#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from cas13_ft.config import load_yaml
from cas13_rl.reward import compute_cas13_rewards

AA = "ACDEFGHIKLMNPQRSTVWY"


def iter_fasta(path: Path) -> Iterable[str]:
    chunks: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            if text.startswith(">"):
                if chunks:
                    yield "".join(chunks)
                    chunks = []
            else:
                chunks.append(text)
    if chunks:
        yield "".join(chunks)


def default_sequences() -> list[str]:
    short = (AA * 4)[:80]
    reasonable = "RAAAAH" + (AA * 55)[:1088] + "RCCCCH"
    illegal = "RAAAAH" + (AA * 20) + "XBZ" + "RCCCCH"
    return [short, reasonable, illegal]


def mock_esmfold(sequence: str) -> dict:
    seq = str(sequence or "")
    valid = bool(seq) and all(ch in set(AA) for ch in seq)
    length_bonus = max(0.0, 1.0 - abs(len(seq) - 950) / 950.0)
    motif_bonus = 8.0 if "RAAAAH" in seq and "RCCCCH" in seq else 0.0
    mean_plddt = 55.0 + 25.0 * length_bonus + motif_bonus
    ptm = min(0.9, 0.25 + 0.55 * length_bonus)
    mean_pae = max(4.0, 24.0 - 16.0 * length_bonus)
    return {
        "sequence": seq,
        "valid": valid,
        "mean_plddt": float(mean_plddt) if valid else None,
        "ptm": float(ptm) if valid else None,
        "mean_pae": float(mean_pae) if valid else None,
        "pdb_path": None,
        "error": None if valid else "mock invalid sequence",
        "backend": "mock",
    }


def mock_progen3(sequence: str) -> dict:
    seq = str(sequence or "")
    valid = bool(seq) and all(ch in set(AA) for ch in seq)
    rare_penalty = sum(1 for ch in seq if ch in {"W", "C", "M"}) / max(1, len(seq))
    mean_logprob = -1.4 - rare_penalty
    return {
        "sequence": seq,
        "valid": valid,
        "mean_logprob": float(mean_logprob) if valid else None,
        "perplexity": float(2.718281828 ** (-mean_logprob)) if valid else None,
        "error": None if valid else "mock invalid sequence",
        "backend": "mock",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rl_cas13_debug_mac.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    sequences = list(iter_fasta(Path(args.input))) if args.input else default_sequences()
    rewards = compute_cas13_rewards(
        sequences,
        [mock_esmfold(seq) for seq in sequences],
        [mock_progen3(seq) for seq in sequences],
        config=cfg.get("reward", {}),
    )
    fields = [
        "sequence_length",
        "score_length",
        "score_hepn",
        "score_lm",
        "score_plddt",
        "score_ptm",
        "r_prop",
        "reward_for_rl",
        "invalid_reason",
    ]
    rows = [{field: row.get(field) for field in fields} for row in rewards]
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    else:
        for row in rows:
            print(json.dumps(row, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
