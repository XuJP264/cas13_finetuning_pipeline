#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored", default="outputs/rl/scored_samples.jsonl")
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.scored).read_text(encoding="utf-8").splitlines() if line.strip()]
    rewards = np.array([row.get("reward", 0.0) for row in rows], dtype=float)
    plddt = np.array([row.get("mean_plddt", 0.0) for row in rows], dtype=float)
    valid = np.array([row.get("validity_score", 0.0) for row in rows], dtype=float)
    print(
        {
            "n": len(rows),
            "reward_mean": float(rewards.mean()) if len(rewards) else 0.0,
            "reward_std": float(rewards.std()) if len(rewards) else 0.0,
            "mean_plddt": float(plddt.mean()) if len(plddt) else 0.0,
            "validity_rate": float((valid > 0).mean()) if len(valid) else 0.0,
        }
    )


if __name__ == "__main__":
    main()
