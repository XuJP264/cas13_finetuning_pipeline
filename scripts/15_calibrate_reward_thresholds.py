#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


FIELDS = {
    "lm": "raw_lm_mean_logprob",
    "plddt": "raw_mean_plddt",
    "ptm": "raw_ptm",
    "pae": "raw_mean_pae",
}


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = min(len(vals) - 1, max(0, round((len(vals) - 1) * q)))
    return float(vals[idx])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/rl/cas13_debug_mac/reward_breakdown.jsonl")
    parser.add_argument("--output", default="outputs/rl/cas13_debug_mac/reward_calibration.yaml")
    parser.add_argument("--low-q", type=float, default=0.1)
    parser.add_argument("--high-q", type=float, default=0.9)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    calibration = {}
    for name, field in FIELDS.items():
        vals = [float(row[field]) for row in rows if row.get(field) is not None]
        calibration[name] = {"low": quantile(vals, args.low_q), "high": quantile(vals, args.high_q)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump({"reward": {"calibration": calibration}}, sort_keys=True), encoding="utf-8")
    print(f"wrote calibration thresholds to {output}")


if __name__ == "__main__":
    main()

