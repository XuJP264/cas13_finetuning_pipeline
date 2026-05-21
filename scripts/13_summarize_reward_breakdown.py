#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/rl/cas13_debug_mac/reward_breakdown.jsonl")
    args = parser.parse_args()
    path = Path(args.input)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    numeric = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric[key].append(float(value))
    summary = {"rows": len(rows), "valid_basic_rate": mean([1.0 if row.get("valid_basic") else 0.0 for row in rows]) if rows else 0.0}
    for key in sorted(numeric):
        vals = numeric[key]
        summary[f"{key}_mean"] = mean(vals) if vals else 0.0
        summary[f"{key}_min"] = min(vals) if vals else 0.0
        summary[f"{key}_max"] = max(vals) if vals else 0.0
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

