#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cas13_ft.config import load_yaml
from cas13_rl.ppo import run_mock_ppo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rl_ppo.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    try:
        logs = run_mock_ppo(cfg)
    except Exception as exc:
        raise SystemExit(f"RL PPO smoke failed: {exc}") from exc
    out = Path(cfg["paths"]["output_dir"]) / "ppo_metrics.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for row in logs:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(f"completed {len(logs)} PPO smoke steps; metrics: {out}")


if __name__ == "__main__":
    main()
