#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cas13_ft.config import load_yaml
from cas13_rl.ppo import run_mock_ppo
from cas13_rl.rl_trainer import run_from_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rl_ppo.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    if cfg.get("training", {}).get("mode") in {"real_ppo", "ppo"}:
        try:
            logs = run_from_config(args.config, resume=args.resume, max_steps=args.max_steps)
        except Exception as exc:
            raise SystemExit(f"RL PPO training failed: {exc}") from exc
        print(f"completed {len(logs)} real PPO steps")
        return
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
