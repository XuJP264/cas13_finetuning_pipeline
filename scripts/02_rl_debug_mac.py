#!/usr/bin/env python
from __future__ import annotations

import argparse

from cas13_rl.rl_trainer import run_from_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rl_cas13_debug_mac.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()
    rows = run_from_config(args.config, resume=args.resume, max_steps=args.max_steps)
    print(f"Mac debug RL completed {len(rows)} steps")


if __name__ == "__main__":
    main()

