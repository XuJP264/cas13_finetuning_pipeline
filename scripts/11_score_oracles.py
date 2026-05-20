#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cas13_ft.config import load_yaml
from cas13_rl.oracle import OracleCache, build_oracle, score_with_cache
from cas13_rl.reward import compute_rewards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rl_ppo.yaml")
    parser.add_argument("--samples", default="outputs/rl/policy_samples.jsonl")
    parser.add_argument("--out", default="outputs/rl/scored_samples.jsonl")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    oracle_cfg = cfg.get("oracle", {})
    rows = [json.loads(line) for line in Path(args.samples).read_text(encoding="utf-8").splitlines() if line.strip()]
    try:
        oracle = build_oracle(cfg)
    except Exception as exc:
        raise SystemExit(f"Oracle loading failed for mode={oracle_cfg.get('mode')}: {exc}") from exc
    cache = OracleCache(cfg["paths"]["oracle_cache"])
    scores = score_with_cache([row["sequence"] for row in rows], oracle, cache)
    rewarded = compute_rewards(scores, cfg.get("reward", {}), oracle_cfg.get("min_len", 200), oracle_cfg.get("max_len", 1500))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("w", encoding="utf-8") as handle:
        for row in rewarded:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    cache.close()
    print(f"scored {len(rewarded)} samples")


if __name__ == "__main__":
    main()
