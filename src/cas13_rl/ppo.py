from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from .generation import generate_mock_samples
from .oracle import OracleCache, build_oracle
from .reward import compute_rewards


def run_mock_ppo(config: Dict) -> List[Dict[str, float]]:
    out_dir = Path(config["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(out_dir / "tb"))
    oracle_cfg = config.get("oracle", {})
    reward_cfg = config.get("reward", {})
    ppo_cfg = config.get("ppo", {})
    gen_cfg = config.get("generation", {})
    cache = OracleCache(config["paths"]["oracle_cache"])
    oracle = build_oracle(config)
    logs = []
    prompts = ["M" * min(16, gen_cfg.get("prompt_length", 64))]
    for step in range(int(ppo_cfg.get("steps", 2))):
        samples = generate_mock_samples(
            prompts,
            num_samples=int(ppo_cfg.get("batch_size", 2)),
            max_new_tokens=int(gen_cfg.get("max_new_tokens", 64)),
            seed=int(config.get("seed", 1337)) + step,
        )
        from .oracle import score_with_cache

        scores = score_with_cache([row["sequence"] for row in samples], oracle, cache)
        rewarded = compute_rewards(scores, reward_cfg, oracle_cfg.get("min_len", 200), oracle_cfg.get("max_len", 1500))
        rewards = np.array([row["reward"] for row in rewarded], dtype=float)
        validity = np.array([row["validity_score"] for row in rewarded], dtype=float)
        plddt = np.array([row["mean_plddt"] for row in rewarded], dtype=float)
        log = {
            "policy_loss": float(-rewards.mean()),
            "value_loss": float(rewards.var()),
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std()),
            "kl_mean": 0.0,
            "entropy": 0.0,
            "mean_plddt": float(plddt.mean()),
            "validity_rate": float((validity > 0).mean()),
            "progen3_nll": float(np.array([row.get("progen3_nll", 0.0) for row in rewarded], dtype=float).mean()),
            "progen3_perplexity": float(np.array([row.get("progen3_perplexity", 0.0) for row in rewarded], dtype=float).mean()),
        }
        for key, value in log.items():
            writer.add_scalar(key, value, step)
        logs.append(log)
    writer.close()
    cache.close()
    return logs
