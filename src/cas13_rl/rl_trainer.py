from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from cas13_ft.config import load_yaml

from .cache import OracleCache
from .generation import generate_mock_samples, load_prompts
from .oracle_esmfold import ESMFoldOracle
from .oracle_progen3 import ProGen3Oracle
from .reward import compute_cas13_rewards


def _require_path(label: str, path: str | None) -> None:
    if not path or not Path(path).exists():
        raise FileNotFoundError(f"{label} path does not exist: {path}")


def validate_nscc_environment(config: Dict[str, Any]) -> None:
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("NSCC mode requires nvidia-smi on PATH")
    subprocess.run(["nvidia-smi"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(f"NSCC mode requires PyTorch with CUDA: {exc}") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("NSCC mode requires torch.cuda.is_available() == True")
    paths = config.get("paths", {})
    oracle = config.get("oracle", {})
    _require_path("train_data", paths.get("train_data"))
    _require_path("policy_model", paths.get("policy_model"))
    _require_path("ESMFold model", oracle.get("esmfold", {}).get("model_path"))
    _require_path("ProGen3 model", oracle.get("progen3", {}).get("model_path"))


class Cas13RLTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.paths = config.get("paths", {})
        self.training = config.get("training", config.get("ppo", {}))
        self.generation = config.get("generation", {})
        self.oracle_cfg = config.get("oracle", {})
        self.output_dir = Path(self.paths.get("output_dir", "outputs/rl/cas13"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.output_dir / "trainer_state.json"
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.reward_log_path = self.output_dir / "reward_components.jsonl"
        if config.get("runtime", {}).get("mode") == "nscc":
            validate_nscc_environment(config)
        self.esm_cache = OracleCache(self.paths.get("esmfold_cache", self.output_dir / "esmfold_cache.sqlite"))
        self.progen_cache = OracleCache(self.paths.get("progen3_cache", self.output_dir / "progen3_cache.sqlite"))
        self.esmfold = ESMFoldOracle(cache=self.esm_cache, **self.oracle_cfg.get("esmfold", {}))
        self.progen3 = ProGen3Oracle(cache=self.progen_cache, **self.oracle_cfg.get("progen3", {}))

    def close(self) -> None:
        self.esm_cache.close()
        self.progen_cache.close()

    def _load_state(self, resume: bool) -> Dict[str, Any]:
        if resume and self.state_path.exists():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {"step": 0, "completed_steps": 0}

    def _save_state(self, step: int, metrics: Dict[str, Any]) -> None:
        state = {"step": step, "completed_steps": step, "last_metrics": metrics}
        self.state_path.write_text(json.dumps(state, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
        checkpoint = self.output_dir / f"checkpoint_step_{step}.json"
        checkpoint.write_text(json.dumps(state, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")

    def _prompts(self) -> List[str]:
        train_data = self.paths.get("train_data")
        prompt_length = int(self.generation.get("prompt_length", 64))
        if train_data and Path(train_data).exists():
            return load_prompts(train_data, prompt_length=prompt_length)
        return ["M" * min(prompt_length, 16)]

    def run(self, resume: bool = False, max_steps: int | None = None) -> List[Dict[str, Any]]:
        state = self._load_state(resume)
        start_step = int(state.get("step", 0)) if resume else 0
        steps = int(max_steps or self.training.get("steps", 1))
        batch_size = int(self.training.get("batch_size", 2))
        prompts = self._prompts()
        metrics_rows: List[Dict[str, Any]] = []
        for step in range(start_step, steps):
            samples = generate_mock_samples(
                prompts,
                num_samples=batch_size,
                max_new_tokens=int(self.generation.get("max_new_tokens", 128)),
                seed=int(self.config.get("seed", 1337)) + step,
            )
            sequences = [row["sequence"] for row in samples]
            esm_rows = self.esmfold.score_many(sequences)
            lm_rows = self.progen3.score_many(sequences)
            rewards = compute_cas13_rewards(
                sequences,
                esm_rows,
                lm_rows,
                config=self.config.get("reward", {}),
                log_path=self.reward_log_path,
                kl_values=[0.0 for _ in sequences],
            )
            reward_values = [float(row["reward"]) for row in rewards]
            valid_values = [1.0 if row["valid"] else 0.0 for row in rewards]
            metrics = {
                "step": step + 1,
                "reward_mean": sum(reward_values) / max(1, len(reward_values)),
                "valid_rate": sum(valid_values) / max(1, len(valid_values)),
                "batch_size": len(sequences),
                "resume_from_step": start_step,
            }
            with self.metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics, ensure_ascii=True, sort_keys=True) + "\n")
            self._save_state(step + 1, metrics)
            metrics_rows.append(metrics)
        return metrics_rows


def run_from_config(config_path: str | Path, resume: bool = False, max_steps: int | None = None) -> List[Dict[str, Any]]:
    cfg = load_yaml(config_path)
    trainer = Cas13RLTrainer(cfg)
    try:
        return trainer.run(resume=resume, max_steps=max_steps)
    finally:
        trainer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    if args.check_only:
        if cfg.get("runtime", {}).get("mode") == "nscc":
            validate_nscc_environment(cfg)
        print("environment checks passed")
        return
    rows = run_from_config(args.config, resume=args.resume, max_steps=args.max_steps)
    print(f"completed {len(rows)} RL steps")


if __name__ == "__main__":
    main()

