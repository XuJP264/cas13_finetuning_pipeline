from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from cas13_ft.config import load_yaml
from cas13_ft.modeling import load_causal_lm, load_tokenizer

from .cache import OracleCache
from .generation import generate_mock_samples, load_prompts
from .oracle_esmfold import ESMFoldOracle
from .oracle_progen3 import ProGen3Oracle
from .reward import compute_cas13_rewards


def _require_path(label: str, path: str | None) -> None:
    if not path or not Path(path).exists():
        raise FileNotFoundError(f"{label} path does not exist: {path}")


def _looks_like_hf_repo_id(value: str | None) -> bool:
    return bool(value and "/" in value and not value.startswith("/") and not value.startswith("."))


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
    esm_cfg = oracle.get("esmfold", {})
    if esm_cfg.get("mode") == "real":
        _require_path("ESMFold model", esm_cfg.get("model_path"))
    progen_cfg = oracle.get("progen3", {})
    if progen_cfg.get("mode") == "real":
        progen_model = progen_cfg.get("model_name_or_path") or progen_cfg.get("model_path")
        if not _looks_like_hf_repo_id(progen_model):
            _require_path("ProGen3 model", progen_model)
        if progen_cfg.get("code_path"):
            _require_path("ProGen3 code", progen_cfg.get("code_path"))


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
        self.reward_log_path = self.output_dir / "reward_breakdown.jsonl"
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

    def _save_real_checkpoint(self, step: int, metrics: Dict[str, Any], model, tokenizer, optimizer) -> None:
        import torch

        checkpoint_dir = self.output_dir / f"checkpoint_step_{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(checkpoint_dir)
        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(checkpoint_dir)
        torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
        state = {
            "step": step,
            "completed_steps": step,
            "last_metrics": metrics,
            "last_checkpoint": str(checkpoint_dir),
        }
        self.state_path.write_text(json.dumps(state, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
        (checkpoint_dir / "trainer_state.json").write_text(json.dumps(state, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")

    def _prompts(self) -> List[str]:
        train_data = self.paths.get("train_data")
        prompt_length = int(self.generation.get("prompt_length", 64))
        if train_data and Path(train_data).exists():
            return load_prompts(train_data, prompt_length=prompt_length)
        return ["M" * min(prompt_length, 16)]

    def _basic_invalid_reason(self, sequence: str) -> str | None:
        reward_cfg = self.config.get("reward", {})
        min_len = int(reward_cfg.get("min_len", 200))
        max_len = int(reward_cfg.get("max_len", 1500))
        seq = str(sequence or "").upper()
        canonical = set("ACDEFGHIKLMNPQRSTVWY")
        if not seq:
            return "empty sequence"
        if any(ch not in canonical for ch in seq):
            return "non-canonical amino acid"
        if len(seq) < min_len or len(seq) > max_len:
            return f"length {len(seq)} outside [{min_len}, {max_len}]"
        return None

    def _score_oracles_with_basic_gate(self, sequences: List[str]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        esm_rows: List[Dict[str, Any] | None] = [None] * len(sequences)
        lm_rows: List[Dict[str, Any] | None] = [None] * len(sequences)
        valid_indices: List[int] = []
        valid_sequences: List[str] = []
        for i, sequence in enumerate(sequences):
            reason = self._basic_invalid_reason(sequence)
            if reason:
                esm_rows[i] = {
                    "sequence": sequence,
                    "valid": False,
                    "mean_plddt": None,
                    "ptm": None,
                    "mean_pae": None,
                    "pdb_path": None,
                    "error": reason,
                    "backend": self.oracle_cfg.get("esmfold", {}).get("mode", "mock"),
                }
                lm_rows[i] = {
                    "sequence": sequence,
                    "valid": False,
                    "mean_logprob": None,
                    "perplexity": None,
                    "error": reason,
                    "backend": self.oracle_cfg.get("progen3", {}).get("mode", "mock"),
                }
            else:
                valid_indices.append(i)
                valid_sequences.append(sequence)
        if valid_sequences:
            scored_esm = self.esmfold.score_many(valid_sequences)
            scored_lm = self.progen3.score_many(valid_sequences)
            for index, esm, lm in zip(valid_indices, scored_esm, scored_lm):
                esm_rows[index] = esm
                lm_rows[index] = lm
        return [row for row in esm_rows if row is not None], [row for row in lm_rows if row is not None]

    def run(self, resume: bool = False, max_steps: int | None = None) -> List[Dict[str, Any]]:
        if self.training.get("mode") in {"real_ppo", "ppo"}:
            return self._run_real_ppo(resume=resume, max_steps=max_steps)
        return self._run_mock_debug(resume=resume, max_steps=max_steps)

    def _run_mock_debug(self, resume: bool = False, max_steps: int | None = None) -> List[Dict[str, Any]]:
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
            esm_rows, lm_rows = self._score_oracles_with_basic_gate(sequences)
            rewards = compute_cas13_rewards(
                sequences,
                esm_rows,
                lm_rows,
                config=self.config.get("reward", {}),
                log_path=self.reward_log_path,
                kl_values=[0.0 for _ in sequences],
            )
            kl_weight = float(self.config.get("reward", {}).get("kl_weight", self.config.get("reward", {}).get("w_kl", 0.0)))
            reward_values = [float(row["reward_for_rl"]) - kl_weight * 0.0 for row in rewards]
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

    def _policy_model_path(self, state: Dict[str, Any], resume: bool) -> str | None:
        if resume and state.get("last_checkpoint"):
            checkpoint = Path(state["last_checkpoint"])
            if checkpoint.exists():
                return str(checkpoint)
        model_path = self.paths.get("policy_model") or self.training.get("policy_model")
        if model_path and Path(model_path).exists():
            return str(model_path)
        return self.training.get("model_name_or_path")

    def _load_policy_components(self, state: Dict[str, Any], resume: bool):
        import torch

        model_path = self._policy_model_path(state, resume)
        tokenizer_path = self.training.get("tokenizer_name_or_path") or model_path
        tokenizer = load_tokenizer(tokenizer_path, vocab_size=self.training.get("tokenizer_vocab_size", 32))
        model = load_causal_lm(model_path, vocab_size=len(tokenizer))
        ref_model = copy.deepcopy(model)
        device_cfg = self.training.get("device", "auto")
        if device_cfg == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available() else "cpu"
        else:
            device = str(device_cfg)
        model.to(device)
        ref_model.to(device)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad_(False)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.training.get("learning_rate", 1e-6)),
            weight_decay=float(self.training.get("weight_decay", 0.0)),
        )
        if resume and state.get("last_checkpoint"):
            opt_path = Path(state["last_checkpoint"]) / "optimizer.pt"
            if opt_path.exists():
                optimizer.load_state_dict(torch.load(opt_path, map_location=device))
        return model, ref_model, tokenizer, optimizer, device

    @staticmethod
    def _encode_prompt(tokenizer, prompt: str) -> List[int]:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if not ids:
            eos = getattr(tokenizer, "eos_token_id", None)
            ids = [int(eos) if eos is not None else 1]
        return [int(x) for x in ids]

    def _generate_policy_batch(self, model, tokenizer, prompts: List[str], batch_size: int, device: str) -> tuple[List[str], List[List[int]], List[int]]:
        import torch

        rows: List[str] = []
        token_rows: List[List[int]] = []
        prompt_lengths: List[int] = []
        model.eval()
        for i in range(batch_size):
            prompt = prompts[i % len(prompts)]
            prompt_ids = self._encode_prompt(tokenizer, prompt)
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                output = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=int(self.generation.get("max_new_tokens", 128)),
                    do_sample=bool(self.generation.get("do_sample", True)),
                    temperature=float(self.generation.get("temperature", 0.9)),
                    top_p=float(self.generation.get("top_p", 0.95)),
                    pad_token_id=getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None) or 0,
                    eos_token_id=getattr(tokenizer, "eos_token_id", None),
                )
            ids = [int(x) for x in output[0].detach().cpu().tolist()]
            token_rows.append(ids)
            prompt_lengths.append(len(prompt_ids))
            rows.append(tokenizer.decode(ids, skip_special_tokens=True))
        model.train()
        return rows, token_rows, prompt_lengths

    @staticmethod
    def _sequence_logprobs(model, token_rows: List[List[int]], prompt_lengths: List[int], device: str, pad_token_id: int) -> tuple[Any, Any]:
        import torch

        max_len = max(len(row) for row in token_rows)
        input_ids = torch.full((len(token_rows), max_len), int(pad_token_id), dtype=torch.long, device=device)
        attention_mask = torch.zeros((len(token_rows), max_len), dtype=torch.long, device=device)
        for i, row in enumerate(token_rows):
            input_ids[i, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)
            attention_mask[i, : len(row)] = 1
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        targets = input_ids[:, 1:]
        target_mask = attention_mask[:, 1:].bool()
        for i, prompt_len in enumerate(prompt_lengths):
            target_positions = torch.arange(targets.shape[1], device=device) + 1
            target_mask[i] &= target_positions >= int(prompt_len)
        log_probs = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        logprob_sums = (log_probs * target_mask).sum(dim=1)
        token_counts = target_mask.sum(dim=1).clamp_min(1)
        return logprob_sums, token_counts

    def _run_real_ppo(self, resume: bool = False, max_steps: int | None = None) -> List[Dict[str, Any]]:
        import torch

        state = self._load_state(resume)
        start_step = int(state.get("step", 0)) if resume else 0
        steps = int(max_steps or self.training.get("steps", 1))
        batch_size = int(self.training.get("batch_size", 1))
        prompts = self._prompts()
        model, ref_model, tokenizer, optimizer, device = self._load_policy_components(state, resume)
        pad_token_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None) or 0
        clip_range = float(self.training.get("ppo_clip_range", 0.2))
        kl_weight = float(self.training.get("kl_weight", self.config.get("reward", {}).get("kl_weight", 0.02)))
        save_steps = int(self.training.get("save_steps", 1))
        metrics_rows: List[Dict[str, Any]] = []

        for step in range(start_step, steps):
            sequences, token_rows, prompt_lengths = self._generate_policy_batch(model, tokenizer, prompts, batch_size, device)
            with torch.no_grad():
                old_logprobs, token_counts = self._sequence_logprobs(model, token_rows, prompt_lengths, device, int(pad_token_id))
                ref_logprobs, _ = self._sequence_logprobs(ref_model, token_rows, prompt_lengths, device, int(pad_token_id))
                kl_values_tensor = (old_logprobs - ref_logprobs) / token_counts
            kl_values = [float(x) for x in kl_values_tensor.detach().cpu().tolist()]
            esm_rows, lm_rows = self._score_oracles_with_basic_gate(sequences)
            rewards = compute_cas13_rewards(
                sequences,
                esm_rows,
                lm_rows,
                config=self.config.get("reward", {}),
                log_path=self.reward_log_path,
                kl_values=kl_values,
            )
            property_rewards = torch.tensor([float(row["reward_for_rl"]) for row in rewards], dtype=torch.float32, device=device)
            kl_tensor = kl_values_tensor.detach().to(device).float()
            total_rewards = property_rewards - kl_weight * kl_tensor
            advantages = total_rewards - total_rewards.mean()
            if advantages.numel() > 1 and float(advantages.std(unbiased=False).detach().cpu()) > 1e-8:
                advantages = advantages / advantages.std(unbiased=False).clamp_min(1e-8)

            new_logprobs, _ = self._sequence_logprobs(model, token_rows, prompt_lengths, device, int(pad_token_id))
            ratio = torch.exp((new_logprobs - old_logprobs.detach()).clamp(-20, 20))
            unclipped = ratio * advantages.detach()
            clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages.detach()
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            current_ref_logprobs, current_counts = self._sequence_logprobs(ref_model, token_rows, prompt_lengths, device, int(pad_token_id))
            current_kl = ((new_logprobs - current_ref_logprobs.detach()) / current_counts).mean()
            loss = policy_loss + kl_weight * current_kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            max_grad_norm = self.training.get("max_grad_norm")
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
            optimizer.step()

            valid_values = [1.0 if row["valid"] else 0.0 for row in rewards]
            metrics = {
                "step": step + 1,
                "mode": "real_ppo",
                "loss": float(loss.detach().cpu()),
                "policy_loss": float(policy_loss.detach().cpu()),
                "reward_mean": float(total_rewards.detach().mean().cpu()),
                "property_reward_mean": float(property_rewards.detach().mean().cpu()),
                "kl_mean": float(kl_tensor.detach().mean().cpu()),
                "current_kl_mean": float(current_kl.detach().cpu()),
                "valid_rate": sum(valid_values) / max(1, len(valid_values)),
                "batch_size": len(sequences),
                "resume_from_step": start_step,
            }
            with self.metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics, ensure_ascii=True, sort_keys=True) + "\n")
            if (step + 1) % save_steps == 0 or (step + 1) == steps:
                self._save_real_checkpoint(step + 1, metrics, model, tokenizer, optimizer)
            else:
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
