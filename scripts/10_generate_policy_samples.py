#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from cas13_ft.modeling import load_causal_lm, load_tokenizer
from cas13_ft.config import load_yaml
from cas13_rl.generation import generate_mock_samples, generate_policy_samples, load_prompts, write_samples_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rl_ppo.yaml")
    parser.add_argument("--out", default="outputs/rl/policy_samples.jsonl")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    gen = cfg.get("generation", {})
    prompts = load_prompts(cfg["paths"]["prompts_file"], gen.get("prompt_length", 64))
    model_path = Path(cfg["paths"].get("sft_model", "outputs/sft/best"))
    if model_path.exists():
        tokenizer = load_tokenizer(str(model_path))
        model = load_causal_lm(str(model_path), vocab_size=len(tokenizer))
        rows = generate_policy_samples(
            model,
            tokenizer,
            prompts,
            gen.get("num_samples", 16),
            gen.get("max_new_tokens", 128),
            gen.get("temperature", 0.9),
            gen.get("top_p", 0.95),
        )
    else:
        rows = generate_mock_samples(
            prompts,
            gen.get("num_samples", 16),
            gen.get("max_new_tokens", 128),
            cfg.get("seed", 1337),
        )
    write_samples_jsonl(rows, args.out)
    print(f"wrote {len(rows)} samples to {Path(args.out)}")


if __name__ == "__main__":
    main()
