#!/usr/bin/env python
from __future__ import annotations

import argparse
import math

from transformers import Trainer, TrainingArguments

from cas13_ft.config import load_yaml
from cas13_ft.dataset import CausalProteinCollator, ProteinJsonlDataset
from cas13_ft.modeling import load_causal_lm, load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default="outputs/sft/best")
    parser.add_argument("--data", default="data/processed/cas13_test.jsonl")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    try:
        cfg = load_yaml(args.config) if args.config else {}
        train_cfg = cfg.get("train", {})
        paths = cfg.get("paths", {})
        model_path = args.model
        data_path = args.data
        max_length = args.max_length
        batch_size = args.batch_size
        if args.config:
            model_path = str(paths.get("output_dir", "outputs/sft") + "/best")
            data_path = "data/processed/test.jsonl"
            max_length = train_cfg.get("max_length", max_length)
            batch_size = train_cfg.get("eval_batch_size", batch_size)
        tokenizer = load_tokenizer(model_path)
        model = load_causal_lm(model_path, vocab_size=len(tokenizer))
        ds = ProteinJsonlDataset(data_path, tokenizer, max_length)
        trainer = Trainer(
            model=model,
            args=TrainingArguments(output_dir="outputs/sft/eval", per_device_eval_batch_size=batch_size, report_to=[]),
            eval_dataset=ds,
            data_collator=CausalProteinCollator(getattr(tokenizer, "pad_token_id", 0) or 0),
        )
        metrics = trainer.evaluate()
        loss = metrics.get("eval_loss")
        if loss is not None:
            metrics["perplexity"] = math.exp(loss) if loss < 20 else float("inf")
        print(metrics)
    except Exception as exc:
        raise SystemExit(f"SFT evaluation failed: {exc}") from exc


if __name__ == "__main__":
    main()
