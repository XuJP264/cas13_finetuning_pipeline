#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments, set_seed

from cas13_ft.config import load_yaml
from cas13_ft.dataset import CausalProteinCollator, ProteinJsonlDataset
from cas13_ft.modeling import load_causal_lm, load_tokenizer


def resolve_precision(value, kind: str) -> bool:
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available() and not torch.cuda.is_available():
        return False
    if value != "auto":
        return bool(value) and torch.cuda.is_available()
    if kind == "bf16":
        return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    if kind == "fp16":
        return bool(torch.cuda.is_available() and not torch.cuda.is_bf16_supported())
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    train_cfg = cfg.get("train", {})
    paths = cfg.get("paths", {})
    set_seed(cfg.get("seed", 1337))
    output_dir = args.output_dir or paths.get("output_dir", "outputs/sft")
    try:
        tokenizer = load_tokenizer(train_cfg.get("tokenizer_name_or_path"), vocab_size=train_cfg.get("tokenizer_vocab_size"))
        model = load_causal_lm(
            train_cfg.get("model_name_or_path"),
            paths.get("checkpoint"),
            paths.get("hf_config"),
            vocab_size=len(tokenizer),
        )
        max_length = train_cfg.get("max_length", 1024)
        append_eos = train_cfg.get("append_eos", True)
        train_ds = ProteinJsonlDataset(train_cfg["train_file"], tokenizer, max_length, append_eos=append_eos)
        valid_ds = ProteinJsonlDataset(train_cfg["valid_file"], tokenizer, max_length, append_eos=append_eos)
        collator = CausalProteinCollator(getattr(tokenizer, "pad_token_id", 0) or 0)
        gradient_checkpointing = bool(train_cfg.get("gradient_checkpointing", False))
        if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = False
        fp16 = resolve_precision(train_cfg.get("fp16", False), "fp16")
        bf16 = resolve_precision(train_cfg.get("bf16", False), "bf16")
        device_name = "cuda" if torch.cuda.is_available() else "mps" if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available() else "cpu"
        dtype = "bf16" if bf16 else "fp16" if fp16 else "fp32"
        train_trunc = train_ds.truncation_summary()
        valid_trunc = valid_ds.truncation_summary()
        print(
            "SFT training setup: "
            f"device={device_name} dtype={dtype} max_length={max_length} "
            f"gradient_checkpointing={gradient_checkpointing} bf16={bf16} fp16={fp16} "
            f"append_eos={append_eos}"
        )
        print(
            "SFT truncation summary: "
            f"train={train_trunc['truncated_count']}/{train_trunc['count']} "
            f"({train_trunc['truncation_ratio']:.4f}) "
            f"valid={valid_trunc['truncated_count']}/{valid_trunc['count']} "
            f"({valid_trunc['truncation_ratio']:.4f})"
        )
        targs = TrainingArguments(
            output_dir=output_dir,
            logging_dir=train_cfg.get("logging_dir", str(Path(output_dir) / "runs")),
            per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", train_cfg.get("batch_size", 1)),
            per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", train_cfg.get("eval_batch_size", 1)),
            learning_rate=train_cfg.get("learning_rate", 5e-5),
            weight_decay=train_cfg.get("weight_decay", 0.01),
            num_train_epochs=train_cfg.get("num_train_epochs", 1),
            max_steps=train_cfg.get("max_steps", -1),
            gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
            fp16=fp16,
            bf16=bf16,
            gradient_checkpointing=gradient_checkpointing,
            logging_steps=train_cfg.get("logging_steps", 1),
            eval_steps=train_cfg.get("eval_steps", 100),
            save_steps=train_cfg.get("save_steps", 500),
            eval_strategy="steps",
            save_strategy="steps",
            report_to=["tensorboard"],
            save_total_limit=train_cfg.get("save_total_limit", 3),
            dataloader_pin_memory=train_cfg.get("dataloader_pin_memory", True),
            load_best_model_at_end=False,
        )
        trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=valid_ds, data_collator=collator)
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or train_cfg.get("resume_from_checkpoint"))
        metrics = trainer.evaluate()
        if "eval_loss" in metrics:
            metrics["perplexity"] = math.exp(metrics["eval_loss"]) if metrics["eval_loss"] < 20 else float("inf")
        trainer.log_metrics("eval", metrics)
        trainer.save_model(str(Path(output_dir) / "best"))
        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(Path(output_dir) / "best")
    except Exception as exc:
        raise SystemExit(f"SFT training failed: {exc}") from exc


if __name__ == "__main__":
    main()
