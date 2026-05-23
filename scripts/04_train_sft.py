#!/usr/bin/env python
from __future__ import annotations

import argparse
import inspect
import math
import os
import subprocess
import sys
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


def _as_report_to(value):
    if value is None:
        return ["tensorboard"]
    if isinstance(value, list):
        return value
    return [value]


def _training_arg_name(preferred: str, fallback: str) -> str:
    params = inspect.signature(TrainingArguments.__init__).parameters
    if preferred in params:
        return preferred
    if fallback in params:
        return fallback
    raise RuntimeError(f"TrainingArguments supports neither {preferred!r} nor {fallback!r}")


def _estimated_total_steps(train_size: int, train_cfg: dict) -> int:
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")) or "1")
    per_device = int(train_cfg.get("per_device_train_batch_size", train_cfg.get("batch_size", 1)))
    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 1))
    epochs = float(train_cfg.get("num_train_epochs", 1))
    denom = max(1, per_device * grad_accum * max(1, world_size))
    return int(math.ceil(train_size * epochs / denom))


def _maybe_plot_curves(output_dir: str) -> None:
    plot_script = Path("scripts/plot_sft_curves.py")
    if not plot_script.exists():
        return
    result = subprocess.run(
        [sys.executable, str(plot_script), "--output_dir", output_dir],
        text=True,
        capture_output=True,
    )
    if result.returncode == 0:
        print(result.stdout.strip())
    else:
        print("LOSS_CURVE_PLOT_FAILED=" + (result.stderr.strip() or result.stdout.strip()))


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
        if not train_cfg.get("valid_file"):
            raise ValueError("train.valid_file is required; refusing to train without validation set")
        if not Path(train_cfg["valid_file"]).exists():
            raise FileNotFoundError(f"Validation file does not exist: {train_cfg['valid_file']}")
        if not train_cfg.get("train_file"):
            raise ValueError("train.train_file is required")
        if not Path(train_cfg["train_file"]).exists():
            raise FileNotFoundError(f"Train file does not exist: {train_cfg['train_file']}")
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
        estimated_total_steps = _estimated_total_steps(len(train_ds), train_cfg)
        logging_dir = train_cfg.get("logging_dir") or paths.get("logging_dir") or str(Path(output_dir) / "runs")
        eval_steps = int(train_cfg.get("eval_steps", 100))
        save_steps = int(train_cfg.get("save_steps", eval_steps))
        load_best_model_at_end = bool(train_cfg.get("load_best_model_at_end", False))
        if load_best_model_at_end and save_steps != eval_steps:
            raise ValueError(f"save_steps ({save_steps}) must equal eval_steps ({eval_steps}) for best checkpoint selection")
        logging_steps = int(train_cfg.get("logging_steps", 10))
        print(
            "SFT training setup: "
            f"device={device_name} dtype={dtype} max_length={max_length} "
            f"gradient_checkpointing={gradient_checkpointing} bf16={bf16} fp16={fp16} "
            f"append_eos={append_eos}"
        )
        print(
            "SFT data sizing: "
            f"train_size={len(train_ds)} valid_size={len(valid_ds)} "
            f"gradient_accumulation_steps={train_cfg.get('gradient_accumulation_steps', 1)} "
            f"estimated_total_steps={estimated_total_steps}"
        )
        print(
            "SFT truncation summary: "
            f"train={train_trunc['truncated_count']}/{train_trunc['count']} "
            f"({train_trunc['truncation_ratio']:.4f}) "
            f"valid={valid_trunc['truncated_count']}/{valid_trunc['count']} "
            f"({valid_trunc['truncation_ratio']:.4f})"
        )
        print(
            "SFT monitoring: "
            f"logging_steps={logging_steps} eval_steps={eval_steps} save_steps={save_steps} "
            f"load_best_model_at_end={load_best_model_at_end} "
            "metric_for_best_model=eval_loss greater_is_better=False"
        )
        eval_strategy_name = _training_arg_name("eval_strategy", "evaluation_strategy")
        targs_kwargs = dict(
            output_dir=output_dir,
            logging_dir=logging_dir,
            run_name=train_cfg.get("run_name"),
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
            logging_strategy=train_cfg.get("logging_strategy", "steps"),
            logging_steps=logging_steps,
            eval_steps=eval_steps,
            save_strategy=train_cfg.get("save_strategy", "steps"),
            save_steps=save_steps,
            report_to=_as_report_to(train_cfg.get("report_to", "tensorboard")),
            save_total_limit=train_cfg.get("save_total_limit", 5),
            dataloader_pin_memory=train_cfg.get("dataloader_pin_memory", True),
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=train_cfg.get("metric_for_best_model", "eval_loss"),
            greater_is_better=bool(train_cfg.get("greater_is_better", False)),
        )
        targs_kwargs[eval_strategy_name] = train_cfg.get(eval_strategy_name, train_cfg.get("eval_strategy", train_cfg.get("evaluation_strategy", "steps")))
        targs = TrainingArguments(**targs_kwargs)
        trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=valid_ds, data_collator=collator)
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or train_cfg.get("resume_from_checkpoint"))
        metrics = trainer.evaluate()
        if "eval_loss" in metrics:
            metrics["perplexity"] = math.exp(metrics["eval_loss"]) if metrics["eval_loss"] < 20 else float("inf")
        trainer.log_metrics("eval", metrics)
        trainer.save_model(str(Path(output_dir) / "best"))
        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(Path(output_dir) / "best")
        best_model_checkpoint = trainer.state.best_model_checkpoint
        best_eval_loss = trainer.state.best_metric
        if best_model_checkpoint and not Path(best_model_checkpoint).exists():
            raise FileNotFoundError(f"Best checkpoint recorded by Trainer does not exist: {best_model_checkpoint}")
        checkpoint_count = len(list(Path(output_dir).glob("checkpoint-*")))
        print(f"BEST_MODEL_CHECKPOINT={best_model_checkpoint}")
        print(f"BEST_EVAL_LOSS={best_eval_loss}")
        print(f"OUTPUT_DIR={output_dir}")
        print(f"LOGGING_DIR={logging_dir}")
        print(f"CHECKPOINT_COUNT={checkpoint_count}")
        print("VALIDATION_SET_USED=YES")
        print("BEST_CHECKPOINT_METRIC=eval_loss")
        _maybe_plot_curves(output_dir)
    except Exception as exc:
        raise SystemExit(f"SFT training failed: {exc}") from exc


if __name__ == "__main__":
    main()
