#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from cas13_ft.atlas import read_jsonl
from cas13_ft.config import load_yaml
from cas13_ft.dataset import CausalProteinCollator, ProteinJsonlDataset
from cas13_ft.modeling import load_tokenizer


def describe(values: list[int]) -> dict:
    return {
        "min": min(values) if values else 0,
        "median": statistics.median(values) if values else 0,
        "mean": statistics.mean(values) if values else 0,
        "max": max(values) if values else 0,
    }


def length_bins(values: list[int]) -> dict:
    bins = {
        "0-200": 0,
        "200-500": 0,
        "500-850": 0,
        "850-1000": 0,
        "1000-1300": 0,
        "1300-1500": 0,
        "1500+": 0,
    }
    for value in values:
        if value < 200:
            bins["0-200"] += 1
        elif value < 500:
            bins["200-500"] += 1
        elif value < 850:
            bins["500-850"] += 1
        elif value < 1000:
            bins["850-1000"] += 1
        elif value < 1300:
            bins["1000-1300"] += 1
        elif value < 1500:
            bins["1300-1500"] += 1
        else:
            bins["1500+"] += 1
    return bins


def audit_split(path: str, tokenizer, max_length: int, append_eos: bool, pad_token_id: int) -> dict:
    records = read_jsonl(path)
    dataset = ProteinJsonlDataset(path, tokenizer, max_length=max_length, append_eos=append_eos)
    collator = CausalProteinCollator(pad_token_id=pad_token_id)
    raw_lengths = [len(record.get("sequence", "")) for record in records]
    tokenized_lengths: list[int] = []
    truncated_count = 0
    eos_input_count = 0
    eos_label_count = 0
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    longer_than_max_length_count = 0
    in_target_length_count = 0

    for idx in range(len(dataset)):
        item = dataset[idx]
        tokenized_lengths.append(item["tokenized_length"])
        truncated_count += int(bool(item["truncated"]))
        longer_than_max_length_count += int(item["raw_tokenized_length"] > max_length)
        in_target_length_count += int(850 <= item["original_length"] <= 1500)
        if eos_token_id is not None:
            eos_input_count += int(eos_token_id in item["input_ids"])
            batch = collator([item])
            labels = batch["labels"][0].tolist()
            eos_label_count += int(eos_token_id in labels)

    count = len(records)
    return {
        "path": path,
        "count": count,
        "raw_sequence_length": describe(raw_lengths),
        "raw_protein_length": describe(raw_lengths),
        "tokenized_length": describe(tokenized_lengths),
        "length_bins": length_bins(raw_lengths),
        "truncated_count": truncated_count,
        "truncated_ratio": truncated_count / count if count else 0.0,
        "eos_in_input_ids_count": eos_input_count,
        "eos_in_input_ids_ratio": eos_input_count / count if count else 0.0,
        "eos_in_labels_count": eos_label_count,
        "eos_in_labels_ratio": eos_label_count / count if count else 0.0,
        "sequences_longer_than_max_length_count": longer_than_max_length_count,
        "sequences_longer_than_max_length_ratio": longer_than_max_length_count / count if count else 0.0,
        "samples_850_1500_count": in_target_length_count,
        "samples_850_1500_ratio": in_target_length_count / count if count else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft_full_length_a100.yaml")
    parser.add_argument("--out", default="outputs/audits/sft_full_length_a100_truncation_eos_audit.json")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    train_cfg = cfg.get("train", {})
    tokenizer = load_tokenizer(
        train_cfg.get("tokenizer_name_or_path"),
        vocab_size=train_cfg.get("tokenizer_vocab_size"),
    )
    max_length = train_cfg.get("max_length", 1536)
    append_eos = train_cfg.get("append_eos", True)
    pad_token_id = getattr(tokenizer, "pad_token_id", 0) or 0
    result = {
        "config": args.config,
        "max_length": max_length,
        "append_eos": append_eos,
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": pad_token_id,
        "splits": {
            "train": audit_split(train_cfg["train_file"], tokenizer, max_length, append_eos, pad_token_id),
            "valid": audit_split(train_cfg["valid_file"], tokenizer, max_length, append_eos, pad_token_id),
            "test": audit_split(train_cfg.get("test_file", "data/processed/test.jsonl"), tokenizer, max_length, append_eos, pad_token_id),
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=True))
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
