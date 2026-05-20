#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import torch

from cas13_ft.atlas import read_jsonl, write_fasta, write_jsonl
from cas13_ft.config import load_yaml
from cas13_ft.modeling import load_causal_lm, load_tokenizer
from cas13_ft.sequence import VALID_AA, clean_protein_sequence, validity_score
from cas13_rl.reward import cas13_motif_score


def simple_identity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    matches = sum(1 for i in range(n) if a[i] == b[i])
    return matches / max(len(a), len(b))


def nearest_identity(sequence: str, train_sequences: list[str]) -> float:
    if not train_sequences:
        return 0.0
    return max(simple_identity(sequence, train_seq) for train_seq in train_sequences)


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "prompt",
        "raw_decoded",
        "sequence",
        "raw_decoded_length",
        "cleaned_protein_length",
        "prompt_length",
        "generated_new_tokens",
        "eos_appeared",
        "stopped_by_eos",
        "stopped_by_max_new_tokens",
        "stop_reason",
        "accepted_by_length_filter",
        "too_short",
        "too_long",
        "prompt_in_cleaned_sequence",
        "length",
        "valid_aa",
        "validity_score",
        "cas13_motif_score",
        "hepn_like_motif_count",
        "train_nearest_identity",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_prompts(path: str | Path, prompt_length: int, limit: int) -> list[str]:
    records = read_jsonl(path)
    prompts = [record["sequence"][:prompt_length] for record in records if record.get("sequence")]
    if not prompts:
        prompts = ["M"]
    return prompts[: max(1, min(limit, len(prompts)))]


def summarize(rows: list[dict], min_len: int, max_len: int) -> dict:
    lengths = [row["cleaned_protein_length"] for row in rows]
    raw_lengths = [row["raw_decoded_length"] for row in rows]
    valid_rate = sum(1 for row in rows if row["valid_aa"]) / len(rows) if rows else 0.0
    unique = len({row["sequence"] for row in rows})
    duplicate_rate = 1.0 - unique / len(rows) if rows else 0.0
    motif_count = sum(row["hepn_like_motif_count"] for row in rows)
    cas13_len_rate = sum(1 for row in rows if min_len <= row["length"] <= max_len) / len(rows) if rows else 0.0
    nearest = [row["train_nearest_identity"] for row in rows]
    stopped_by_eos_count = sum(1 for row in rows if row["stopped_by_eos"])
    stopped_by_max_new_tokens_count = sum(1 for row in rows if row["stopped_by_max_new_tokens"])
    accepted_count = sum(1 for row in rows if row["accepted_by_length_filter"])
    too_short_count = sum(1 for row in rows if row["too_short"])
    too_long_count = sum(1 for row in rows if row["too_long"])
    return {
        "n": len(rows),
        "generated_count": len(rows),
        "accepted_count": accepted_count,
        "accepted_rate": accepted_count / len(rows) if rows else 0.0,
        "valid_amino_acid_rate": valid_rate,
        "raw_length_min": min(raw_lengths) if raw_lengths else 0,
        "raw_length_median": statistics.median(raw_lengths) if raw_lengths else 0,
        "raw_length_mean": statistics.mean(raw_lengths) if raw_lengths else 0,
        "raw_length_max": max(raw_lengths) if raw_lengths else 0,
        "cleaned_length_min": min(lengths) if lengths else 0,
        "cleaned_length_median": statistics.median(lengths) if lengths else 0,
        "cleaned_length_mean": statistics.mean(lengths) if lengths else 0,
        "cleaned_length_max": max(lengths) if lengths else 0,
        "length_min": min(lengths) if lengths else 0,
        "length_median": statistics.median(lengths) if lengths else 0,
        "length_mean": statistics.mean(lengths) if lengths else 0,
        "length_max": max(lengths) if lengths else 0,
        "duplicate_rate": duplicate_rate,
        "train_nearest_identity_mean": statistics.mean(nearest) if nearest else 0,
        "train_nearest_identity_max": max(nearest) if nearest else 0,
        "hepn_like_motif_count": motif_count,
        "cas13_like_length_rate": cas13_len_rate,
        "stopped_by_eos_count": stopped_by_eos_count,
        "stopped_by_max_new_tokens_count": stopped_by_max_new_tokens_count,
        "stopped_by_max_length_count": stopped_by_max_new_tokens_count,
        "eos_rate": stopped_by_eos_count / len(rows) if rows else 0.0,
        "too_short_count": too_short_count,
        "too_long_count": too_long_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft_formal.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--train-file", default=None)
    parser.add_argument("--prompt-length", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=1400)
    parser.add_argument("--min-new-tokens", type=int, default=850)
    parser.add_argument("--target-min-len", type=int, default=850)
    parser.add_argument("--target-max-len", type=int, default=1500)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    paths = cfg.get("paths", {})
    train_cfg = cfg.get("train", {})
    extract_cfg = cfg.get("extract", {})
    model_path = args.model or str(Path(paths.get("output_dir", "outputs/sft/formal")) / "best")
    prompt_file = args.prompt_file or train_cfg.get("valid_file", "data/processed/valid.jsonl")
    train_file = args.train_file or train_cfg.get("train_file", "data/processed/train.jsonl")
    out_dir = Path(args.out_dir) if args.out_dir else Path(paths.get("output_dir", "outputs/sft/formal")) / "generated_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        tokenizer = load_tokenizer(model_path)
        model = load_causal_lm(model_path, vocab_size=len(tokenizer))
        if not hasattr(model.config, "num_hidden_layers") and hasattr(model.config, "n_layer"):
            model.config.num_hidden_layers = model.config.n_layer
        model._supports_default_dynamic_cache = lambda: False
        model.config.use_cache = True
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        model.to(device).eval()
        prompts = load_prompts(prompt_file, args.prompt_length, args.num_samples)
        train_sequences = [row["sequence"] for row in read_jsonl(train_file)]
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        pad_token_id = getattr(tokenizer, "pad_token_id", 0) or 0
        bos_token_id = getattr(tokenizer, "bos_token_id", None)
    except Exception as exc:
        raise SystemExit(f"SFT sample generation setup failed: {exc}") from exc

    generation_parameters = {
        "max_length": None,
        "max_new_tokens": args.max_new_tokens,
        "min_length": None,
        "min_new_tokens": args.min_new_tokens,
        "target_min_len": args.target_min_len,
        "target_max_len": args.target_max_len,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
        "bos_token_id": bos_token_id,
        "do_sample": True,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "stopping_criteria": "eos_token_id or max_new_tokens",
        "prompt_length": args.prompt_length,
        "cleaning": "decode full prompt+generation with skip_special_tokens=True, then remove non-canonical amino acids",
        "fixed_length_truncation": False,
        "counts_prompt_prefix_in_sequence": True,
        "training_max_length_from_config": train_cfg.get("max_length"),
    }
    print("generation parameters:")
    print(json.dumps(generation_parameters, indent=2, ensure_ascii=True), flush=True)

    rows = []
    for i in range(args.num_samples):
        prompt = prompts[i % len(prompts)]
        print(f"generating sample {i + 1}/{args.num_samples}", flush=True)
        input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], dtype=torch.long, device=device)
        generate_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": True,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "use_cache": True,
            "pad_token_id": pad_token_id,
            "eos_token_id": eos_token_id,
        }
        if args.min_new_tokens is not None:
            generate_kwargs["min_new_tokens"] = args.min_new_tokens
        with torch.no_grad():
            output = model.generate(**generate_kwargs)
        output_ids = output[0].detach().cpu().tolist()
        prompt_token_count = int(input_ids.shape[-1])
        new_token_ids = output_ids[prompt_token_count:]
        eos_appeared = eos_token_id is not None and eos_token_id in new_token_ids
        stopped_by_eos = bool(eos_appeared)
        stopped_by_max_new_tokens = not stopped_by_eos and len(new_token_ids) >= args.max_new_tokens
        stop_reason = "eos" if stopped_by_eos else "max_new_tokens" if stopped_by_max_new_tokens else "other"
        decoded = tokenizer.decode(output_ids, skip_special_tokens=True)
        sequence = clean_protein_sequence(decoded)
        cleaned_len = len(sequence)
        too_short = cleaned_len < args.target_min_len
        too_long = cleaned_len > args.target_max_len
        accepted_by_length_filter = not too_short and not too_long
        hepn_count = 2 if cas13_motif_score(sequence) >= 1.0 else 1 if cas13_motif_score(sequence) >= 0.5 else 0
        rows.append(
            {
                "id": f"sft_sample_{i}",
                "prompt": prompt,
                "raw_decoded": decoded,
                "sequence": sequence,
                "raw_decoded_length": len(decoded),
                "cleaned_protein_length": cleaned_len,
                "prompt_length": len(prompt),
                "generated_new_tokens": len(new_token_ids),
                "eos_appeared": bool(eos_appeared),
                "stopped_by_eos": stopped_by_eos,
                "stopped_by_max_new_tokens": stopped_by_max_new_tokens,
                "stop_reason": stop_reason,
                "accepted_by_length_filter": accepted_by_length_filter,
                "too_short": too_short,
                "too_long": too_long,
                "prompt_in_cleaned_sequence": sequence.startswith(clean_protein_sequence(prompt)),
                "length": cleaned_len,
                "valid_aa": bool(sequence) and all(ch in VALID_AA for ch in sequence),
                "validity_score": validity_score(
                    sequence,
                    min_len=args.target_min_len,
                    max_len=args.target_max_len,
                ),
                "cas13_motif_score": cas13_motif_score(sequence),
                "hepn_like_motif_count": hepn_count,
                "train_nearest_identity": nearest_identity(sequence, train_sequences),
            }
        )

    jsonl_path = out_dir / "samples.jsonl"
    csv_path = out_dir / "samples.csv"
    fasta_path = out_dir / "samples.fasta"
    summary_path = out_dir / "summary.json"
    write_jsonl(rows, jsonl_path)
    write_csv(rows, csv_path)
    write_fasta(rows, fasta_path)
    summary = summarize(rows, args.target_min_len, args.target_max_len)
    summary["generation_parameters"] = generation_parameters
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    print(f"wrote: {jsonl_path} {csv_path} {fasta_path} {summary_path}")


if __name__ == "__main__":
    main()
