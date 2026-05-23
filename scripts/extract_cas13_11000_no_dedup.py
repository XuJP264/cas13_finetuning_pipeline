#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from collections import Counter


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract 11000 Cas13 sequences from CRISPR-Cas Atlas without deduplication")
    parser.add_argument("--atlas", default="data/raw/crispr-cas-atlas-v1.0.json", help="Input Atlas JSON file")
    parser.add_argument("--out-dir", default="data/processed/cas13_11000", help="Output directory")
    parser.add_argument("--target-size", type=int, default=11000, help="Target number of sequences")
    parser.add_argument("--min-length", type=int, default=50, help="Minimum sequence length")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for splitting")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Training set ratio")
    parser.add_argument("--valid-ratio", type=float, default=0.05, help="Validation set ratio")
    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load Atlas data
    print(f"Loading Atlas from {args.atlas}", flush=True)
    with open(args.atlas, 'r', encoding='utf-8') as f:
        atlas = json.load(f)

    # Extract Cas13/Type VI/HEPN related sequences
    raw_records = []
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")  # Standard amino acids

    # Process Atlas records - adapt based on actual Atlas structure
    # This assumes standard CRISPR-Cas Atlas structure; adjust parsing as needed
    for organism in atlas.get("organisms", []):
        for cas_system in organism.get("cas_systems", []):
            # Check for Type VI/Cas13 systems
            system_type = cas_system.get("type", "").upper()
            if "VI" in system_type or "CAS13" in system_type.upper():
                for protein in cas_system.get("proteins", []):
                    seq = protein.get("sequence", "").strip().upper()
                    # Filter valid sequences
                    if len(seq) >= args.min_length and all(aa in valid_aa for aa in seq):
                        raw_records.append({
                            "sequence": seq,
                            "length": len(seq),
                            "organism": organism.get("name", ""),
                            "system_type": system_type
                        })

    print(f"Found {len(raw_records)} raw valid Cas13 sequences", flush=True)

    # If we have more than target_size, sample randomly (without dedup)
    if len(raw_records) > args.target_size:
        selected = random.sample(raw_records, args.target_size)
    else:
        selected = raw_records.copy()
        if len(selected) < args.target_size:
            print(f"Warning: Only found {len(selected)} sequences, less than target {args.target_size}", flush=True)

    # Calculate duplicates statistics
    sequences = [r["sequence"] for r in selected]
    seq_counter = Counter(sequences)
    unique_sequences = len(seq_counter)
    duplicate_records = len(selected) - unique_sequences

    # Split into train/valid/test
    random.shuffle(selected)
    n = len(selected)
    n_train = int(n * args.train_ratio)
    n_valid = int(n * args.valid_ratio)
    train = selected[:n_train]
    valid = selected[n_train:n_train + n_valid]
    test = selected[n_train + n_valid:]

    # Write all sequences
    def write_jsonl(records, path):
        with open(path, 'w', encoding='utf-8') as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

    write_jsonl(selected, out_dir / "cas13_sequences.jsonl")
    write_jsonl(train, out_dir / "train.jsonl")
    write_jsonl(valid, out_dir / "valid.jsonl")
    write_jsonl(test, out_dir / "test.jsonl")

    # Write split stats
    split_stats = {
        "mode": "raw_clean_no_dedup",
        "dedup": False,
        "target_size": args.target_size,
        "selected_records": len(selected),
        "selected_unique_exact_sequences": unique_sequences,
        "selected_exact_duplicate_records": duplicate_records,
        "split_counts": {
            "train": len(train),
            "valid": len(valid),
            "test": len(test),
            "total": len(train) + len(valid) + len(test)
        },
        "min_length_filter": args.min_length,
        "seed": args.seed
    }

    with open(out_dir / "split_stats.json", 'w', encoding='utf-8') as f:
        json.dump(split_stats, f, indent=2)

    print(f"Wrote {len(selected)} sequences to {out_dir}", flush=True)
    print(f"  Train: {len(train)}, Valid: {len(valid)}, Test: {len(test)}", flush=True)
    print(f"  Unique sequences: {unique_sequences}, Duplicate records: {duplicate_records}", flush=True)


if __name__ == "__main__":
    main()