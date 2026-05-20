#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from cas13_ft.atlas import extract_cas13_records, load_atlas, write_csv, write_fasta, write_jsonl
from cas13_ft.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft.yaml")
    parser.add_argument("--atlas", default=None)
    parser.add_argument("--out-prefix", default=None)
    parser.add_argument("--min-len", type=int, default=None)
    parser.add_argument("--max-len", type=int, default=None)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    extract_cfg = cfg.get("extract", {})
    atlas_path = args.atlas or cfg.get("paths", {}).get("raw_atlas")
    min_len = args.min_len or extract_cfg.get("min_len", 200)
    max_len = args.max_len or extract_cfg.get("max_len", 1500)
    try:
        records = extract_cas13_records(load_atlas(atlas_path), min_len=min_len, max_len=max_len)
    except Exception as exc:
        raise SystemExit(f"Cas13 extraction failed: {exc}") from exc
    prefix = Path(args.out_prefix or cfg.get("paths", {}).get("extracted_prefix", "data/processed/cas13_sequences"))
    prefix.parent.mkdir(parents=True, exist_ok=True)
    write_fasta(records, prefix.with_suffix(".fasta"))
    write_jsonl(records, prefix.with_suffix(".jsonl"))
    write_csv(records, prefix.with_suffix(".csv"))
    lengths = [r["length"] for r in records]
    if lengths:
        import statistics

        print(f"clean_sequences: {len(records)}")
        print(f"deduplicated_sequences: {len(records)}")
        print(
            "length_distribution: "
            f"min={min(lengths)} median={statistics.median(lengths)} "
            f"mean={statistics.mean(lengths):.2f} max={max(lengths)}"
        )
    print(f"extracted {len(records)} Cas13 sequences")


if __name__ == "__main__":
    main()
