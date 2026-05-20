#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from cas13_ft.atlas import read_jsonl, write_jsonl
from cas13_ft.config import load_yaml
from cas13_ft.splits import make_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--out-dir", default="data/processed")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    split_cfg = cfg.get("split", {})
    input_path = args.input or (cfg.get("paths", {}).get("extracted_prefix", "data/processed/cas13_sequences") + ".jsonl")
    records = read_jsonl(input_path)
    train, valid, test = make_splits(
        records,
        split_cfg.get("train", 0.9),
        split_cfg.get("valid", 0.05),
        split_cfg.get("test", 0.05),
        cfg.get("seed", 1337),
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(train, out / "train.jsonl")
    write_jsonl(valid, out / "valid.jsonl")
    write_jsonl(test, out / "test.jsonl")
    write_jsonl(train, out / "cas13_train.jsonl")
    write_jsonl(valid, out / "cas13_valid.jsonl")
    write_jsonl(test, out / "cas13_test.jsonl")
    train_s = {r["sequence"] for r in train}
    valid_s = {r["sequence"] for r in valid}
    test_s = {r["sequence"] for r in test}
    print(f"split_duplicates_train_valid={len(train_s & valid_s)}")
    print(f"split_duplicates_train_test={len(train_s & test_s)}")
    print(f"split_duplicates_valid_test={len(valid_s & test_s)}")
    print(f"train={len(train)} valid={len(valid)} test={len(test)}")


if __name__ == "__main__":
    main()
