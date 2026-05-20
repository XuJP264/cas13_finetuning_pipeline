#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter

from cas13_ft.atlas import inspect_atlas, iter_operons, load_atlas
from cas13_ft.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--atlas", default="data/raw/crispr-cas-atlas-v1.0.json")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    cfg = load_yaml(args.config) if args.config else {}
    atlas_path = cfg.get("paths", {}).get("raw_atlas", args.atlas)
    atlas = load_atlas(atlas_path)
    stats = inspect_atlas(atlas)
    operons = list(iter_operons(atlas))
    print(f"operons: {len(operons)}")
    print(f"total_cas_entries: {stats['total_cas_entries']}")
    print(f"raw_cas13_or_type_vi_sequences: {stats['raw_candidate_sequences']}")
    print(f"root_type: {type(atlas).__name__}")
    key_counts = Counter()
    subtypes = Counter()
    for op in operons:
        key_counts.update(op.keys())
        summary = op.get("summary") or {}
        subtypes.update([summary.get("subtype")])
    print("top operon keys:", key_counts.most_common(20))
    print("top subtypes:", subtypes.most_common(20))
    for op in operons[: args.limit]:
        print(op.keys())


if __name__ == "__main__":
    main()
