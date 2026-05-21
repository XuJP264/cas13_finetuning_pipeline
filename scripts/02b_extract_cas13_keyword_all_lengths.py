#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any

from cas13_ft.atlas import CAS13_KEYWORDS, iter_operons, load_atlas, write_fasta, write_jsonl
from cas13_ft.sequence import clean_protein_sequence


def percentile(values: list[int], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def length_summary(values: list[int]) -> dict:
    return {
        "min": min(values) if values else 0,
        "p05": percentile(values, 0.05),
        "p25": percentile(values, 0.25),
        "median": statistics.median(values) if values else 0,
        "mean": statistics.mean(values) if values else 0,
        "p75": percentile(values, 0.75),
        "p95": percentile(values, 0.95),
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


def contains_keyword(value: Any) -> bool:
    text = str(value or "")
    return any(keyword in text for keyword in CAS13_KEYWORDS)


def keyword_match(gene: dict[str, Any]) -> bool:
    return contains_keyword(gene.get("gene_name")) or contains_keyword(gene.get("hmm_name"))


def operon_id(operon: dict[str, Any], index: int) -> str:
    for key in ("operon_id", "id", "operon", "accession", "contig_id"):
        value = operon.get(key)
        if value not in (None, ""):
            return str(value)
    return f"operon_{index}"


def top_counter(counter: Counter, n: int = 20) -> list[dict]:
    return [{"value": str(value), "count": count} for value, count in counter.most_common(n)]


def write_csv(records: list[dict], path: Path) -> None:
    fields = [
        "id",
        "operon_id",
        "subtype",
        "subtype_score",
        "gene_name",
        "hmm_name",
        "length",
        "sequence",
        "protein",
        "duplicate_count",
        "duplicate_operon_ids_sample",
        "sha256",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def build_records(atlas: Any, max_operons: int | None = None) -> tuple[list[dict], dict]:
    raw_keyword_count = 0
    nonempty_protein_count = 0
    post_clean_count = 0
    changed_by_cleaning = 0
    seq_groups: dict[str, list[dict]] = defaultdict(list)
    gene_counter: Counter = Counter()
    hmm_counter: Counter = Counter()
    subtype_counter: Counter = Counter()

    for op_index, operon in enumerate(iter_operons(atlas)):
        if max_operons is not None and op_index >= max_operons:
            break
        cas_genes = operon.get("cas") or operon.get("cas_genes") or []
        if not isinstance(cas_genes, list):
            continue
        summary = operon.get("summary") or {}
        op_id = operon_id(operon, op_index)
        for gene_index, gene in enumerate(cas_genes):
            if not isinstance(gene, dict) or not keyword_match(gene):
                continue
            raw_keyword_count += 1
            protein_raw = gene.get("protein")
            if not protein_raw:
                continue
            nonempty_protein_count += 1
            cleaned = clean_protein_sequence(protein_raw)
            if not cleaned:
                continue
            post_clean_count += 1
            raw_compact = str(protein_raw).upper().replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")
            if cleaned != raw_compact:
                changed_by_cleaning += 1
            digest = sha256(cleaned.encode("utf-8")).hexdigest()
            gene_name = gene.get("gene_name")
            hmm_name = gene.get("hmm_name")
            subtype = summary.get("subtype")
            gene_counter.update([gene_name or ""])
            hmm_counter.update([hmm_name or ""])
            subtype_counter.update([subtype or ""])
            seq_groups[digest].append(
                {
                    "operon_id": op_id,
                    "subtype": subtype,
                    "subtype_score": summary.get("subtype_score"),
                    "gene_name": gene_name,
                    "hmm_name": hmm_name,
                    "length": len(cleaned),
                    "sequence": cleaned,
                    "protein": protein_raw,
                    "sha256": digest,
                    "gene_index": gene_index,
                }
            )

    records: list[dict] = []
    duplicate_examples = []
    duplicate_sizes = Counter()
    for digest, group in seq_groups.items():
        duplicate_sizes.update([len(group)])
        first = group[0]
        duplicate_operons = [item["operon_id"] for item in group[:10]]
        if len(group) > 1 and len(duplicate_examples) < 20:
            duplicate_examples.append(
                {
                    "sha256": digest,
                    "size": len(group),
                    "length": first["length"],
                    "operon_ids_sample": duplicate_operons,
                    "gene_names_sample": [item.get("gene_name") for item in group[:10]],
                    "hmm_names_sample": [item.get("hmm_name") for item in group[:10]],
                }
            )
        records.append(
            {
                "id": f"cas13kw_{len(records):06d}_{digest[:12]}",
                "operon_id": first["operon_id"],
                "subtype": first["subtype"],
                "subtype_score": first["subtype_score"],
                "gene_name": first["gene_name"],
                "hmm_name": first["hmm_name"],
                "length": first["length"],
                "sequence": first["sequence"],
                "protein": first["protein"],
                "duplicate_count": len(group),
                "duplicate_operon_ids_sample": duplicate_operons,
                "sha256": digest,
            }
        )

    summary = {
        "raw_keyword_cas13_or_c2c2_entries": raw_keyword_count,
        "nonempty_protein_count": nonempty_protein_count,
        "post_clean_entries": post_clean_count,
        "unique_exact_cleaned_sequences": len(records),
        "duplicate_exact_cleaned_sequence_removed": post_clean_count - len(records),
        "changed_by_cleaning_noncanonical_or_symbols": changed_by_cleaning,
        "length": length_summary([record["length"] for record in records]),
        "length_bins": length_bins([record["length"] for record in records]),
        "top_gene_name": top_counter(gene_counter),
        "top_hmm_name": top_counter(hmm_counter),
        "top_subtype": top_counter(subtype_counter),
        "duplicate_group_size_distribution": {str(size): count for size, count in sorted(duplicate_sizes.items())},
        "example_duplicate_groups": duplicate_examples,
    }
    return records, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas", default="data/raw/crispr-cas-atlas-v1.0.json")
    parser.add_argument("--out-dir", default="data/processed/keyword_all_lengths")
    parser.add_argument("--dedup", choices=["exact"], default="exact")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.90)
    parser.add_argument("--valid-ratio", type=float, default=0.05)
    parser.add_argument("--max-operons", type=int, default=None, help="Optional dry-run limit for local tests.")
    args = parser.parse_args()

    atlas = load_atlas(args.atlas)
    records, summary = build_records(atlas, max_operons=args.max_operons)
    rng = random.Random(args.seed)
    rng.shuffle(records)
    n = len(records)
    n_train = int(n * args.train_ratio)
    n_valid = int(n * args.valid_ratio)
    train = records[:n_train]
    valid = records[n_train : n_train + n_valid]
    test = records[n_train + n_valid :]
    summary["split"] = {"train": len(train), "valid": len(valid), "test": len(test)}
    summary["dedup"] = args.dedup
    summary["seed"] = args.seed
    summary["max_operons"] = args.max_operons

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(records, out_dir / "all.jsonl")
    write_jsonl(train, out_dir / "train.jsonl")
    write_jsonl(valid, out_dir / "valid.jsonl")
    write_jsonl(test, out_dir / "test.jsonl")
    write_csv(records, out_dir / "all.csv")
    write_fasta(records, out_dir / "all.fasta")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    print(f"wrote keyword-only all-length exact-dedup dataset to {out_dir}")


if __name__ == "__main__":
    main()
