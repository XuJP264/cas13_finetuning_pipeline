#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from cas13_ft.oracle import Cas13Oracle, oracle_score_to_dict


def read_fasta(path: Path) -> list[dict]:
    records = []
    seq_id = None
    chunks: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            if text.startswith(">"):
                if seq_id is not None:
                    records.append({"id": seq_id, "sequence": "".join(chunks)})
                seq_id = text[1:].split()[0] or f"seq_{len(records)}"
                chunks = []
            else:
                chunks.append(text)
    if seq_id is not None:
        records.append({"id": seq_id, "sequence": "".join(chunks)})
    return records


def read_csv(path: Path, sequence_column: str, id_column: str | None) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for i, row in enumerate(csv.DictReader(handle)):
            records.append({"id": row.get(id_column) if id_column else str(i), "sequence": row[sequence_column]})
    return records


def read_records(path: Path, sequence_column: str, id_column: str | None) -> list[dict]:
    if path.suffix.lower() in {".csv", ".tsv"}:
        return read_csv(path, sequence_column, id_column)
    return read_fasta(path)


def print_summary(rows: list[dict]) -> None:
    total = len(rows)
    passed = sum(1 for row in rows if row["passed_hard_filters"])
    rewards = [float(row["final_reward"]) for row in rows]
    reasons = Counter(reason for row in rows for reason in row["hard_fail_reasons"])
    top = sorted(rows, key=lambda row: row["final_reward"], reverse=True)[:10]
    summary = {
        "total": total,
        "hard_filter_pass": passed,
        "hard_filter_fail": total - passed,
        "mean_reward": mean(rewards) if rewards else 0.0,
        "top_10": [{"id": row["sequence_id"], "reward": row["final_reward"]} for row in top],
        "failure_reason_counts": dict(reasons),
    }
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/oracle_cas13.yaml")
    parser.add_argument("--sequence-column", default="sequence")
    parser.add_argument("--id-column", default="id")
    args = parser.parse_args()

    records = read_records(Path(args.input), args.sequence_column, args.id_column)
    oracle = Cas13Oracle.from_config(args.config)
    try:
        scores = [oracle_score_to_dict(score) for score in oracle.score_records(records)]
    finally:
        oracle.close()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in scores:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    print_summary(scores)


if __name__ == "__main__":
    main()
