#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from cas13_ft.config import load_yaml
from cas13_rl.cache import OracleCache
from cas13_rl.oracle_progen3 import ProGen3Oracle


def iter_fasta(path: Path) -> Iterable[str]:
    chunks: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            if text.startswith(">"):
                if chunks:
                    yield "".join(chunks)
                    chunks = []
            else:
                chunks.append(text)
    if chunks:
        yield "".join(chunks)


def iter_jsonl(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            yield str(row.get("sequence", ""))


def iter_sequences(path: Path, limit: int | None) -> Iterable[str]:
    suffix = path.suffix.lower()
    source = iter_fasta(path) if suffix in {".fa", ".faa", ".fasta"} else iter_jsonl(path)
    for count, sequence in enumerate(source):
        if limit is not None and count >= limit:
            return
        yield sequence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rl_cas13_nscc.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    input_path = Path(args.input or cfg.get("paths", {}).get("train_data", "data/processed/train.jsonl"))
    output_path = Path(args.output or Path(cfg.get("paths", {}).get("output_dir", "outputs/rl")) / "progen3_likelihood_probe.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    progen_cfg = dict(cfg.get("oracle", {}).get("progen3", {}))
    cache_path = cfg.get("paths", {}).get("progen3_cache")
    if not cache_path:
        cache_dir = progen_cfg.get("cache_dir")
        cache_path = Path(cache_dir) / "progen3_cache.sqlite" if cache_dir else output_path.parent / "progen3_cache.sqlite"
    cache = OracleCache(cache_path)
    oracle = ProGen3Oracle(cache=cache, **progen_cfg)
    written = 0
    try:
        with output_path.open("w", encoding="utf-8") as handle:
            for sequence in iter_sequences(input_path, args.limit):
                try:
                    row = oracle.score_one(sequence)
                except Exception as exc:
                    row = {
                        "sequence": sequence,
                        "valid": False,
                        "mean_logprob": None,
                        "perplexity": None,
                        "error": f"{type(exc).__name__}: {exc}",
                        "backend": "real" if progen_cfg.get("mode") == "real" else "mock",
                    }
                handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
                written += 1
    finally:
        cache.close()
    print(f"wrote {written} ProGen3 likelihood rows to {output_path}")


if __name__ == "__main__":
    main()
