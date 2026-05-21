#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/rl_cas13_nscc.yaml}"
INPUT_JSONL="${2:-data/processed/valid.jsonl}"
OUTPUT_JSONL="${3:-outputs/rl/cas13_nscc/esmfold_scores.jsonl}"
export PYTHONPATH="${PYTHONPATH:-src}"

python - <<'PY' "${CONFIG}" "${INPUT_JSONL}" "${OUTPUT_JSONL}"
import json
import sys
from pathlib import Path
from cas13_ft.config import load_yaml
from cas13_rl.cache import OracleCache
from cas13_rl.oracle_esmfold import ESMFoldOracle
from cas13_rl.rl_trainer import validate_nscc_environment

cfg = load_yaml(sys.argv[1])
input_path = Path(sys.argv[2])
output_path = Path(sys.argv[3])
validate_nscc_environment(cfg)
output_path.parent.mkdir(parents=True, exist_ok=True)
cache = OracleCache(cfg["paths"].get("esmfold_cache", output_path.parent / "esmfold_cache.sqlite"))
oracle = ESMFoldOracle(cache=cache, **cfg["oracle"]["esmfold"])
count = 0
with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
    for line in src:
        if not line.strip():
            continue
        record = json.loads(line)
        seq = record.get("sequence", "")
        dst.write(json.dumps(oracle.score_one(seq), ensure_ascii=True, sort_keys=True) + "\n")
        count += 1
cache.close()
print(f"wrote {count} ESMFold rows to {output_path}")
PY
