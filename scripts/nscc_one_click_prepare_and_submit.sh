#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements-nscc-cu121.txt
python -m pip install -e .

python - <<'PY'
import torch
import transformers
import tokenizers
import huggingface_hub
print("python ok")
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
    print("bf16_supported", torch.cuda.is_bf16_supported())
print("transformers", transformers.__version__)
print("tokenizers", tokenizers.__version__)
print("huggingface_hub", huggingface_hub.__version__)
PY

export PYTHONPATH=src
PYTHONPATH=src python scripts/00_download_assets.py

PYTHONPATH=src python scripts/02b_extract_cas13_keyword_all_lengths.py \
  --atlas data/raw/crispr-cas-atlas-v1.0.json \
  --out-dir data/processed/keyword_all_lengths

PYTHONPATH=src python scripts/07_audit_sft_lengths_and_eos.py \
  --config configs/sft_a100_keyword_all_lengths.yaml \
  --out outputs/audits/a100_keyword_all_lengths_audit.json

test -f scripts/nscc_a100_keyword_all_lengths_smoke.pbs
test -f scripts/nscc_a100_keyword_all_lengths_2epoch.pbs

qsub scripts/nscc_a100_keyword_all_lengths_smoke.pbs

echo "Next commands:"
echo "qstat -u ${USER}"
echo "tail -f outputs/sft/a100_keyword_all_lengths_smoke/logs/train_console.log"
echo "After smoke succeeds:"
echo "qsub scripts/nscc_a100_keyword_all_lengths_2epoch.pbs"
