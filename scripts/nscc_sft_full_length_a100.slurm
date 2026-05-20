#!/bin/bash
#SBATCH --job-name=cas13-sft-a100
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/sft/full_length_a100/logs/slurm-%j.out
#SBATCH --error=outputs/sft/full_length_a100/logs/slurm-%j.err

set -euo pipefail

mkdir -p outputs/sft/full_length_a100/logs outputs/audits

if command -v module >/dev/null 2>&1; then
  module purge || true
  module load cuda || true
fi

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate cas13-ft

export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false

python3 - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("bf16_supported", torch.cuda.is_bf16_supported())
PY

PYTHONPATH=src python3 -m pytest tests/test_sft_pipeline.py

PYTHONPATH=src python3 scripts/01_inspect_atlas_schema.py --config configs/sft_full_length_a100.yaml
PYTHONPATH=src python3 scripts/02_extract_cas13_from_atlas.py --config configs/sft_full_length_a100.yaml
PYTHONPATH=src python3 scripts/03_make_splits.py --config configs/sft_full_length_a100.yaml

PYTHONPATH=src python3 scripts/07_audit_sft_lengths_and_eos.py \
  --config configs/sft_full_length_a100.yaml \
  --out outputs/audits/sft_full_length_a100_truncation_eos_audit.json

PYTHONPATH=src python3 scripts/04_train_sft.py \
  --config configs/sft_full_length_a100.yaml \
  2>&1 | tee outputs/sft/full_length_a100/logs/train_console.log

PYTHONPATH=src python3 scripts/05_eval_sft.py \
  --config configs/sft_full_length_a100.yaml

PYTHONPATH=src python3 scripts/06_generate_sft_samples.py \
  --config configs/sft_full_length_a100.yaml \
  --model outputs/sft/full_length_a100/best \
  --num-samples 64 \
  --max-new-tokens 1400 \
  --min-new-tokens 850 \
  --target-min-len 850 \
  --target-max-len 1500 \
  --temperature 0.9 \
  --top-p 0.95 \
  --top-k 50 \
  --repetition-penalty 1.05 \
  --out-dir outputs/sft/full_length_a100/generated_samples
