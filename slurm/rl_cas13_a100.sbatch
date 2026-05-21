#!/usr/bin/env bash
#SBATCH --job-name=cas13-rl
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/rl_cas13_%j.out
#SBATCH --error=logs/rl_cas13_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

module purge || true
module load cuda/12.1 || true

export PYTHONPATH="${PYTHONPATH:-src}"
CONFIG="${CONFIG:-configs/rl_cas13_nscc.yaml}"

bash scripts/10_run_rl_nscc.sh "${CONFIG}"

