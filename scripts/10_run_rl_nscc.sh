#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/rl_cas13_nscc.yaml}"
RESUME_FLAG="${RESUME_FLAG:---resume}"

export PYTHONPATH="${PYTHONPATH:-src}"

python -m cas13_rl.rl_trainer --config "${CONFIG}" --check-only
python -m cas13_rl.rl_trainer --config "${CONFIG}" ${RESUME_FLAG}

