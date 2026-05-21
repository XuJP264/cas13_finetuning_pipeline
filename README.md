# Cas13 FT Pipeline

This repository contains a Cas13 SFT pipeline and an RL PPO scaffold with cached oracle scoring.

## Current Machine Status

Run date: 2026-05-20. Machine has PyTorch 2.8.0 with MPS available and no CUDA GPU.

- Mock tests: completed, `10 passed, 1 skipped`.
- Real SFT smoke train: completed with the downloaded ProGen2 checkpoint and real Cas13 splits.
- Formal SFT training: pending but runnable via `configs/sft_formal.yaml`.
- ProGen3 real oracle: code path added, but not runnable on this machine. The official ProGen3 repo requires CUDA-class GPU dependencies such as `megablocks`/flash-attention; this MPS machine fails with `ModuleNotFoundError: megablocks`.
- ESMFold real oracle: code path added, but not runnable on this machine. `fair-esm[esmfold]` installs most deps, but ESMFold still requires OpenFold; `pip install openfold` fails on this macOS/MPS environment.
- RL mock PPO: passed.
- RL real PPO: pending CUDA validation; attempted on Mac and failed at ProGen3 load, without falling back to mock.

## Assets

Downloaded files:

```text
data/raw/progen2-base-ft.ckpt                3.0G  sha256 49185c41a548407efd9d1017c05154f1067e824bf744e280cddce724430c31d8
data/raw/progen2-base-ft-config.json         829B  sha256 d399fad9aaeb115e29cbfe3c9a917b6a6d749f6bd23aa26012beae70bd3b4b64
data/raw/crispr-cas-atlas-v1.0.json          4.9G  sha256 5b4ba2fb99638d279e0c126100e19a4b77aba487b37b7df118e4bf4acd494720
```

Commands:

```bash
PYTHONPATH=src python3 scripts/00_download_assets.py
shasum -a 256 data/raw/progen2-base-ft.ckpt data/raw/progen2-base-ft-config.json data/raw/crispr-cas-atlas-v1.0.json
```

## Real Cas13 Data Processing

Commands run:

```bash
PYTHONPATH=src python3 scripts/01_inspect_atlas_schema.py --config configs/sft.yaml
PYTHONPATH=src python3 scripts/02_extract_cas13_from_atlas.py --config configs/sft.yaml
PYTHONPATH=src python3 scripts/03_make_splits.py --config configs/sft.yaml
```

Output summary:

```text
Atlas operons: 1,246,088
Total cas entries: 6,174,375
Raw Cas13/Type VI candidates: 20,676
Raw candidates with protein: 20,676
Clean + length-filtered pre-dedup: 18,924
Deduplicated sequences: 5,716
Length min/median/mean/max: 200 / 913.0 / 770.23 / 1498
Train/valid/test: 5,144 / 285 / 287
Split duplicates train-valid/train-test/valid-test: 0 / 0 / 0
```

Expected files exist:

```text
data/processed/cas13_sequences.jsonl
data/processed/cas13_sequences.fasta
data/processed/cas13_sequences.csv
data/processed/train.jsonl
data/processed/valid.jsonl
data/processed/test.jsonl
```

## Real SFT Smoke

Command run:

```bash
PYTHONPATH=src python3 scripts/04_train_sft.py --config configs/sft_smoke.yaml
PYTHONPATH=src python3 scripts/05_eval_sft.py --config configs/sft_smoke.yaml
```

Notes:

- Transformers 4.57 does not include native `model_type=progen`; a bundled ProGen adapter under `src/cas13_ft/progen_remote/` was added from HF ProGen remote-code.
- Checkpoint loaded with `missing=0 unexpected=54`.
- Device selected by Trainer: MPS.
- Smoke train ran `max_steps=5`.

Train/eval records:

```text
step 1 train loss 6.9129
step 2 train loss 4.6661
step 2 valid eval_loss 4.0618
step 3 train loss 4.2349
step 4 train loss 5.3511
step 4 valid eval_loss 4.2869
step 5 train loss 4.0439
final train_loss 5.0418
final valid eval_loss 4.0098, perplexity 55.1339
test eval_loss 4.0202, perplexity 55.7134
```

Artifacts:

```text
outputs/sft/smoke/checkpoint-5/
outputs/sft/smoke/best/
outputs/sft/smoke/runs/
```

## Formal SFT

Formal training config:

```text
configs/sft_formal.yaml
```

It uses the real `data/processed/train.jsonl` and `data/processed/valid.jsonl` splits, writes to `outputs/sft/formal`, and uses:

```text
max_steps: 1000
logging_steps: 10
eval_steps: 100
save_steps: 100
gradient_accumulation_steps: 8
fp16/bf16: auto
```

On CUDA, `auto` enables bf16 when supported, otherwise fp16. On MPS/CPU it keeps fp32.

Run:

```bash
PYTHONPATH=src python3 scripts/04_train_sft.py --config configs/sft_formal.yaml
PYTHONPATH=src python3 scripts/05_eval_sft.py --config configs/sft_formal.yaml
```

Generate and evaluate SFT samples:

```bash
PYTHONPATH=src python3 scripts/06_generate_sft_samples.py \
  --config configs/sft_formal.yaml \
  --model outputs/sft/formal/best \
  --num-samples 32
```

Outputs:

```text
outputs/sft/formal/generated_samples/samples.jsonl
outputs/sft/formal/generated_samples/samples.csv
outputs/sft/formal/generated_samples/samples.fasta
outputs/sft/formal/generated_samples/summary.json
```

## RL

Cas13 RL engineering entry points:

```bash
# Mac debug: mock ESMFold/ProGen3, no CUDA dependency, writes reward components JSONL.
PYTHONPATH=src python3 scripts/02_rl_debug_mac.py --config configs/rl_cas13_debug_mac.yaml

# Resume Mac debug from outputs/rl/cas13_debug_mac/trainer_state.json.
PYTHONPATH=src python3 scripts/02_rl_debug_mac.py --config configs/rl_cas13_debug_mac.yaml --resume

# NSCC one-click shell launch. Edit model/data paths in configs/rl_cas13_nscc.yaml first.
bash scripts/10_run_rl_nscc.sh configs/rl_cas13_nscc.yaml

# NSCC SLURM launch.
sbatch slurm/rl_cas13_a100.sbatch
```

The new RL path stores expensive oracle calls in SQLite caches keyed by sequence
SHA256, records `struct/lm/motif/length/diversity/kl` reward components in
`reward_components.jsonl`, and keeps resume state in `trainer_state.json`.
NSCC mode checks CUDA, `nvidia-smi`, model paths, and data paths before training.

ProGen3 real oracle deployment checks and acceptance probe:

```bash
# Mac dry check: verifies config/model metadata and official-code imports when present.
PYTHONPATH=src python scripts/16_download_or_check_progen3.py \
  --config configs/rl_cas13_nscc.yaml

# Optional Hugging Face cache population.
PYTHONPATH=src python scripts/16_download_or_check_progen3.py \
  --config configs/rl_cas13_nscc.yaml \
  --download

# NSCC acceptance: valid rows should contain mean_logprob and perplexity.
PYTHONPATH=src python scripts/12_probe_progen3_likelihood.py \
  --config configs/rl_cas13_nscc.yaml \
  --input data/processed/valid_probe.fasta \
  --output outputs/oracle_probe/progen3_probe.jsonl \
  --limit 10
```

Mock PPO command:

```bash
PYTHONPATH=src python3 scripts/12_train_rl_ppo.py --config configs/rl_ppo.yaml
```

Mock PPO completed:

```text
outputs/rl/ppo_metrics.jsonl
outputs/rl/oracle_cache.sqlite  # 4 cached mock scores
outputs/rl/tb/
```

Real ProGen3 smoke command attempted:

```bash
git clone --depth 1 https://github.com/Profluent-AI/progen3 external/progen3
PYTHONPATH=src python3 scripts/12_train_rl_ppo.py --config configs/rl_ppo_real_smoke.yaml
```

Actual failure:

```text
Found official ProGen3 code at external/progen3/src, but failed to load
'Profluent-Bio/progen3-219m' on device 'mps'. ProGen3 upstream requires
CUDA-class GPU dependencies such as megablocks/flash attention per its README.
Reason: ModuleNotFoundError: No module named 'megablocks'
```

ESMFold attempt:

```bash
python3 -m pip install fair-esm 'fair-esm[esmfold]'
PYTHONPATH=src python3 - <<'PY'
from cas13_rl.oracles.esmfold import ESMFoldOracle
ESMFoldOracle(enabled=True, device='auto', max_length=32)
PY
```

Actual failure:

```text
ModuleNotFoundError: No module named 'openfold'
pip install openfold failed during package metadata generation on this machine.
```

CUDA setup details are in:

```text
docs/rl_cuda_setup.md
```

## Tests

```bash
PYTHONPATH=src python3 -m pytest
```

Current result:

```text
10 passed, 1 skipped, 2 warnings
```

The skipped test is the opt-in real ProGen3 oracle smoke. Run it only on a CUDA machine with the official ProGen3 dependencies:

```bash
RUN_REAL_ORACLE_TESTS=1 PYTHONPATH=src python3 -m pytest tests/test_real_oracle_imports.py
```

## NSCC A100 Full-Length SFT

Mac/MPS is only a smoke/debug environment for this project. Full-length Cas13 SFT should run on NSCC or another Linux CUDA host with an A100-class GPU. GitHub should contain code, configs, tests, and docs only; do not commit Atlas JSON, processed splits, checkpoints, TensorBoard logs, or generated outputs.

The main NSCC training dataset is `keyword_all_lengths`: Cas13/C2c2 keyword-only proteins, exact cleaned protein sequence deduplicated, with no extraction-stage length filter. Exact protein sequence deduplication is reasonable here because identical cleaned amino-acid sequences provide duplicate training targets even if they appear in multiple operons. Length filtering is intentionally not applied during extraction, because short/long records are useful audit signals and truncation should be measured at tokenization time rather than hidden upstream.

Training uses `max_length=1536`. Sequences longer than that are truncated only by the tokenizer/trainer path, with EOS retained where possible, and `scripts/07_audit_sft_lengths_and_eos.py` reports the truncation ratio before training.

Recommended repository flow:

```bash
git status --short
git add .gitignore README.md configs scripts src tests docs pyproject.toml setup.py setup.cfg requirements-nscc-cu121.txt
git commit -m "Prepare Cas13 keyword all-length SFT for NSCC A100"
git push origin main
```

Pinned NSCC dependency versions:

```text
python: 3.10 preferred; 3.13 is smoke/debug only
torch: 2.5.1+cu121 or compatible CUDA 12.1 build
transformers==4.49.0
tokenizers==0.21.4
huggingface-hub==0.36.1
accelerate>=0.26.0
safetensors
tensorboard
pytest
pyyaml
```

On NSCC:

```bash
git clone <YOUR_GITHUB_REPO_URL> cas13_ft_pipeline
cd cas13_ft_pipeline
bash scripts/nscc_one_click_prepare_and_submit.sh
```

The one-click script creates `.venv`, installs pinned dependencies, downloads assets, builds the keyword-only all-length exact-dedup dataset, runs the audit, and submits the smoke PBS job with `qsub`.

If compute nodes have no internet, download assets on a login node or download locally and `rsync`:

```bash
PYTHONPATH=src python scripts/00_download_assets.py
shasum -a 256 data/raw/progen2-base-ft.ckpt data/raw/progen2-base-ft-config.json data/raw/crispr-cas-atlas-v1.0.json
```

Keyword-only all-length extraction:

```bash
PYTHONPATH=src python scripts/02b_extract_cas13_keyword_all_lengths.py \
  --atlas data/raw/crispr-cas-atlas-v1.0.json \
  --out-dir data/processed/keyword_all_lengths
```

Audit EOS and truncation:

```bash
PYTHONPATH=src python scripts/07_audit_sft_lengths_and_eos.py \
  --config configs/sft_a100_keyword_all_lengths.yaml \
  --out outputs/audits/a100_keyword_all_lengths_audit.json
```

Smoke job:

```bash
qsub scripts/nscc_a100_keyword_all_lengths_smoke.pbs
qstat -u $USER
tail -f outputs/sft/a100_keyword_all_lengths_smoke/logs/train_console.log
```

After smoke succeeds, submit the temporary 2-epoch A100 run:

```bash
qsub scripts/nscc_a100_keyword_all_lengths_2epoch.pbs
```

Resume from a checkpoint if needed:

```bash
PYTHONPATH=src python scripts/04_train_sft.py \
  --config configs/sft_a100_keyword_all_lengths.yaml \
  --resume-from-checkpoint outputs/sft/a100_keyword_all_lengths/checkpoint-500
```

TensorBoard from your laptop via SSH port forwarding:

```bash
ssh -L 6006:127.0.0.1:6006 <user>@<nscc-login-host>
cd cas13_ft_pipeline
source .venv/bin/activate
python -m tensorboard.main --logdir outputs/sft/a100_keyword_all_lengths/runs --host 127.0.0.1 --port 6006
```

If the A100 40GB job OOMs, first reduce `max_length` from `1536` to `1280`; if needed increase `gradient_accumulation_steps` to `16`, keep `per_device_train_batch_size: 1`, keep `gradient_checkpointing: false` for the current ProGen wrapper, and reduce eval batch size to `1`.
