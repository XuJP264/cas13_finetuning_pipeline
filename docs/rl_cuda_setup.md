# RL CUDA Setup

This Mac/MPS machine cannot complete the real ProGen3 or ESMFold oracle path. The SFT pipeline is working here, so keep RL real-oracle dependencies in a separate CUDA environment instead of installing them into the SFT environment.

## Why Mac/MPS Is Blocked

ProGen3 is not a plain Transformers model in the current environment. The HF weight repo has `model_type=progen3`, but no bundled remote-code files, so real scoring requires the official Profluent code from `https://github.com/Profluent-AI/progen3`.

The official ProGen3 README states local usage requires at least one CUDA GPU and was tested on A100/H100-class devices with bf16 and flash-attention-style kernels. On this Mac/MPS machine, importing the official code fails before scoring because `megablocks` is unavailable.

ESMFold through `fair-esm[esmfold]` additionally needs OpenFold structure modules. On this machine, ESMFold fails with `ModuleNotFoundError: openfold`, and `pip install openfold` fails during package metadata generation. A CUDA-compatible PyTorch plus nvcc/OpenFold setup is expected.

## Recommended Environment

Use an isolated conda environment on a CUDA host:

```bash
conda create -n cas13-rl-cuda python=3.10 -y
conda activate cas13-rl-cuda
python -m pip install --upgrade pip setuptools wheel
```

Install CUDA PyTorch matching the host driver. Example for CUDA 12.1:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Install this project in editable mode:

```bash
cd /path/to/cas13_ft_pipeline
python -m pip install -e ".[test,rl]"
```

## ProGen3

Clone and install the official code:

```bash
git clone --depth 1 https://github.com/Profluent-AI/progen3 external/progen3
cd external/progen3
bash setup.sh
cd -
```

Verify imports:

```bash
PYTHONPATH=external/progen3/src python - <<'PY'
from progen3.modeling import ProGen3ForCausalLM
from progen3.batch_preparer import ProGen3BatchPreparer
from progen3.scorer import ProGen3Scorer
print("progen3 imports ok")
PY
```

Run the real ProGen3 oracle smoke:

```bash
RUN_REAL_ORACLE_TESTS=1 PYTHONPATH=src python -m pytest tests/test_real_oracle_imports.py
```

## ESMFold

Install ESMFold dependencies in the same CUDA environment only after ProGen3 works:

```bash
python -m pip install "fair-esm[esmfold]"
```

Install OpenFold according to the CUDA host and compiler setup. Validate `nvcc` first:

```bash
nvcc --version
python - <<'PY'
import torch
print(torch.__version__, torch.cuda.is_available())
PY
```

Then verify ESMFold:

```bash
PYTHONPATH=src python - <<'PY'
from cas13_rl.oracles.esmfold import ESMFoldOracle
oracle = ESMFoldOracle(enabled=True, device="cuda", max_length=64, output_dir="outputs/rl/esmfold_smoke")
row = oracle.score_one("MKTAYIAKQRQISFVKSHFSRQ", seq_id="smoke")
print(row["mean_plddt"], row.get("pdb_path"))
PY
```

## Real PPO Smoke

Keep ESMFold disabled at first and validate ProGen3-only PPO:

```bash
PYTHONPATH=src python scripts/12_train_rl_ppo.py --config configs/rl_ppo_real_smoke.yaml
```

After ESMFold validation passes, edit `configs/rl_ppo_real_smoke.yaml`:

```yaml
oracle:
  progen3:
    enabled: true
  esmfold:
    enabled: true
```

Then rerun:

```bash
PYTHONPATH=src python scripts/12_train_rl_ppo.py --config configs/rl_ppo_real_smoke.yaml
```
