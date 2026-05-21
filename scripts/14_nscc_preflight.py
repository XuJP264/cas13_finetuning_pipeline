#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from cas13_ft.config import load_yaml
from cas13_rl.rl_trainer import validate_nscc_environment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rl_cas13_nscc.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    report = {"config": args.config, "nvidia_smi_on_path": shutil.which("nvidia-smi") is not None}
    if report["nvidia_smi_on_path"]:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True)
        report["nvidia_smi"] = result.stdout.strip()
    validate_nscc_environment(cfg)
    report["status"] = "ok"
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
