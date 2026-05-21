#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from cas13_ft.config import load_yaml


def looks_like_hf_repo_id(value: str | None) -> bool:
    return bool(value and "/" in value and not value.startswith("/") and not value.startswith("."))


def check_local_model(path: Path) -> dict:
    files = list(path.glob("*")) if path.exists() else []
    return {
        "exists": path.exists(),
        "path": str(path),
        "has_config": (path / "config.json").exists(),
        "has_tokenizer": any(item.name.startswith("tokenizer") for item in files) or (path / "special_tokens_map.json").exists(),
        "has_weights": any(item.suffix in {".bin", ".safetensors"} for item in files),
    }


def code_import_report(code_path: str | None) -> dict:
    candidates = []
    if code_path:
        root = Path(code_path).expanduser()
        candidates.extend([root / "src", root])
    candidates.append(Path("external/progen3/src"))
    report = {"checked_paths": [str(p) for p in candidates], "importable": False, "error": None}
    old_path = list(sys.path)
    try:
        for candidate in candidates:
            if candidate.exists():
                sys.path.insert(0, str(candidate.resolve()))
                break
        for module in ["progen3.batch_preparer", "progen3.modeling", "progen3.scorer"]:
            if importlib.util.find_spec(module) is None:
                raise ImportError(f"cannot import {module}")
        report["importable"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        sys.path = old_path
    return report


def cuda_report(require_cuda: bool) -> dict:
    report = {"required": require_cuda, "available": None, "error": None}
    try:
        import torch

        report["available"] = bool(torch.cuda.is_available())
        if require_cuda and not report["available"]:
            report["error"] = "CUDA is required on NSCC but torch.cuda.is_available() is False"
    except Exception as exc:
        report["available"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rl_cas13_nscc.yaml")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    progen_cfg = cfg.get("oracle", {}).get("progen3", {})
    model_name_or_path = progen_cfg.get("model_name_or_path") or progen_cfg.get("model_path")
    cache_dir = progen_cfg.get("cache_dir")
    report = {
        "config": args.config,
        "mode": progen_cfg.get("mode"),
        "model_name_or_path": model_name_or_path,
        "cache_dir": cache_dir,
        "model": {},
        "official_code": code_import_report(progen_cfg.get("code_path")),
    }

    if not model_name_or_path:
        report["model"] = {"ok": False, "error": "oracle.progen3.model_name_or_path or model_path is required"}
    elif looks_like_hf_repo_id(model_name_or_path):
        report["model"] = {"type": "huggingface_repo", "repo_id": model_name_or_path, "ok": True}
        try:
            from huggingface_hub import snapshot_download

            if args.download:
                local_dir = snapshot_download(repo_id=model_name_or_path, cache_dir=cache_dir)
                report["model"]["snapshot_path"] = local_dir
                report["model"].update(check_local_model(Path(local_dir)))
            else:
                report["model"]["hint"] = (
                    "Run with --download or rely on transformers.from_pretrained to populate the Hugging Face cache."
                )
        except Exception as exc:
            report["model"]["ok"] = False
            report["model"]["error"] = f"{type(exc).__name__}: {exc}"
    else:
        local = check_local_model(Path(model_name_or_path).expanduser())
        local["ok"] = bool(local["exists"] and local["has_config"] and local["has_weights"])
        report["model"] = local

    is_mac = platform.system() == "Darwin"
    require_cuda = bool(args.require_cuda or (not is_mac and (os.environ.get("SLURM_JOB_ID") or progen_cfg.get("device") == "cuda")))
    report["cuda"] = cuda_report(require_cuda=require_cuda)
    report["ok"] = bool(report["model"].get("ok", False) and (not report["cuda"].get("error")))
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
