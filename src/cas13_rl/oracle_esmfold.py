from __future__ import annotations

import math
import os
import json
import tempfile
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .cache import OracleCache


def _mock_struct_scores(sequence: str) -> Dict[str, Any]:
    seq = str(sequence or "")
    valid = bool(seq)
    motif_bonus = 8.0 if "R" in seq and "H" in seq else 0.0
    length_penalty = min(20.0, abs(len(seq) - 900) / 50.0)
    mean_plddt = max(1.0, min(95.0, 62.0 + motif_bonus - length_penalty))
    ptm = max(0.05, min(0.9, mean_plddt / 100.0 - 0.08))
    mean_pae = max(2.0, 30.0 - mean_plddt / 4.0)
    return {
        "mean_plddt": float(mean_plddt),
        "ptm": float(ptm),
        "mean_pae": float(mean_pae),
        "pdb_path": None,
        "valid": valid,
        "error": None if valid else "empty sequence",
        "backend": "mock",
    }


def _schema(sequence: str, payload: Dict[str, Any], backend: str) -> Dict[str, Any]:
    valid = bool(payload.get("valid", False))
    return {
        "sequence": str(payload.get("sequence", sequence)),
        "valid": valid,
        "mean_plddt": float(payload["mean_plddt"]) if payload.get("mean_plddt") is not None else (float(payload["mean_pLDDT"]) if payload.get("mean_pLDDT") is not None else None),
        "ptm": float(payload["ptm"]) if payload.get("ptm") is not None else (float(payload["pTM"]) if payload.get("pTM") is not None else None),
        "mean_pae": float(payload["mean_pae"]) if payload.get("mean_pae") is not None else (float(payload["mean_PAE"]) if payload.get("mean_PAE") is not None else None),
        "pdb_path": payload.get("pdb_path"),
        "error": payload.get("error"),
        "backend": payload.get("backend", backend),
    }


@dataclass
class ESMFoldOracle:
    mode: str = "mock"
    model: str = "facebook/esmfold_v1"
    python: str = ".venv_oracle/bin/python"
    script: str = "scripts/19_esmfold_batch_oracle.py"
    device: str = "cuda:0"
    max_length: int = 1500
    chunk_size: int = 64
    cuda_visible_devices: str = "1"
    cache_dir: str | None = None
    cache: OracleCache | None = None
    project_root: Path = Path("/Users/bytedance/python_project")

    def __post_init__(self) -> None:
        if self.cache is None and self.cache_dir:
            self.cache = OracleCache(Path(self.cache_dir) / "esmfold_cache.sqlite")

    def _run_esmfold_subprocess(self, sequences: List[str]) -> List[Dict[str, Any]]:
        # Create temporary input/output files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, dir='/tmp') as inp_f:
            for seq in sequences:
                inp_f.write(json.dumps({"sequence": seq}) + '\n')
            inp_path = inp_f.name

        out_path = inp_path + ".out.jsonl"

        try:
            # Prepare environment variables
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(self.cuda_visible_devices)
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            env["PYTHONUNBUFFERED"] = "1"
            # Preserve existing HF_HOME and TORCH_HOME if they exist
            for env_var in ["HF_HOME", "TORCH_HOME"]:
                if env_var in os.environ:
                    env[env_var] = os.environ[env_var]

            # Build command
            cmd = [
                self.python, "-u",
                str(self.project_root / self.script),
                "--input", inp_path,
                "--output", out_path,
                "--model", self.model,
                "--device", self.device,
                "--max-length", str(self.max_length),
                "--chunk-size", str(self.chunk_size)
            ]

            # Run subprocess
            result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
            
            # Read output
            results = []
            if Path(out_path).exists():
                with open(out_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            results.append(json.loads(line))

            # Ensure we have results for all sequences
            if len(results) != len(sequences):
                missing = len(sequences) - len(results)
                for _ in range(missing):
                    results.append({
                        "sequence": "",
                        "valid": False,
                        "error": f"Subprocess missing result: exit_code={result.returncode}, stderr={result.stderr[-500:]}",
                        "mean_plddt": None,
                        "ptm": None,
                        "mean_pae": None,
                        "pdb_path": None,
                        "backend": "esmfold_v1"
                    })
            return results

        finally:
            # Clean up temporary files
            try:
                os.unlink(inp_path)
                if os.path.exists(out_path):
                    os.unlink(out_path)
            except:
                pass

    def score_one_uncached(self, sequence: str) -> Dict[str, Any]:
        seq = str(sequence or "")
        try:
            if self.mode == "mock":
                payload = _mock_struct_scores(seq)
            else:
                # Run real ESMFold via subprocess
                raw_results = self._run_esmfold_subprocess([seq])
                raw = raw_results[0] if raw_results else {}
                payload = {
                    "mean_plddt": float(raw.get("mean_plddt", 0.0) or 0.0),
                    "ptm": float(raw.get("ptm", math.nan) or 0.0),
                    "mean_pae": float(raw.get("mean_pae", math.nan) or 0.0),
                    "pdb_path": raw.get("pdb_path"),
                    "valid": raw.get("valid", False),
                    "error": raw.get("error"),
                    "backend": "esmfold_v1",
                }
            payload["sequence"] = seq
            return _schema(seq, payload, "mock" if self.mode == "mock" else "esmfold_v1")
        except Exception as exc:
            return {
                "sequence": seq,
                "valid": False,
                "mean_plddt": None,
                "ptm": None,
                "mean_pae": None,
                "pdb_path": None,
                "error": f"{type(exc).__name__}: {exc}",
                "backend": "mock" if self.mode == "mock" else "esmfold_v1",
            }

    def score_one(self, sequence: str) -> Dict[str, Any]:
        if self.cache is None:
            return self.score_one_uncached(sequence)
        cached = self.cache.get(sequence)
        if cached is not None:
            return _schema(sequence, cached, "mock" if self.mode == "mock" else "esmfold_v1")
        payload = self.score_one_uncached(sequence)
        self.cache.set(sequence, payload)
        return payload

    def score_many(self, sequences: Iterable[str]) -> List[Dict[str, Any]]:
        # Check cache first
        results = []
        to_process = []
        to_process_indices = []
        
        for idx, seq in enumerate(sequences):
            if self.cache is not None:
                cached = self.cache.get(seq)
                if cached is not None:
                    results.append(_schema(seq, cached, "mock" if self.mode == "mock" else "esmfold_v1"))
                    continue
            # If not cached or no cache, need to process
            to_process.append(seq)
            to_process_indices.append(idx)
            # Add a placeholder to maintain order
            results.append(None)

        # Process any cache misses in batch
        if to_process and self.mode != "mock":
            batch_results = self._run_esmfold_subprocess(to_process)
            for i, (orig_idx, seq, res) in enumerate(zip(to_process_indices, to_process, batch_results)):
                # Cache the result
                if self.cache is not None:
                    self.cache.set(seq, res)
                # Add to results in correct position
                results[orig_idx] = _schema(seq, res, "esmfold_v1")
        elif to_process:
            # Mock mode, process individually
            for orig_idx, seq in zip(to_process_indices, to_process):
                res = self.score_one_uncached(seq)
                results[orig_idx] = res

        return results