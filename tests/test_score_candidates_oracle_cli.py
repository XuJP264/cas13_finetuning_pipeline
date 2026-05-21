from __future__ import annotations
from pathlib import Path

import json
import subprocess
import sys

import yaml

AA = "ACDEFGHIKLMNPQRSTVWY"


def cas13_like() -> str:
    core = (AA * 50)[:888]
    return core[:300] + "RAAAAH" + core[300:600] + "RCCCCH" + core[600:]


def test_score_candidates_oracle_cli_outputs_jsonl_and_summary(tmp_path):
    fasta = tmp_path / "candidates.fasta"
    output = tmp_path / "scores.jsonl"
    config = tmp_path / "oracle.yaml"
    fasta.write_text(f">good\n{cas13_like()}\n>bad\nACDX\n", encoding="utf-8")
    cfg = {
        "hard_filters": {
            "valid_amino_acids": True,
            "length_range": [800, 1400],
            "require_two_hepn_rx4h": True,
            "low_complexity_filter": True,
            "low_complexity": {"max_aa_fraction": 0.20, "max_run": 8},
        },
        "naturalness": {
            "enabled": True,
            "scorer": "mock",
            "batch_size": 2,
            "normalize": {"mean": -2.2, "std": 0.5, "clip": [-3, 3]},
            "use_cache": True,
        },
        "cas13_identity": {"enabled": True, "mode": "mock", "normalize": {"mean": 0.5, "std": 0.25, "clip": [-3, 3]}},
        "structure": {"enabled": True, "mode": "mock", "normalize": {"ptm_mean": 0.5, "ptm_std": 0.15, "plddt_mean": 70.0, "plddt_std": 10.0, "clip": [-3, 3]}, "use_cache": True},
        "diversity": {"enabled": True, "mode": "simple_hamming"},
        "reward": {"hard_fail_reward": -10.0, "clip_final_reward": [-5, 5]},
        "cache": {"path": str(tmp_path / "oracle_cache.sqlite")},
    }
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/08_score_candidates_oracle.py",
            "--input",
            str(fasta),
            "--output",
            str(output),
            "--config",
            str(config),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "hard_filter_pass" in result.stdout
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["sequence_id"] == "good"
    assert rows[0]["passed_hard_filters"] is True
    assert rows[1]["passed_hard_filters"] is False

