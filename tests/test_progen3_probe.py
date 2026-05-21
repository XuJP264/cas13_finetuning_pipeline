from __future__ import annotations
from pathlib import Path

import json
import subprocess
import sys

import yaml


def test_probe_script_handles_failed_sample_without_stopping(tmp_path):
    cfg = {
        "paths": {"progen3_cache": str(tmp_path / "pg.sqlite")},
        "oracle": {"progen3": {"mode": "mock", "device": "cpu"}},
    }
    config_path = tmp_path / "config.yaml"
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "out.jsonl"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    input_path.write_text('{"sequence": "ACDEFGHIK"}\n{"sequence": ""}\n{"sequence": "LMNPQRSTV"}\n', encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/12_probe_progen3_likelihood.py",
            "--config",
            str(config_path),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "wrote 3" in result.stdout
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["valid"] is True
    assert rows[1]["valid"] is False
    assert rows[2]["mean_logprob"] is not None


def test_probe_script_reads_fasta(tmp_path):
    cfg = {
        "paths": {"progen3_cache": str(tmp_path / "pg.sqlite")},
        "oracle": {"progen3": {"mode": "mock", "device": "cpu"}},
    }
    config_path = tmp_path / "config.yaml"
    input_path = tmp_path / "input.fasta"
    output_path = tmp_path / "out.jsonl"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    input_path.write_text(">a\nACDEFGHIK\n>b\nLMNPQRSTV\n", encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            "scripts/12_probe_progen3_likelihood.py",
            "--config",
            str(config_path),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--limit",
            "1",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )
    assert len(output_path.read_text(encoding="utf-8").splitlines()) == 1

