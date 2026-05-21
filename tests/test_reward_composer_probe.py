from __future__ import annotations
from pathlib import Path

import json
import subprocess
import sys


def test_reward_composer_probe_runs(tmp_path):
    output = tmp_path / "reward_probe.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/17_probe_reward_composer.py",
            "--config",
            "configs/rl_cas13_debug_mac.yaml",
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    assert {"score_length", "score_hepn", "score_lm", "score_plddt", "score_ptm", "r_prop", "reward_for_rl", "invalid_reason"} <= rows[0].keys()
    assert rows[-1]["invalid_reason"] is not None

