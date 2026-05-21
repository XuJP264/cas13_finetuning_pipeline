from __future__ import annotations

import json

from cas13_rl.reward import compute_cas13_rewards


def test_reward_outputs_all_components_to_jsonl(tmp_path):
    sequences = ["RAAAAH" + "ACDEFGHIKLMNPQRSTVWY" * 4 + "RCCCCH", "ACDEFGHIKLMNPQRSTVWY" * 4]
    esm = [
        {"sequence": sequences[0], "mean_pLDDT": 80.0, "pTM": 0.7, "mean_PAE": 5.0, "valid": True, "error": None},
        {"sequence": sequences[1], "mean_pLDDT": 60.0, "pTM": 0.4, "mean_PAE": 12.0, "valid": True, "error": None},
    ]
    lm = [
        {"sequence": sequences[0], "mean_logprob": -1.2, "perplexity": 3.3, "valid": True, "error": None},
        {"sequence": sequences[1], "mean_logprob": -2.1, "perplexity": 8.1, "valid": True, "error": None},
    ]
    log_path = tmp_path / "reward.jsonl"
    rows = compute_cas13_rewards(
        sequences,
        esm,
        lm,
        {"weights": {"w_struct": 1, "w_lm": 1, "w_motif": 1, "w_len": 1, "w_div": 1, "w_kl": 1}},
        log_path=log_path,
        kl_values=[0.1, 0.2],
    )
    assert {"R_struct", "R_lm", "R_motif", "R_len", "R_div", "KL"} <= rows[0].keys()
    written = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert written[0]["reward_components"]["struct"] == rows[0]["R_struct"]
    assert written[0]["KL"] == 0.1


def test_reward_marks_single_failed_oracle_row_invalid():
    rows = compute_cas13_rewards(
        ["ACDEFG"],
        [{"valid": False, "error": "fold failed", "mean_pLDDT": 0.0, "pTM": 0.0, "mean_PAE": 0.0}],
        [{"valid": True, "error": None, "mean_logprob": -1.0, "perplexity": 2.7}],
        {"weights": {}},
    )
    assert rows[0]["valid"] is False
    assert "fold failed" in rows[0]["error"]

