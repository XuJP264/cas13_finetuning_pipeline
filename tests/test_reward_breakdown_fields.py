from __future__ import annotations

import json

from cas13_rl.reward import compute_cas13_rewards


REQUIRED = {
    "sequence_sha256",
    "sequence_length",
    "valid_basic",
    "raw_lm_mean_logprob",
    "raw_lm_perplexity",
    "raw_mean_plddt",
    "raw_ptm",
    "raw_mean_pae",
    "raw_hepn_count",
    "score_lm",
    "score_plddt",
    "score_ptm",
    "score_pae",
    "score_hepn",
    "score_length",
    "score_diversity",
    "r_prop",
    "reward_for_rl",
    "invalid_reason",
    "oracle_errors",
}


def test_reward_breakdown_jsonl_contains_required_fields(tmp_path):
    seq = "RAAAAH" + "A" * 54 + "RCCCCH"
    log_path = tmp_path / "reward_breakdown.jsonl"
    compute_cas13_rewards(
        [seq],
        [{"sequence": seq, "valid": True, "mean_plddt": 80.0, "ptm": 0.7, "mean_pae": 8.0, "pdb_path": None, "error": None, "backend": "mock"}],
        [{"sequence": seq, "valid": True, "mean_logprob": -1.0, "perplexity": 2.7, "error": None, "backend": "mock"}],
        {"min_len": 10, "max_len": 200},
        log_path=log_path,
    )
    row = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert REQUIRED <= row.keys()


def test_reward_reads_progen3_mean_logprob_and_perplexity_fields():
    seq = "RAAAAH" + "A" * 54 + "RCCCCH"
    rows = compute_cas13_rewards(
        [seq],
        [{"sequence": seq, "valid": True, "mean_plddt": 80.0, "ptm": 0.7, "mean_pae": 8.0, "pdb_path": None, "error": None, "backend": "mock"}],
        [{"sequence": seq, "valid": True, "mean_logprob": -1.25, "perplexity": 3.49, "error": None, "backend": "real"}],
        {"min_len": 10, "max_len": 200},
    )
    assert rows[0]["raw_lm_mean_logprob"] == -1.25
    assert rows[0]["raw_lm_perplexity"] == 3.49
    assert rows[0]["score_lm"] > 0.0
