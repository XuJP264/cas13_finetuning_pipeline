from __future__ import annotations

from cas13_rl.generation import generate_mock_samples, write_samples_jsonl
from cas13_rl.oracle import MockOracle, OracleCache, score_with_cache
from cas13_rl.ppo import run_mock_ppo
from cas13_rl.reward import cas13_motif_score, compute_rewards


AA = "ACDEFGHIKLMNPQRSTVWY"


def test_oracle_cache_reuses_sequence(tmp_path):
    cache = OracleCache(tmp_path / "cache.sqlite")
    oracle = MockOracle(min_len=5, max_len=100)
    seq = "ACDEFRAAAAH" + AA
    first = score_with_cache([seq], oracle, cache)
    second = score_with_cache([seq], oracle, cache)
    assert first == second
    assert cache.get(seq)["sequence"] == seq
    cache.close()


def test_reward_and_motif():
    seq1 = "RAAAAH" + "ACDEFGHIKLMNPQRSTVWY" * 3 + "RCCCCH"
    seq2 = "ACD" * 10
    rows = [
        {"sequence": seq1, "progen3_nll": 2.0, "mean_plddt": 80.0},
        {"sequence": seq2, "progen3_nll": 5.0, "mean_plddt": 40.0},
    ]
    rewards = compute_rewards(rows, {"w_pg3": 0.3, "w_plddt": 0.4, "w_valid": 0.2, "w_motif": 0.1}, 10, 200)
    assert cas13_motif_score(seq1) == 1.0
    assert "reward" in rewards[0]
    assert rewards[0]["validity_score"] == 1.0


def test_generate_sample_format(tmp_path):
    rows = generate_mock_samples(["MKT"], num_samples=3, max_new_tokens=5, seed=1)
    out = tmp_path / "samples.jsonl"
    write_samples_jsonl(rows, out)
    assert len(rows) == 3
    assert {"id", "prompt", "sequence"} <= rows[0].keys()
    assert out.read_text(encoding="utf-8").count("\n") == 3


def test_mock_ppo_smoke(tmp_path):
    cfg = {
        "seed": 1,
        "paths": {"output_dir": str(tmp_path / "rl"), "oracle_cache": str(tmp_path / "rl" / "cache.sqlite")},
        "generation": {"prompt_length": 8, "max_new_tokens": 16},
        "oracle": {"min_len": 5, "max_len": 100, "mode": "mock"},
        "reward": {"w_pg3": 0.3, "w_plddt": 0.4, "w_valid": 0.2, "w_motif": 0.1, "w_kl": 0.0},
        "ppo": {"steps": 2, "batch_size": 2},
    }
    logs = run_mock_ppo(cfg)
    assert len(logs) == 2
    assert {"policy_loss", "value_loss", "reward_mean", "kl_mean", "entropy", "mean_plddt", "validity_rate"} <= logs[0].keys()
