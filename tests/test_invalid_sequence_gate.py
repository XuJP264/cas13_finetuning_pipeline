from __future__ import annotations

from cas13_rl.rl_trainer import Cas13RLTrainer
from cas13_rl.reward import compute_cas13_rewards


def test_reward_invalid_sequence_gets_fixed_invalid_reward():
    rows = compute_cas13_rewards(
        ["ACDX"],
        [{"sequence": "ACDX", "valid": False, "mean_plddt": None, "ptm": None, "mean_pae": None, "pdb_path": None, "error": "not called", "backend": "mock"}],
        [{"sequence": "ACDX", "valid": False, "mean_logprob": None, "perplexity": None, "error": "not called", "backend": "mock"}],
        {"min_len": 1, "max_len": 10, "invalid_reward": -3.0},
    )
    assert rows[0]["valid_basic"] is False
    assert rows[0]["reward_for_rl"] == -3.0
    assert "non-canonical" in rows[0]["invalid_reason"]


def test_trainer_basic_gate_skips_oracles_for_invalid_sequence(tmp_path):
    cfg = {
        "runtime": {"mode": "mac"},
        "paths": {"output_dir": str(tmp_path), "esmfold_cache": str(tmp_path / "esm.sqlite"), "progen3_cache": str(tmp_path / "pg.sqlite")},
        "oracle": {"esmfold": {"mode": "mock", "device": "cpu"}, "progen3": {"mode": "mock", "device": "cpu"}},
        "reward": {"min_len": 10, "max_len": 20},
    }
    trainer = Cas13RLTrainer(cfg)
    try:
        esm, lm = trainer._score_oracles_with_basic_gate(["ACDX"])
    finally:
        trainer.close()
    assert esm[0]["valid"] is False
    assert lm[0]["valid"] is False
    assert esm[0]["mean_plddt"] is None
    assert lm[0]["mean_logprob"] is None

