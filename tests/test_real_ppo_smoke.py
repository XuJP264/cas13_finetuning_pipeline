from __future__ import annotations

import json

from cas13_rl.rl_trainer import Cas13RLTrainer


def _config(tmp_path):
    train_data = tmp_path / "train.jsonl"
    train_data.write_text(json.dumps({"sequence": "ACDEFGHIKLMNPQRSTVWY"}) + "\n", encoding="utf-8")
    return {
        "seed": 11,
        "runtime": {"mode": "mac"},
        "paths": {
            "train_data": str(train_data),
            "policy_model": str(tmp_path / "missing_policy_model"),
            "output_dir": str(tmp_path / "rl"),
            "esmfold_cache": str(tmp_path / "rl" / "esm.sqlite"),
            "progen3_cache": str(tmp_path / "rl" / "pg.sqlite"),
        },
        "generation": {"prompt_length": 4, "max_new_tokens": 4, "do_sample": True, "temperature": 1.0, "top_p": 1.0},
        "training": {
            "mode": "real_ppo",
            "device": "cpu",
            "steps": 2,
            "batch_size": 1,
            "learning_rate": 1e-5,
            "ppo_clip_range": 0.2,
            "kl_weight": 0.01,
            "save_steps": 1,
            "tokenizer_vocab_size": 32,
        },
        "oracle": {
            "esmfold": {"mode": "mock", "device": "cpu"},
            "progen3": {"mode": "mock", "device": "cpu"},
        },
        "reward": {
            "min_len": 1,
            "max_len": 200,
            "length_center": 20,
            "length_tolerance": 100,
            "kl_weight": 0.01,
            "weights": {"lm": 1.0, "plddt": 1.0, "ptm": 1.0, "hepn": 0.0, "length": 0.1, "diversity": 0.0},
        },
    }


def test_real_ppo_tiny_mock_oracle_checkpoint_and_resume(tmp_path):
    trainer = Cas13RLTrainer(_config(tmp_path))
    try:
        first = trainer.run(max_steps=1)
    finally:
        trainer.close()
    assert first[0]["mode"] == "real_ppo"
    assert "kl_mean" in first[0]
    checkpoint = tmp_path / "rl" / "checkpoint_step_1"
    assert checkpoint.exists()
    assert (tmp_path / "rl" / "reward_breakdown.jsonl").exists()

    trainer = Cas13RLTrainer(_config(tmp_path))
    try:
        resumed = trainer.run(resume=True, max_steps=2)
    finally:
        trainer.close()
    assert resumed[0]["step"] == 2
    assert resumed[0]["resume_from_step"] == 1
    assert (tmp_path / "rl" / "checkpoint_step_2").exists()


def test_mock_rl_smoke_generates_sequence_past_hard_filter(tmp_path):
    cfg = _config(tmp_path)
    cfg["training"] = {"mode": "mock_debug", "steps": 1, "batch_size": 1}
    cfg["generation"] = {
        "prompt_length": 8,
        "min_new_tokens": 850,
        "max_new_tokens": 900,
        "target_min_len": 850,
        "target_max_len": 1500,
    }
    cfg["reward"]["min_len"] = 850
    cfg["reward"]["max_len"] = 1500

    trainer = Cas13RLTrainer(cfg)
    try:
        trainer.run(max_steps=1)
    finally:
        trainer.close()

    rows = [
        json.loads(line)
        for line in (tmp_path / "rl" / "reward_breakdown.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["sequence_length"] >= cfg["reward"]["min_len"]
    assert rows[0]["sequence_length"] <= cfg["reward"]["max_len"]
    assert rows[0]["invalid_reason"] is None
