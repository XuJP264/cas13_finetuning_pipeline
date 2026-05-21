from __future__ import annotations

from cas13_rl.rl_trainer import Cas13RLTrainer


def _config(tmp_path):
    return {
        "seed": 7,
        "runtime": {"mode": "mac"},
        "paths": {
            "output_dir": str(tmp_path / "rl"),
            "esmfold_cache": str(tmp_path / "rl" / "esm.sqlite"),
            "progen3_cache": str(tmp_path / "rl" / "pg.sqlite"),
        },
        "generation": {"prompt_length": 8, "max_new_tokens": 8},
        "training": {"steps": 2, "batch_size": 2},
        "oracle": {"esmfold": {"mode": "mock", "device": "cpu"}, "progen3": {"mode": "mock", "device": "cpu"}},
        "reward": {"weights": {"w_struct": 0.1, "w_lm": 0.1, "w_motif": 0.1, "w_len": 0.1, "w_div": 0.1, "w_kl": 0.0}},
    }


def test_checkpoint_resume_continues_from_saved_step(tmp_path):
    trainer = Cas13RLTrainer(_config(tmp_path))
    try:
        first = trainer.run(max_steps=1)
    finally:
        trainer.close()
    assert first[0]["step"] == 1

    trainer = Cas13RLTrainer(_config(tmp_path))
    try:
        resumed = trainer.run(resume=True, max_steps=2)
    finally:
        trainer.close()
    assert resumed[0]["step"] == 2
    assert resumed[0]["resume_from_step"] == 1

