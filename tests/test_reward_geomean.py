from __future__ import annotations

import math

from cas13_rl.reward import compute_cas13_rewards


def test_reward_uses_weighted_geometric_mean():
    seq = "RAAAAH" + "A" * 54 + "RCCCCH"
    rows = compute_cas13_rewards(
        [seq],
        [{"sequence": seq, "valid": True, "mean_plddt": 70.0, "ptm": 0.5, "mean_pae": 15.0, "pdb_path": None, "error": None, "backend": "mock"}],
        [{"sequence": seq, "valid": True, "mean_logprob": -2.75, "perplexity": 10.0, "error": None, "backend": "mock"}],
        {
            "min_len": 10,
            "max_len": 200,
            "length_center": len(seq),
            "weights": {"lm": 1.0, "plddt": 1.0, "ptm": 1.0, "pae": 0.0, "hepn": 0.0, "length": 0.0, "diversity": 0.0},
        },
    )
    expected = math.exp((math.log(0.5) + math.log(0.5) + math.log(0.5)) / 3.0)
    assert abs(rows[0]["r_prop"] - expected) < 1e-9
    assert abs(rows[0]["reward_for_rl"] - (2 * expected - 1)) < 1e-9

