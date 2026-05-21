from __future__ import annotations

from cas13_ft.oracle.filters import apply_hard_filters
from cas13_ft.oracle.reward import combine_oracle_scores, z_normalize

AA = "ACDEFGHIKLMNPQRSTVWY"


def cas13_like() -> str:
    core = (AA * 50)[:888]
    return core[:300] + "RAAAAH" + core[300:600] + "RCCCCH" + core[600:]


def test_z_normalize_and_clipping():
    assert z_normalize(2.0, 1.0, 0.5) == 2.0
    assert z_normalize(10.0, 0.0, 1.0, clip=[-3, 3]) == 3.0
    assert z_normalize(None, 0.0, 1.0) == 0.0


def test_hard_fail_gets_hard_fail_reward():
    hard = apply_hard_filters("ACDX", {"length_range": [800, 1400]})
    combined = combine_oracle_scores(
        hard_filter=hard,
        naturalness_score=1.0,
        cas13_identity_score=1.0,
        plddt_mean=90.0,
        ptm=0.8,
        reward_config={"hard_fail_reward": -10.0},
    )
    assert combined["final_reward"] == -10.0


def test_reward_clips_and_missing_structure_is_ok():
    hard = apply_hard_filters(cas13_like(), {"length_range": [800, 1400], "require_two_hepn_rx4h": True})
    combined = combine_oracle_scores(
        hard_filter=hard,
        naturalness_score=100.0,
        cas13_identity_score=1.0,
        plddt_mean=None,
        ptm=None,
        diversity_score=1.0,
        reward_config={"clip_final_reward": [-1, 1]},
        naturalness_config={"mean": 0.0, "std": 1.0, "clip": [-3, 3]},
        cas13_identity_config={"mean": 0.0, "std": 0.1, "clip": [-3, 3]},
        structure_config={},
    )
    assert combined["metadata"]["structure_missing"] is True
    assert -1.0 <= combined["final_reward"] <= 1.0

