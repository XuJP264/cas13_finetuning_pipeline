from __future__ import annotations

from cas13_rl.oracle_esmfold import ESMFoldOracle
from cas13_rl.oracle_progen3 import ProGen3Oracle


def test_esmfold_mock_schema():
    row = ESMFoldOracle(mode="mock", device="cpu").score_one("ACDEFGHIK")
    assert set(row) == {"sequence", "valid", "mean_plddt", "ptm", "mean_pae", "pdb_path", "error", "backend"}
    assert row["valid"] is True


def test_progen3_mock_schema():
    row = ProGen3Oracle(mode="mock", device="cpu").score_one("ACDEFGHIK")
    assert set(row) == {"sequence", "valid", "mean_logprob", "perplexity", "error", "backend"}
    assert row["valid"] is True
