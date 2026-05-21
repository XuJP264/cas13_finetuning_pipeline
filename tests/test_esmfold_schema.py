from __future__ import annotations

from cas13_rl.oracle_esmfold import ESMFoldOracle


def test_esmfold_schema_is_fixed():
    row = ESMFoldOracle(mode="mock", device="cpu").score_one("ACDEFGHIK")
    assert set(row) == {"sequence", "valid", "mean_plddt", "ptm", "mean_pae", "pdb_path", "error", "backend"}
    assert row["backend"] == "mock"
    assert isinstance(row["mean_plddt"], float)

