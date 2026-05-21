from __future__ import annotations

from cas13_rl.oracle_progen3 import ProGen3Oracle


def test_progen3_schema_is_fixed():
    row = ProGen3Oracle(mode="mock", device="cpu").score_one("ACDEFGHIK")
    assert set(row) == {"sequence", "valid", "mean_logprob", "perplexity", "error", "backend"}
    assert row["backend"] == "mock"
    assert isinstance(row["mean_logprob"], float)


def test_progen3_real_backend_requires_model_configuration():
    try:
        ProGen3Oracle(mode="real", device="cpu")
    except ValueError as exc:
        assert "model_name_or_path" in str(exc)
    else:
        raise AssertionError("expected clear missing real backend configuration error")
