from __future__ import annotations

import pytest
import os

from cas13_rl.oracle import build_oracle


def test_mock_oracle_builder_still_works():
    oracle = build_oracle({"oracle": {"mode": "mock", "min_len": 5, "max_len": 100}})
    row = oracle.score_one("ACDEFGHIK")
    assert row["sequence"] == "ACDEFGHIK"
    assert "progen3_nll" in row


def test_real_progen3_smoke_when_available():
    if os.environ.get("RUN_REAL_ORACLE_TESTS") != "1":
        pytest.skip("set RUN_REAL_ORACLE_TESTS=1 to run real ProGen3 smoke")
    transformers = pytest.importorskip("transformers")
    cfg = {
        "oracle": {
            "mode": "real_progen3",
            "progen3": {
                "model_name": "Profluent-Bio/progen3-219m",
                "device": "cpu",
                "max_length": 32,
                "dtype": "float32",
            },
            "esmfold": {"enabled": False},
        }
    }
    try:
        oracle = build_oracle(cfg)
    except Exception as exc:
        pytest.skip(f"real ProGen3 unavailable in this environment: {exc}")
    row = oracle.score_one("MKTAYIAKQRQISFVKSHFSRQ")
    assert row["progen3_nll"] >= 0
    assert row["progen3_perplexity"] >= 1
