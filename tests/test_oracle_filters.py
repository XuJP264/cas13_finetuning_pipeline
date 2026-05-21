from __future__ import annotations

from cas13_ft.oracle.filters import apply_hard_filters, count_hepn_rx4h, low_complexity_score, validate_amino_acid_sequence

AA = "ACDEFGHIKLMNPQRSTVWY"


def cas13_like(length: int = 900) -> str:
    core = (AA * ((length // len(AA)) + 2))[: length - 12]
    return core[:300] + "RAAAAH" + core[300:600] + "RCCCCH" + core[600:]


def test_hepn_detection_and_invalid_alphabet():
    assert count_hepn_rx4h("RAAAAH") == [0]
    assert count_hepn_rx4h("RAAAAHRCCCCH") == [0, 6]
    valid, reasons = validate_amino_acid_sequence("ACDX")
    assert valid is False
    assert any("invalid_alphabet" in reason for reason in reasons)


def test_hard_filters_failures_and_pass():
    cfg = {"length_range": [800, 1400], "require_two_hepn_rx4h": True, "low_complexity_filter": True}
    assert apply_hard_filters("RAAAAHRCCCCH", cfg).passed is False
    assert apply_hard_filters((AA * 50)[:900], cfg).passed is False
    assert low_complexity_score("A" * 900)["is_low_complexity"] is True
    passed = apply_hard_filters(cas13_like(), cfg)
    assert passed.passed is True
    assert len(passed.hepn_positions) == 2

