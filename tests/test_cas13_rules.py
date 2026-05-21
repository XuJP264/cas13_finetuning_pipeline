from __future__ import annotations

from cas13_rl.cas13_rules import hepn_motif_score, length_distribution_score, validate_cas13_sequence


def test_hepn_rxxxxh_motif_score():
    assert hepn_motif_score("RAAAAHACDRCCCCH") == 1.0
    assert hepn_motif_score("RAAAAHACD") == 0.5
    assert hepn_motif_score("ACDEFG") == 0.0


def test_cas13_sequence_rules_report_error():
    result = validate_cas13_sequence("ACDX", min_len=5, max_len=20)
    assert result.valid is False
    assert "non-canonical" in (result.error or "")
    assert "length" in (result.error or "")


def test_length_distribution_score_is_bounded():
    assert length_distribution_score("A" * 900, target_len=900, tolerance=700) == 1.0
    assert 0.0 <= length_distribution_score("A" * 200, target_len=900, tolerance=700) <= 1.0

