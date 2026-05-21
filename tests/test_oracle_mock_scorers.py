from __future__ import annotations

from cas13_ft.oracle.cas13_identity import Cas13IdentityScorer
from cas13_ft.oracle.esm2_pll_scorer import ESM2PseudoPLLScorer
from cas13_ft.oracle.esmfold_scorer import ESMFoldScorer

AA = "ACDEFGHIKLMNPQRSTVWY"


def cas13_like() -> str:
    core = (AA * 50)[:888]
    return core[:300] + "RAAAAH" + core[300:600] + "RCCCCH" + core[600:]


def test_mock_cas13_identity_is_deterministic():
    seq = cas13_like()
    scorer = Cas13IdentityScorer(mode="mock")
    assert scorer.score_sequences([seq]) == scorer.score_sequences([seq])
    assert scorer.score_sequences([seq])[0] > scorer.score_sequences([(AA * 45)[:900]])[0]


def test_mock_esmfold_is_deterministic_and_disabled_is_safe():
    records = [{"id": "x", "sequence": cas13_like()}]
    scorer = ESMFoldScorer(mode="mock")
    assert scorer.score_sequences(records) == scorer.score_sequences(records)
    assert scorer.score_sequences(records)[0]["plddt_mean"] is not None
    assert ESMFoldScorer(mode="disabled").score_sequences(records) == [None]


def test_esm2_mock_and_disabled_modes():
    seq = cas13_like()
    assert ESM2PseudoPLLScorer(mode="disabled").score_sequences([seq]) == [None]
    assert ESM2PseudoPLLScorer(mode="mock").score_sequences([seq]) == ESM2PseudoPLLScorer(mode="mock").score_sequences([seq])

