from __future__ import annotations


def test_rl_cascade_oracle_adapter_exports_cas13_oracle():
    from cas13_ft.oracle import Cas13Oracle as FtCas13Oracle
    from cas13_rl.cascade_oracle import Cas13Oracle as RlCas13Oracle

    assert RlCas13Oracle is FtCas13Oracle

