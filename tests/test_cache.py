from __future__ import annotations

from cas13_rl.cache import OracleCache, sequence_sha256


def test_cache_uses_sequence_sha256(tmp_path):
    cache = OracleCache(tmp_path / "oracle.sqlite")
    seq = "ACDEFGHIK"
    payload = {"sequence": seq, "value": 1}
    cache.set(seq, payload)
    assert sequence_sha256(seq) == "c729ebc224388368ab8c8df88487ef137ad8bd5097651cf67c37bda5622c9f9a"
    assert cache.get(seq) == payload
    cache.close()
