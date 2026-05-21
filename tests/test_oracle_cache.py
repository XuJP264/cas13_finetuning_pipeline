from __future__ import annotations

from cas13_ft.oracle.cache import OracleSQLiteCache, make_cache_key


def test_oracle_cache_hit_miss_and_overwrite(tmp_path):
    cache = OracleSQLiteCache(tmp_path / "oracle.sqlite")
    seq = "ACDEFGHIK"
    assert cache.get_cached(seq, "mock", "v1", "a") is None
    cache.set_cached(seq, "mock", "v1", "a", {"score": 1})
    assert cache.get_cached(seq, "mock", "v1", "a") == {"score": 1}
    assert cache.get_cached(seq, "mock", "v1", "b") is None
    cache.set_cached(seq, "mock", "v1", "a", {"score": 2})
    assert cache.get_cached(seq, "mock", "v1", "a") == {"score": 2}
    assert make_cache_key(seq, "mock", "v1", "a") != make_cache_key(seq, "mock", "v1", "b")
    cache.close()

