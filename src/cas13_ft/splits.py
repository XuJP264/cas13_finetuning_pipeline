from __future__ import annotations

import random
from typing import Dict, List, Tuple


def make_splits(
    records: List[dict],
    train_frac: float = 0.90,
    valid_frac: float = 0.05,
    test_frac: float = 0.05,
    seed: int = 1337,
) -> Tuple[List[dict], List[dict], List[dict]]:
    total = train_frac + valid_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1.0, got {total}")
    unique: Dict[str, dict] = {}
    for record in records:
        seq = record["sequence"]
        if seq not in unique:
            unique[seq] = record
    shuffled = list(unique.values())
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_frac)
    n_valid = int(n * valid_frac)
    train = shuffled[:n_train]
    valid = shuffled[n_train : n_train + n_valid]
    test = shuffled[n_train + n_valid :]
    assert not ({r["sequence"] for r in train} & {r["sequence"] for r in valid})
    assert not ({r["sequence"] for r in train} & {r["sequence"] for r in test})
    assert not ({r["sequence"] for r in valid} & {r["sequence"] for r in test})
    return train, valid, test
