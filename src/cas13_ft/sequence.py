from __future__ import annotations

import re
from hashlib import sha256

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
AA_RE = re.compile(r"[^ACDEFGHIKLMNPQRSTVWY]")


def clean_protein_sequence(sequence: str) -> str:
    """Uppercase and remove any non-canonical amino acid characters."""
    if sequence is None:
        return ""
    return AA_RE.sub("", str(sequence).upper())


def is_valid_length(sequence: str, min_len: int = 200, max_len: int = 1500) -> bool:
    return min_len <= len(sequence) <= max_len


def sequence_hash(sequence: str) -> str:
    return sha256(sequence.encode("utf-8")).hexdigest()


def validity_score(sequence: str, min_len: int = 200, max_len: int = 1500) -> float:
    raw = str(sequence or "").upper()
    if any(ch not in VALID_AA for ch in raw):
        return 0.0
    length = len(raw)
    if min_len <= length <= max_len:
        return 1.0
    if length == 0:
        return 0.0
    if length < min_len:
        return max(0.0, length / float(min_len))
    return max(0.0, 1.0 - (length - max_len) / float(max_len))
