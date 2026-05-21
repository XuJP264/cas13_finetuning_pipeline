from __future__ import annotations

from collections import Counter
from pathlib import Path

from .filters import count_hepn_rx4h, low_complexity_score, validate_amino_acid_sequence

AA = "ACDEFGHIKLMNPQRSTVWY"


def handcrafted_features(sequence: str) -> list[float]:
    seq = str(sequence or "").upper()
    valid, _ = validate_amino_acid_sequence(seq)
    hepn = len(count_hepn_rx4h(seq)) if valid else 0
    counts = Counter(seq)
    lc = low_complexity_score(seq)
    return [
        float(len(seq)),
        float(hepn),
        *[counts.get(aa, 0) / max(1, len(seq)) for aa in AA],
        float(lc["max_aa_fraction"]),
        float(lc["max_run"]),
    ]


class Cas13IdentityScorer:
    def __init__(self, mode: str = "mock", model_path: str | None = None, length_range: list[int] | None = None, **_: object):
        self.mode = mode
        self.model_path = model_path
        self.length_range = length_range or [800, 1400]
        self.model = None
        if self.mode == "sklearn_joblib":
            if not model_path:
                raise ValueError("Cas13IdentityScorer sklearn_joblib requires cas13_identity.model_path")
            try:
                import joblib  # type: ignore
            except Exception as exc:
                raise RuntimeError(f"Cas13IdentityScorer requires joblib/sklearn dependencies: {exc}") from exc
            try:
                self.model = joblib.load(Path(model_path))
            except Exception as exc:
                raise RuntimeError(f"Cas13IdentityScorer failed loading model_path={model_path}: {exc}") from exc

    def score_sequences(self, sequences: list[str]) -> list[float | None]:
        if self.mode == "disabled":
            return [None for _ in sequences]
        if self.mode == "mock":
            scores = []
            lo, hi = self.length_range
            for seq in sequences:
                text = str(seq or "").upper()
                valid, _ = validate_amino_acid_sequence(text)
                hepn = len(count_hepn_rx4h(text)) if valid else 0
                in_len = lo <= len(text) <= hi
                if hepn == 2 and in_len:
                    score = 0.9
                elif hepn == 1:
                    score = 0.45
                elif hepn == 2:
                    score = 0.65
                else:
                    score = 0.1
                scores.append(float(score))
            return scores
        if self.mode == "sklearn_joblib":
            features = [handcrafted_features(seq) for seq in sequences]
            if hasattr(self.model, "predict_proba"):
                return [float(x) for x in self.model.predict_proba(features)[:, 1]]
            if hasattr(self.model, "decision_function"):
                return [float(x) for x in self.model.decision_function(features)]
            raise RuntimeError("Cas13IdentityScorer sklearn_joblib model needs predict_proba or decision_function")
        raise ValueError(f"Cas13IdentityScorer unsupported mode={self.mode!r}")

