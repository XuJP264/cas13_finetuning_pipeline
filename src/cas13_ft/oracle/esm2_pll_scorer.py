from __future__ import annotations


class ESM2PseudoPLLScorer:
    def __init__(self, mode: str = "disabled", model_name: str = "facebook/esm2_t33_650M_UR50D", **_: object):
        self.mode = mode
        self.model_name = model_name

    def score_sequences(self, sequences: list[str], batch_size: int = 1) -> list[float | None]:
        if self.mode == "disabled":
            return [None for _ in sequences]
        if self.mode == "mock":
            out = []
            for seq in sequences:
                text = str(seq or "")
                aromatic = sum(1 for ch in text if ch in "FWY") / max(1, len(text))
                out.append(float(-2.0 + aromatic - abs(len(text) - 1000) / 5000.0))
            return out
        if self.mode == "single_mask_exact":
            raise NotImplementedError(
                "ESM2PseudoPLLScorer single_mask_exact is intentionally not implemented: "
                "逐位 mask 对 Cas13 长序列成本很高；建议后续使用 OFS 近似或离线计算。"
            )
        raise ValueError(f"ESM2PseudoPLLScorer unsupported mode={self.mode!r}")

