from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import torch

from .progen3 import resolve_device


def mean_plddt_from_pdb(pdb: str) -> tuple[float, list[float]]:
    values = []
    for line in pdb.splitlines():
        if line.startswith("ATOM") and len(line) >= 66:
            try:
                values.append(float(line[60:66].strip()))
            except ValueError:
                continue
    if not values:
        return 0.0, []
    return float(sum(values) / len(values)), values


@dataclass
class ESMFoldOracle:
    enabled: bool = False
    device: str = "auto"
    max_length: int = 800
    output_dir: str | None = None

    def __post_init__(self) -> None:
        self.device = resolve_device(self.device)
        self.output_path = Path(self.output_dir) if self.output_dir else None
        if self.output_path:
            self.output_path.mkdir(parents=True, exist_ok=True)
        if not self.enabled:
            self.model = None
            return
        try:
            import esm  # type: ignore

            self.model = esm.pretrained.esmfold_v1()
            self.model = self.model.eval().to(self.device)
            if hasattr(self.model, "set_chunk_size"):
                self.model.set_chunk_size(64)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load real ESMFold oracle via fair-esm. Install fair-esm with ESMFold extras and its "
                f"structure dependencies, then retry. Reason: {type(exc).__name__}: {exc}"
            ) from exc

    def score_one(self, sequence: str, seq_id: str | None = None) -> Dict[str, object]:
        text = str(sequence)
        if not self.enabled:
            return {"sequence": text, "mean_plddt": 0.0, "plddt_summary": {"skipped": True, "reason": "disabled"}}
        if len(text) > self.max_length:
            return {
                "sequence": text,
                "mean_plddt": 0.0,
                "plddt_summary": {"skipped": True, "reason": f"length {len(text)} > max_length {self.max_length}"},
            }
        with torch.no_grad():
            pdb = self.model.infer_pdb(text)
        mean_plddt, values = mean_plddt_from_pdb(pdb)
        pdb_path = None
        if self.output_path:
            name = seq_id or f"esmfold_{abs(hash(text))}"
            pdb_path = self.output_path / f"{name}.pdb"
            pdb_path.write_text(pdb, encoding="utf-8")
        return {
            "sequence": text,
            "mean_plddt": mean_plddt,
            "plddt_summary": {"n_atoms": len(values), "min": min(values) if values else 0.0, "max": max(values) if values else 0.0},
            "pdb_path": str(pdb_path) if pdb_path else None,
        }

    def score_many(self, sequences: Iterable[str]) -> List[Dict[str, object]]:
        return [self.score_one(seq) for seq in sequences]
