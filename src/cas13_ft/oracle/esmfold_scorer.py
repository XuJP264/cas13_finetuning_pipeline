from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def mean_plddt_from_pdb(path: str | Path) -> float | None:
    values = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("ATOM") and len(line) >= 66:
                try:
                    values.append(float(line[60:66].strip()))
                except ValueError:
                    pass
    return float(sum(values) / len(values)) if values else None


class ESMFoldScorer:
    def __init__(self, mode: str = "disabled", esmfold_command: str | None = None, top_k: int = 32, cache: object | None = None, **_: Any):
        self.mode = mode
        self.esmfold_command = esmfold_command
        self.top_k = int(top_k)
        self.cache = cache

    def score_sequences(self, records: list[dict[str, Any]]) -> list[dict[str, Any] | None]:
        if self.mode == "disabled":
            return [None for _ in records]
        if self.mode == "mock":
            rows = []
            for record in records:
                seq = str(record.get("sequence", ""))
                hepn_bonus = 0.08 if seq.count("R") >= 2 and seq.count("H") >= 2 else 0.0
                length_score = max(0.0, 1.0 - abs(len(seq) - 1000) / 1000.0)
                rows.append(
                    {
                        "ptm": float(min(0.9, 0.35 + 0.45 * length_score + hepn_bonus)),
                        "plddt_mean": float(min(95.0, 55.0 + 30.0 * length_score + hepn_bonus * 50)),
                        "tm_to_reference": None,
                        "pdb_path": None,
                        "metadata": {"backend": "mock"},
                    }
                )
            return rows
        if self.mode == "local_esmfold_cli":
            if not self.esmfold_command:
                raise ValueError("ESMFoldScorer local_esmfold_cli requires structure.esmfold_command")
            return self._score_with_cli(records[: self.top_k])
        raise ValueError(f"ESMFoldScorer unsupported mode={self.mode!r}")

    def _score_with_cli(self, records: list[dict[str, Any]]) -> list[dict[str, Any] | None]:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fasta = tmp_path / "input.fasta"
            out_dir = tmp_path / "esmfold_out"
            out_dir.mkdir()
            with fasta.open("w", encoding="utf-8") as handle:
                for i, record in enumerate(records):
                    seq_id = record.get("id") or f"seq_{i}"
                    handle.write(f">{seq_id}\n{record.get('sequence', '')}\n")
            command = self.esmfold_command.format(input_fasta=str(fasta), output_dir=str(out_dir))
            try:
                subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
            except Exception as exc:
                raise RuntimeError(f"ESMFoldScorer local_esmfold_cli failed command={command!r}: {exc}") from exc
            rows: list[dict[str, Any] | None] = []
            pdbs = sorted(out_dir.glob("*.pdb"))
            for pdb in pdbs[: len(records)]:
                rows.append(
                    {
                        "ptm": None,
                        "plddt_mean": mean_plddt_from_pdb(pdb),
                        "tm_to_reference": None,
                        "pdb_path": str(pdb),
                        "metadata": {"ptm_unavailable": True, "backend": "local_esmfold_cli"},
                    }
                )
            while len(rows) < len(records):
                rows.append({"ptm": None, "plddt_mean": None, "tm_to_reference": None, "pdb_path": None, "metadata": {"missing_pdb": True}})
            return rows

