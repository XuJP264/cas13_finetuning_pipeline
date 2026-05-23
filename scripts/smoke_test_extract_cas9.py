#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        input_json = root / "mini_atlas.json"
        out_fasta = root / "cas9.fasta"
        out_csv = root / "cas9.csv"
        out_jsonl = root / "cas9.jsonl"
        cas9_seq = "M" + "ACDEFGHIKLMNPQRSTVWY" * 5 + "X"
        payload = [
            {
                "operon_id": "op1",
                "summary": {"subtype": "II-A"},
                "cas": [
                    {"gene_name": "Cas1", "hmm_name": "cas1", "protein": "AAAA"},
                    {"gene_name": "cas2", "hmm_name": "cas2", "protein": "CCCC"},
                    {"gene_name": "Cas9_c4", "hmm_name": "profile", "protein": cas9_seq},
                    {"gene_name": "other", "hmm_name": "nope", "product": "hypothetical", "protein": "DDDD"},
                ],
            },
            {
                "operon_id": "op2",
                "summary": {"subtype": "II-C"},
                "cas": [
                    {"gene_name": "foo", "hmm_name": "Cas9-HMM", "sequence": cas9_seq.lower()},
                    {"gene_name": "bar", "annotation": "not this one", "protein": "EEEE"},
                ],
            },
        ]
        input_json.write_text(json.dumps(payload), encoding="utf-8")
        cmd = [
            sys.executable,
            "scripts/extract_cas9_from_atlas.py",
            "--input",
            str(input_json),
            "--out_fasta",
            str(out_fasta),
            "--out_csv",
            str(out_csv),
            "--out_jsonl",
            str(out_jsonl),
        ]
        result = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=True)
        assert "CAS9_RAW_MATCHES=2" in result.stdout, result.stdout
        assert "CAS9_UNIQUE_SEQUENCES=1" in result.stdout, result.stdout
        assert out_fasta.exists() and out_fasta.read_text(encoding="utf-8").count(">") == 1
        assert out_csv.exists() and len(out_csv.read_text(encoding="utf-8").splitlines()) == 2
        rows = [json.loads(line) for line in out_jsonl.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 1
        assert rows[0]["protein"] == cas9_seq
        print("SMOKE_TEST_EXTRACT_CAS9_OK")


if __name__ == "__main__":
    main()

