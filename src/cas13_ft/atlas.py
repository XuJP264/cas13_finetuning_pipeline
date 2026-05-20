from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

from .sequence import clean_protein_sequence, is_valid_length, sequence_hash


CAS13_KEYWORDS = ("Cas13", "cas13", "C2c2", "c2c2")


def load_atlas(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_operons(atlas: Any) -> Iterator[Dict[str, Any]]:
    if isinstance(atlas, list):
        for item in atlas:
            if isinstance(item, dict):
                yield item
        return
    if isinstance(atlas, dict):
        for key in ("operons", "data", "records", "items"):
            value = atlas.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        yield item
                return
        for value in atlas.values():
            if isinstance(value, dict):
                yield value


def _starts_with_vi(value: Any) -> bool:
    if value is None:
        return False
    return str(value).strip().upper().startswith("VI")


def is_vi_operon(operon: Dict[str, Any]) -> bool:
    summary = operon.get("summary") or {}
    return _starts_with_vi(summary.get("subtype")) or _starts_with_vi(summary.get("subtype_score"))


def _contains_keyword(value: Any, keywords: Iterable[str]) -> bool:
    text = str(value or "")
    return any(keyword in text for keyword in keywords)


def cas_gene_matches(cas_gene: Dict[str, Any], keywords: Iterable[str] = CAS13_KEYWORDS) -> bool:
    return _contains_keyword(cas_gene.get("gene_name"), keywords) or _contains_keyword(
        cas_gene.get("hmm_name"), keywords
    )


def inspect_atlas(atlas: Any, keywords: Iterable[str] = CAS13_KEYWORDS) -> Dict[str, int]:
    total_operons = 0
    total_cas_entries = 0
    raw_candidate_sequences = 0
    raw_candidate_with_protein = 0
    for operon in iter_operons(atlas):
        total_operons += 1
        cas_genes = operon.get("cas") or operon.get("cas_genes") or []
        if not isinstance(cas_genes, list):
            continue
        operon_is_vi = is_vi_operon(operon)
        total_cas_entries += len(cas_genes)
        for gene in cas_genes:
            if not isinstance(gene, dict):
                continue
            if operon_is_vi or cas_gene_matches(gene, keywords):
                raw_candidate_sequences += 1
                if gene.get("protein"):
                    raw_candidate_with_protein += 1
    return {
        "total_operons": total_operons,
        "total_cas_entries": total_cas_entries,
        "raw_candidate_sequences": raw_candidate_sequences,
        "raw_candidate_with_protein": raw_candidate_with_protein,
    }


def extract_cas13_records(
    atlas: Any,
    min_len: int = 200,
    max_len: int = 1500,
    keywords: Iterable[str] = CAS13_KEYWORDS,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    seen = set()
    for index, operon in enumerate(iter_operons(atlas)):
        cas_genes = operon.get("cas") or operon.get("cas_genes") or []
        if not isinstance(cas_genes, list):
            continue
        operon_is_vi = is_vi_operon(operon)
        for gene_index, gene in enumerate(cas_genes):
            if not isinstance(gene, dict):
                continue
            if not (operon_is_vi or cas_gene_matches(gene, keywords)):
                continue
            protein = clean_protein_sequence(gene.get("protein", ""))
            if not is_valid_length(protein, min_len, max_len):
                continue
            digest = sequence_hash(protein)
            if digest in seen:
                continue
            seen.add(digest)
            summary = operon.get("summary") or {}
            records.append(
                {
                    "id": f"operon{index}_cas{gene_index}_{digest[:12]}",
                    "sequence": protein,
                    "length": len(protein),
                    "sha256": digest,
                    "subtype": summary.get("subtype"),
                    "subtype_score": summary.get("subtype_score"),
                    "gene_name": gene.get("gene_name"),
                    "hmm_name": gene.get("hmm_name"),
                }
            )
    return records


def write_jsonl(records: Iterable[Dict[str, Any]], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_fasta(records: Iterable[Dict[str, Any]], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(f">{record['id']} len={record['length']} subtype={record.get('subtype')}\n")
            seq = record["sequence"]
            for start in range(0, len(seq), 80):
                handle.write(seq[start : start + 80] + "\n")


def write_csv(records: Iterable[Dict[str, Any]], path: str | Path) -> None:
    rows = list(records)
    fieldnames = ["id", "sequence", "length", "sha256", "subtype", "subtype_score", "gene_name", "hmm_name"]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
