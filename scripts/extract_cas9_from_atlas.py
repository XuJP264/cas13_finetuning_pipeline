#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterator

VALID_PROTEIN_CHARS = set("ACDEFGHIKLMNPQRSTVWYX")
CAS9_FIELDS = (
    "gene_name",
    "hmm_name",
    "product",
    "name",
    "annotation",
    "type",
    "family",
)
PROTEIN_FIELDS = ("protein", "sequence", "aa_sequence", "protein_sequence")
WRAPPER_KEYS = ("records", "items", "data", "operons")
EXPECTED_ATLAS_CAS9 = 238_917


def is_cas9(cas_entry: dict[str, Any]) -> bool:
    gene_name = str(cas_entry.get("gene_name") or "").strip()
    if gene_name == "Cas9" or "cas9" in gene_name.lower():
        return True
    for field in CAS9_FIELDS:
        value = cas_entry.get(field)
        if value is not None and "cas9" in str(value).lower():
            return True
    return False


def clean_protein(value: Any) -> str:
    raw = re.sub(r"\s+", "", str(value or "")).upper()
    return "".join(ch for ch in raw if ch in VALID_PROTEIN_CHARS)


def protein_from_entry(cas_entry: dict[str, Any]) -> str:
    for field in PROTEIN_FIELDS:
        protein = clean_protein(cas_entry.get(field))
        if protein:
            return protein
    return ""


def subtype_from_record(record: dict[str, Any]) -> Any:
    summary = record.get("summary")
    if isinstance(summary, dict):
        return summary.get("subtype")
    return record.get("subtype")


def operon_id_from_record(record: dict[str, Any], index: int) -> str:
    for key in ("operon_id", "id", "record_id", "accession"):
        value = record.get(key)
        if value is not None:
            return str(value)
    return f"operon_{index:06d}"


def _read_more(handle, buffer: str, chunk_size: int) -> str:
    chunk = handle.read(chunk_size)
    if not chunk:
        return buffer
    return buffer + chunk


def _skip_ws_and_commas(handle, buffer: str, pos: int, chunk_size: int) -> tuple[str, int]:
    while True:
        while pos < len(buffer) and buffer[pos] in " \t\r\n,":
            pos += 1
        if pos < len(buffer):
            return buffer, pos
        new_buffer = _read_more(handle, buffer[pos:], chunk_size)
        if len(new_buffer) == len(buffer[pos:]):
            return new_buffer, 0
        buffer = new_buffer
        pos = 0


def iter_json_array_objects_from_stream(handle, initial_buffer: str = "", initial_pos: int = 0, chunk_size: int = 1 << 20) -> Iterator[dict[str, Any]]:
    decoder = json.JSONDecoder()
    buffer = initial_buffer
    pos = initial_pos
    if not buffer:
        buffer = handle.read(chunk_size)
    buffer, pos = _skip_ws_and_commas(handle, buffer, pos, chunk_size)
    if pos >= len(buffer) or buffer[pos] != "[":
        raise ValueError("Expected JSON array")
    pos += 1
    while True:
        buffer, pos = _skip_ws_and_commas(handle, buffer, pos, chunk_size)
        if pos < len(buffer) and buffer[pos] == "]":
            return
        while True:
            try:
                value, end = decoder.raw_decode(buffer, pos)
                break
            except json.JSONDecodeError:
                old_tail = buffer[pos:]
                chunk = handle.read(chunk_size)
                if not chunk:
                    raise
                buffer = old_tail + chunk
                pos = 0
        if isinstance(value, dict):
            yield value
        pos = end
        if pos > chunk_size:
            buffer = buffer[pos:]
            pos = 0


def iter_records_from_dict_payload(payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for key in WRAPPER_KEYS:
        value = payload.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    yield item
            return
    for value in payload.values():
        if isinstance(value, dict):
            yield value


def iter_atlas_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        first = handle.read(1)
        while first and first.isspace():
            first = handle.read(1)
        if first == "[":
            yield from iter_json_array_objects_from_stream(handle, initial_buffer="[", initial_pos=0)
            return
        if first == "{":
            # The public v1.0 Atlas file is a top-level list. Dict support is kept
            # for smaller wrapped exports using only stdlib JSON.
            handle.seek(0)
            payload = json.load(handle)
            if not isinstance(payload, dict):
                raise ValueError("Expected top-level dict payload")
            yield from iter_records_from_dict_payload(payload)
            return
        raise ValueError(f"Unsupported JSON top-level token: {first!r}")


def fasta_wrap(sequence: str, width: int = 80) -> str:
    return "\n".join(sequence[i : i + width] for i in range(0, len(sequence), width))


def maybe_collect_unmatched_subtype_ii(record: dict[str, Any], cas_entry: dict[str, Any], samples: list[dict[str, Any]]) -> None:
    if len(samples) >= 20:
        return
    subtype = str(subtype_from_record(record) or "")
    if re.match(r"^II($|[-_])", subtype.upper()) is None:
        return
    if is_cas9(cas_entry):
        return
    samples.append(
        {
            "operon_id": record.get("operon_id") or record.get("id"),
            "subtype": subtype,
            "cas_entry_keys": sorted(cas_entry.keys()),
            "gene_name": cas_entry.get("gene_name"),
            "hmm_name": cas_entry.get("hmm_name"),
            "product": cas_entry.get("product"),
            "name": cas_entry.get("name"),
            "annotation": cas_entry.get("annotation"),
            "type": cas_entry.get("type"),
            "family": cas_entry.get("family"),
        }
    )


def extract(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input)
    out_fasta = Path(args.out_fasta)
    out_csv = Path(args.out_csv)
    out_jsonl = Path(args.out_jsonl)
    for path in (out_fasta, out_csv, out_jsonl):
        path.parent.mkdir(parents=True, exist_ok=True)

    input_records = 0
    cas_entries_scanned = 0
    raw_matches = 0
    unique_count = 0
    seen_sequences: set[str] = set()
    unmatched_subtype_ii: list[dict[str, Any]] = []

    with out_fasta.open("w", encoding="utf-8") as fasta_handle, out_csv.open("w", newline="", encoding="utf-8") as csv_handle, out_jsonl.open("w", encoding="utf-8") as jsonl_handle:
        csv_fields = ["id", "operon_id", "subtype", "gene_name", "hmm_name", "length", "sequence_length", "protein"]
        writer = csv.DictWriter(csv_handle, fieldnames=csv_fields)
        writer.writeheader()

        for record in iter_atlas_records(input_path):
            input_records += 1
            cas_entries = record.get("cas") or []
            if not isinstance(cas_entries, list):
                continue
            operon_id = operon_id_from_record(record, input_records - 1)
            subtype = subtype_from_record(record)
            for cas_entry in cas_entries:
                if not isinstance(cas_entry, dict):
                    continue
                cas_entries_scanned += 1
                maybe_collect_unmatched_subtype_ii(record, cas_entry, unmatched_subtype_ii)
                if not is_cas9(cas_entry):
                    continue
                raw_matches += 1
                protein = protein_from_entry(cas_entry)
                if not protein or protein in seen_sequences:
                    continue
                seen_sequences.add(protein)
                unique_count += 1
                row_id = f"cas9_{unique_count:06d}"
                gene_name = cas_entry.get("gene_name")
                hmm_name = cas_entry.get("hmm_name")
                fasta_header = f">{row_id}|operon_id={operon_id}|gene_name={gene_name or ''}|hmm_name={hmm_name or ''}|length={len(protein)}"
                fasta_handle.write(fasta_header + "\n" + fasta_wrap(protein) + "\n")
                writer.writerow(
                    {
                        "id": row_id,
                        "operon_id": operon_id,
                        "subtype": subtype,
                        "gene_name": gene_name,
                        "hmm_name": hmm_name,
                        "length": len(protein),
                        "sequence_length": len(protein),
                        "protein": protein,
                    }
                )
                jsonl_handle.write(
                    json.dumps(
                        {
                            "id": row_id,
                            "operon_id": operon_id,
                            "subtype": subtype,
                            "metadata": {
                                "length": len(protein),
                                "gene_name": gene_name,
                                "hmm_name": hmm_name,
                            },
                            "cas_entry": cas_entry,
                            "protein": protein,
                        },
                        ensure_ascii=True,
                        sort_keys=True,
                    )
                    + "\n"
                )

    return {
        "INPUT_RECORDS": input_records,
        "CAS_ENTRIES_SCANNED": cas_entries_scanned,
        "CAS9_RAW_MATCHES": raw_matches,
        "CAS9_UNIQUE_SEQUENCES": unique_count,
        "OUT_FASTA": str(out_fasta),
        "OUT_CSV": str(out_csv),
        "OUT_JSONL": str(out_jsonl),
        "UNMATCHED_SUBTYPE_II_SAMPLES": unmatched_subtype_ii,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract deduplicated Cas9 protein sequences from CRISPR-Cas Atlas JSON.")
    parser.add_argument("--input", default="data/raw/crispr-cas-atlas-v1.0.json")
    parser.add_argument("--out_fasta", default="data/processed/cas9_atlas.fasta")
    parser.add_argument("--out_csv", default="data/processed/cas9_atlas.csv")
    parser.add_argument("--out_jsonl", default="data/processed/cas9_atlas.jsonl")
    args = parser.parse_args()

    stats = extract(args)
    for key in ("INPUT_RECORDS", "CAS_ENTRIES_SCANNED", "CAS9_RAW_MATCHES", "CAS9_UNIQUE_SEQUENCES", "OUT_FASTA", "OUT_CSV", "OUT_JSONL"):
        print(f"{key}={stats[key]}")

    input_name = Path(args.input).name
    unique = int(stats["CAS9_UNIQUE_SEQUENCES"])
    if "crispr-cas-atlas" in input_name and abs(unique - EXPECTED_ATLAS_CAS9) > EXPECTED_ATLAS_CAS9 * 0.10:
        print("WARNING=CAS9_UNIQUE_SEQUENCES differs from public reference 238917 by more than 10%")
        print("UNMATCHED_SUBTYPE_II_SAMPLES_JSONL_BEGIN")
        for sample in stats["UNMATCHED_SUBTYPE_II_SAMPLES"]:
            print(json.dumps(sample, ensure_ascii=True, sort_keys=True))
        print("UNMATCHED_SUBTYPE_II_SAMPLES_JSONL_END")


if __name__ == "__main__":
    main()
