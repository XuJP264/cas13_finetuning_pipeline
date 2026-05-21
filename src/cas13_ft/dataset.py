from __future__ import annotations

from dataclasses import dataclass
from typing import List
import warnings

import torch
from torch.utils.data import Dataset

from .atlas import read_jsonl


class ProteinJsonlDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int = 1024, append_eos: bool = True):
        self.records = read_jsonl(path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.append_eos = append_eos
        self.eos_token_id = getattr(tokenizer, "eos_token_id", None)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        sequence = record.get("sequence")
        if sequence is None:
            sequence = record.get("protein")
            warnings.warn(
                f"{self.__class__.__name__} record {idx} is missing 'sequence'; falling back to 'protein'. "
                "Please regenerate JSONL files with a canonical 'sequence' field.",
                RuntimeWarning,
                stacklevel=2,
            )
        if sequence is None:
            raise KeyError(f"Record {idx} has neither 'sequence' nor 'protein'")
        ids = self.tokenizer.encode(sequence, add_special_tokens=self.append_eos)
        raw_tokenized_length = len(ids)
        truncated = raw_tokenized_length > self.max_length
        ids = ids[: self.max_length]
        if self.append_eos and self.eos_token_id is not None and ids:
            if truncated and ids[-1] != self.eos_token_id:
                ids[-1] = self.eos_token_id
            elif not truncated and ids[-1] != self.eos_token_id:
                ids.append(self.eos_token_id)
                if len(ids) > self.max_length:
                    ids = ids[: self.max_length]
                    ids[-1] = self.eos_token_id
                    truncated = True
        return {
            "input_ids": ids,
            "sequence": sequence,
            "original_length": len(sequence),
            "raw_protein_length": len(sequence),
            "raw_tokenized_length": raw_tokenized_length,
            "tokenized_length": len(ids),
            "truncated": truncated,
            "eos_in_input_ids": self.eos_token_id is not None and self.eos_token_id in ids,
            "has_eos": self.eos_token_id is not None and self.eos_token_id in ids,
        }

    def truncation_summary(self) -> dict:
        if not self.records:
            return {"count": 0, "truncated_count": 0, "truncation_ratio": 0.0}
        truncated_count = 0
        for idx in range(len(self.records)):
            item = self[idx]
            truncated_count += int(bool(item["truncated"]))
        return {
            "count": len(self.records),
            "truncated_count": truncated_count,
            "truncation_ratio": truncated_count / len(self.records),
        }


@dataclass
class CausalProteinCollator:
    pad_token_id: int = 0

    def __call__(self, features: List[dict]) -> dict:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids = []
        attention_mask = []
        for feature in features:
            ids = list(feature["input_ids"])
            pad = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad)
            attention_mask.append([1] * len(ids) + [0] * pad)
        input_tensor = torch.tensor(input_ids, dtype=torch.long)
        mask_tensor = torch.tensor(attention_mask, dtype=torch.long)
        labels = input_tensor.clone()
        labels[mask_tensor == 0] = -100
        return {"input_ids": input_tensor, "attention_mask": mask_tensor, "labels": labels}
