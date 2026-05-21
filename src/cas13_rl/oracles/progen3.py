from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_device(device: str | None = None) -> str:
    if device and device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(dtype: str | None):
    if dtype in (None, "auto"):
        return None
    mapping = {"float32": torch.float32, "fp32": torch.float32, "float16": torch.float16, "fp16": torch.float16, "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}
    if dtype not in mapping:
        raise ValueError(f"Unsupported ProGen3 dtype: {dtype}")
    return mapping[dtype]


@dataclass
class ProGen3Oracle:
    model_name: str = "Profluent-Bio/progen3-219m"
    device: str = "auto"
    max_length: int = 1024
    dtype: str = "auto"
    code_path: str | None = None

    def __post_init__(self) -> None:
        self.device = resolve_device(self.device)
        torch_dtype = resolve_dtype(self.dtype)
        official_code = Path(self.code_path).expanduser() if self.code_path else Path("external/progen3/src")
        if official_code.exists():
            sys.path.insert(0, str(official_code.resolve()))
            try:
                from progen3.batch_preparer import ProGen3BatchPreparer  # type: ignore
                from progen3.modeling import ProGen3ForCausalLM  # type: ignore
                from progen3.scorer import ProGen3Scorer  # type: ignore

                kwargs = {}
                if torch_dtype is not None:
                    kwargs["torch_dtype"] = torch_dtype
                self.model = ProGen3ForCausalLM.from_pretrained(self.model_name, **kwargs)
                self.model = self.model.eval().to(self.device)
                self.batch_preparer = ProGen3BatchPreparer()
                self.scorer = ProGen3Scorer(model=self.model)
                self.tokenizer = None
                self.official = True
                return
            except Exception as exc:
                raise RuntimeError(
                    f"Found official ProGen3 code at {official_code}, but failed to load '{self.model_name}' "
                    f"on device '{self.device}'. ProGen3 upstream requires CUDA-class GPU dependencies such as "
                    f"megablocks/flash attention per its README. Reason: {type(exc).__name__}: {exc}"
                ) from exc
        self.official = False
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            kwargs = {"trust_remote_code": True}
            if torch_dtype is not None:
                kwargs["torch_dtype"] = torch_dtype
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
            self.model.to(self.device)
            self.model.eval()
            if getattr(self.tokenizer, "pad_token_id", None) is None and getattr(self.tokenizer, "eos_token", None):
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load real ProGen3 oracle '{self.model_name}' on device '{self.device}'. "
                f"Reason: {type(exc).__name__}: {exc}"
            ) from exc

    def score_one(self, sequence: str) -> Dict[str, float | str]:
        text = str(sequence)
        if getattr(self, "official", False):
            try:
                scores = self.scorer.score_batch(sequences=[text])
                if "mean_logprob" in scores:
                    mean_logprob = float(scores["mean_logprob"][0])
                elif "mean_log_likelihood" in scores:
                    mean_logprob = float(scores["mean_log_likelihood"][0])
                elif "normalized_log_likelihood" in scores:
                    mean_logprob = float(scores["normalized_log_likelihood"][0])
                else:
                    mean_logprob = float(scores["log_likelihood"][0]) / max(1, len(text))
                ppl = float(scores["perplexity"][0]) if "perplexity" in scores else float(math.exp(-mean_logprob))
            except Exception as exc:
                raise RuntimeError(f"Official ProGen3 scoring failed: {type(exc).__name__}: {exc}") from exc
            nll = -mean_logprob
            return {
                "sequence": text,
                "progen3_nll": nll,
                "progen3_perplexity": ppl,
                "progen3_mean_logprob": mean_logprob,
                "progen3_normalized_score": mean_logprob,
            }
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if input_ids.shape[-1] < 2:
            raise ValueError("ProGen3 scoring needs at least two tokens after tokenization")
        labels = input_ids.clone()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        nll = float(outputs.loss.detach().cpu().item())
        ppl = float(math.exp(nll)) if nll < 20 else float("inf")
        return {
            "sequence": text,
            "progen3_nll": nll,
            "progen3_perplexity": ppl,
            "progen3_mean_logprob": -nll,
            "progen3_normalized_score": -nll,
        }

    def score_many(self, sequences: Iterable[str]) -> List[Dict[str, float | str]]:
        return [self.score_one(seq) for seq in sequences]
