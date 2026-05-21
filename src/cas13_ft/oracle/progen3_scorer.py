from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


@dataclass
class ProGen3Scorer:
    model_name: str = "Profluent-Bio/progen3-1b"
    device: str = "auto"
    dtype: str = "auto"
    cache_dir: str | None = None

    def __post_init__(self) -> None:
        self.device = _resolve_device(self.device)
        self.tokenizer = None
        self.model = None
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise RuntimeError(f"ProGen3Scorer requires torch/transformers for model_name={self.model_name}: {exc}") from exc
        try:
            self.torch = torch
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, cache_dir=self.cache_dir)
            if getattr(self.tokenizer, "pad_token_id", None) is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            kwargs: dict[str, Any] = {"trust_remote_code": True, "cache_dir": self.cache_dir}
            if self.dtype not in {None, "auto"}:
                dtype_map = {"float16": torch.float16, "fp16": torch.float16, "bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "float32": torch.float32, "fp32": torch.float32}
                kwargs["torch_dtype"] = dtype_map.get(str(self.dtype), None)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs).to(self.device)
            self.model.eval()
        except Exception as exc:
            raise RuntimeError(
                f"ProGen3Scorer failed to load model_name={self.model_name} device={self.device} dtype={self.dtype}: {type(exc).__name__}: {exc}"
            ) from exc

    def score_sequences(self, sequences: list[str], batch_size: int = 1) -> list[dict[str, float | int]]:
        rows: list[dict[str, float | int]] = []
        torch = self.torch
        for start in range(0, len(sequences), max(1, int(batch_size))):
            batch = [str(seq) for seq in sequences[start : start + max(1, int(batch_size))]]
            encoded = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, add_special_tokens=True)
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            labels = input_ids.clone()
            pad_id = getattr(self.tokenizer, "pad_token_id", None)
            if pad_id is not None:
                labels[labels == pad_id] = -100
            labels[attention_mask == 0] = -100
            with torch.no_grad():
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(shift_labels.shape)
            valid_mask = shift_labels.ne(-100)
            for i in range(len(batch)):
                token_count = int(valid_mask[i].sum().detach().cpu().item())
                if token_count == 0:
                    nll = float("inf")
                    mean_ll = float("-inf")
                else:
                    nll = float((losses[i] * valid_mask[i]).sum().detach().cpu().item() / token_count)
                    mean_ll = -nll
                rows.append(
                    {
                        "mean_log_likelihood": mean_ll,
                        "nll": nll,
                        "ppl": float(math.exp(nll)) if nll < 20 else float("inf"),
                        "num_tokens": token_count,
                    }
                )
        return rows

