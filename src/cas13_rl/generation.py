from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

import torch

AA = "ACDEFGHIKLMNPQRSTVWY"


def load_prompts(path: str | Path, prompt_length: int = 64) -> List[str]:
    prompts = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                seq = json.loads(line)["sequence"]
                prompts.append(seq[:prompt_length])
    return prompts


def generate_mock_samples(prompts: Iterable[str], num_samples: int, max_new_tokens: int, seed: int = 1337) -> List[dict]:
    gen = torch.Generator().manual_seed(seed)
    prompts = list(prompts) or [""]
    rows = []
    for i in range(num_samples):
        prompt = prompts[i % len(prompts)]
        ids = torch.randint(0, len(AA), (max_new_tokens,), generator=gen).tolist()
        continuation = "".join(AA[j] for j in ids)
        seq = prompt + continuation
        rows.append({"id": f"sample_{i}", "prompt": prompt, "sequence": seq})
    return rows


def generate_policy_samples(
    model,
    tokenizer,
    prompts: Iterable[str],
    num_samples: int,
    max_new_tokens: int,
    temperature: float = 0.9,
    top_p: float = 0.95,
) -> List[dict]:
    rows = []
    prompts = list(prompts) or ["M"]
    model.eval()
    device = next(model.parameters()).device
    for i in range(num_samples):
        prompt = prompts[i % len(prompts)]
        input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], dtype=torch.long, device=device)
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=getattr(tokenizer, "pad_token_id", 0) or 0,
                eos_token_id=getattr(tokenizer, "eos_token_id", None),
            )
        sequence = tokenizer.decode(output[0].detach().cpu().tolist(), skip_special_tokens=True)
        rows.append({"id": f"sample_{i}", "prompt": prompt, "sequence": sequence})
    return rows


def write_samples_jsonl(rows: Iterable[dict], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
