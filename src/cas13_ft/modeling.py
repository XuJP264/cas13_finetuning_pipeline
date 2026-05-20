from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase


class ProteinCharTokenizer:
    """Tiny local tokenizer for smoke tests and amino-acid CLM fallback."""

    alphabet = list("ACDEFGHIKLMNPQRSTVWY")

    def __init__(self, vocab_size: int | None = None):
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"
        tokens = [self.pad_token, self.eos_token, self.unk_token] + self.alphabet
        if vocab_size and vocab_size > len(tokens):
            tokens.extend([f"<extra_{i}>" for i in range(vocab_size - len(tokens))])
        self.vocab = {tok: i for i, tok in enumerate(tokens)}
        self.inv_vocab = {i: tok for tok, i in self.vocab.items()}
        self.pad_token_id = self.vocab[self.pad_token]
        self.eos_token_id = self.vocab[self.eos_token]
        self.unk_token_id = self.vocab[self.unk_token]

    def __len__(self) -> int:
        return len(self.vocab)

    def encode(self, text: str, add_special_tokens: bool = True, **_: object) -> list[int]:
        ids = [self.vocab.get(ch, self.unk_token_id) for ch in str(text)]
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids, skip_special_tokens: bool = True, **_: object) -> str:
        chars = []
        for idx in ids:
            token = self.inv_vocab.get(int(idx), self.unk_token)
            if skip_special_tokens and token.startswith("<"):
                continue
            chars.append(token)
        return "".join(chars)

    def save_pretrained(self, path: str | Path) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        with (p / "protein_char_tokenizer.json").open("w", encoding="utf-8") as handle:
            json.dump({"type": "ProteinCharTokenizer", "vocab_size": len(self)}, handle)

    @classmethod
    def from_pretrained(cls, path: str | Path):
        meta_path = Path(path) / "protein_char_tokenizer.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return cls(vocab_size=meta.get("vocab_size"))
        return cls()


def load_tokenizer(tokenizer_name_or_path: Optional[str] = None, vocab_size: int | None = None):
    if tokenizer_name_or_path:
        path = Path(tokenizer_name_or_path)
        if (path / "protein_char_tokenizer.json").exists():
            return ProteinCharTokenizer.from_pretrained(path)
        return AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
    return ProteinCharTokenizer(vocab_size=vocab_size)


def _load_progen_from_config(config_path: str | Path):
    import json

    from cas13_ft.progen_remote import ProGenConfig, ProGenForCausalLM

    raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    config = ProGenConfig(
        vocab_size_emb=raw.get("vocab_size", raw.get("vocab_size_emb", 32)),
        vocab_size_lm_head=raw.get("vocab_size", raw.get("vocab_size_lm_head", 32)),
        n_positions=raw.get("n_positions", 2048),
        embed_dim=raw.get("n_embd", raw.get("embed_dim", 1536)),
        n_layer=raw.get("n_layer", 27),
        n_head=raw.get("n_head", 16),
        rotary_dim=raw.get("rotary_dim", 48),
        activation_function=raw.get("activation_function", "gelu_new"),
        embd_pdrop=raw.get("embd_pdrop", 0.0),
        attn_pdrop=raw.get("attn_pdrop", 0.0),
        layer_norm_epsilon=raw.get("layer_norm_epsilon", 1e-5),
        initializer_range=raw.get("initializer_range", 0.02),
        gradient_checkpointing=raw.get("gradient_checkpointing", False),
        use_cache=raw.get("use_cache", True),
        bos_token_id=raw.get("bos_token_id", 1),
        eos_token_id=raw.get("eos_token_id", 2),
    )
    return ProGenForCausalLM(config)


def load_causal_lm(
    model_name_or_path: Optional[str],
    checkpoint_path: Optional[str] = None,
    config_path: Optional[str] = None,
    vocab_size: Optional[int] = None,
):
    if model_name_or_path:
        path = Path(model_name_or_path)
        config_file = path / "config.json"
        if config_file.exists() and '"model_type": "progen"' in config_file.read_text(encoding="utf-8"):
            model = _load_progen_from_config(config_file)
            safetensors_path = path / "model.safetensors"
            bin_path = path / "pytorch_model.bin"
            if safetensors_path.exists():
                from safetensors.torch import load_file

                state = load_file(str(safetensors_path), device="cpu")
            elif bin_path.exists():
                state = torch.load(bin_path, map_location="cpu")
            else:
                raise FileNotFoundError(f"No model weights found in {path}")
            model.load_state_dict(state, strict=False)
            return model
        return AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)
    if config_path and Path(config_path).exists():
        try:
            config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
            if vocab_size is not None and hasattr(config, "vocab_size"):
                config.vocab_size = vocab_size
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        except ValueError as exc:
            raw = Path(config_path).read_text(encoding="utf-8")
            if '"model_type": "progen"' not in raw:
                raise
            print(f"AutoConfig does not include ProGen support; using bundled ProGen remote-code adapter: {exc}")
            model = _load_progen_from_config(config_path)
    else:
        from transformers import GPT2Config, GPT2LMHeadModel

        config = GPT2Config(
            vocab_size=vocab_size or 23,
            n_positions=256,
            n_embd=64,
            n_layer=2,
            n_head=2,
            bos_token_id=1,
            eos_token_id=1,
            pad_token_id=0,
        )
        model = GPT2LMHeadModel(config)
    if checkpoint_path and Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict):
            state = state.get("state_dict") or state.get("model_state_dict") or state
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"Loaded checkpoint with missing={len(missing)} unexpected={len(unexpected)}")
    return model
