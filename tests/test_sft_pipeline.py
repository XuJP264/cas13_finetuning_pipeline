from __future__ import annotations

import json

from transformers import Trainer, TrainingArguments

from cas13_ft.atlas import extract_cas13_records, read_jsonl, write_jsonl
from cas13_ft.dataset import CausalProteinCollator, ProteinJsonlDataset
from cas13_ft.modeling import load_causal_lm, load_tokenizer
from cas13_ft.sequence import clean_protein_sequence
from cas13_ft.splits import make_splits


AA = "ACDEFGHIKLMNPQRSTVWY"


def test_clean_sequence_removes_illegal_characters():
    assert clean_protein_sequence("acdxBZ*-") == "ACD"


def test_extract_cas13_records_vi_and_keyword():
    good = AA * 15
    atlas = {
        "operons": [
            {"summary": {"subtype": "VI-A"}, "cas": [{"gene_name": "other", "hmm_name": "x", "protein": good}]},
            {"summary": {"subtype": "I-E"}, "cas": [{"gene_name": "Cas13a", "hmm_name": "x", "protein": good + "X"}]},
            {"summary": {"subtype": "I-E"}, "cas": [{"gene_name": "cas9", "hmm_name": "x", "protein": good}]},
        ]
    }
    records = extract_cas13_records(atlas, min_len=20, max_len=400)
    assert len(records) == 1
    assert records[0]["sequence"] == good


def test_split_has_no_duplicate_sequences():
    records = [{"id": str(i), "sequence": AA * 10 + str(i % 7)} for i in range(30)]
    train, valid, test = make_splits(records, seed=7)
    train_s = {r["sequence"] for r in train}
    valid_s = {r["sequence"] for r in valid}
    test_s = {r["sequence"] for r in test}
    assert not train_s & valid_s
    assert not train_s & test_s
    assert not valid_s & test_s


def test_dataset_collator(tmp_path):
    data = tmp_path / "data.jsonl"
    write_jsonl([{"id": "a", "sequence": "ACD"}, {"id": "b", "sequence": "ACDEFG"}], data)
    tok = load_tokenizer()
    ds = ProteinJsonlDataset(str(data), tok, max_length=16)
    batch = CausalProteinCollator(tok.pad_token_id)([ds[0], ds[1]])
    assert batch["input_ids"].shape[0] == 2
    assert batch["labels"].shape == batch["input_ids"].shape
    assert (batch["labels"][batch["attention_mask"] == 0] == -100).all()


def test_tiny_train_smoke(tmp_path):
    data = tmp_path / "train.jsonl"
    rows = [{"id": f"s{i}", "sequence": AA[:10] * 3} for i in range(4)]
    write_jsonl(rows, data)
    tok = load_tokenizer()
    model = load_causal_lm(None, vocab_size=len(tok))
    ds = ProteinJsonlDataset(str(data), tok, max_length=64)
    args = TrainingArguments(
        output_dir=str(tmp_path / "out"),
        per_device_train_batch_size=2,
        max_steps=1,
        logging_steps=1,
        report_to=[],
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=CausalProteinCollator(tok.pad_token_id))
    result = trainer.train()
    assert result.training_loss >= 0
