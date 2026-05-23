#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import EsmForProteinFolding


def main():
    parser = argparse.ArgumentParser(description="ESMFold batch sequence scoring oracle")
    parser.add_argument("--input", required=True, help="Input JSONL file with sequences")
    parser.add_argument("--output", required=True, help="Output JSONL file with scores")
    parser.add_argument("--model", default="facebook/esmfold_v1", help="ESMFold model name or path")
    parser.add_argument("--device", default="cuda:0", help="Device to use (cuda:0, cpu, etc.)")
    parser.add_argument("--max-length", type=int, default=1500, help="Maximum sequence length")
    parser.add_argument("--chunk-size", type=int, default=64, help="Chunk size for ESMFold trunk")
    args = parser.parse_args()

    # Print required debug information and flush immediately
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}", flush=True)
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}", flush=True)
    if torch.cuda.is_available():
        print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}", flush=True)
    print(f"args.device: {args.device}", flush=True)
    print(f"args.chunk_size: {args.chunk_size}", flush=True)

    # Load sequences from input
    sequences = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                seq = data.get("sequence", "").strip()
                if seq:
                    sequences.append(seq)
            except Exception as e:
                print(f"Warning: Could not parse input line: {line[:100]}... Error: {e}", flush=True)

    # Load ESMFold model
    print(f"Loading model: {args.model}", flush=True)
    model = EsmForProteinFolding.from_pretrained(args.model)
    model = model.to(args.device)
    model.eval()

    # Set chunk size
    chunk_size_set = False
    if hasattr(model, "trunk") and hasattr(model.trunk, "set_chunk_size"):
        model.trunk.set_chunk_size(args.chunk_size)
        chunk_size_set = True
        print(f"Set model.trunk.set_chunk_size({args.chunk_size})", flush=True)
    elif hasattr(model, "set_chunk_size"):
        model.set_chunk_size(args.chunk_size)
        chunk_size_set = True
        print(f"Set model.set_chunk_size({args.chunk_size})", flush=True)
    else:
        print(f"WARNING: Could not set chunk_size - model does not have supported set_chunk_size method", flush=True)

    # Process sequences
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as out_f:
        for seq_idx, sequence in enumerate(sequences):
            seq_len = len(sequence)
            print(f"[seq {seq_idx+1}/{len(sequences)}] length={seq_len}", flush=True)

            result = {
                "backend": "esmfold_v1",
                "sequence": sequence,
                "length": seq_len,
                "valid": False,
                "error": None,
                "mean_plddt": None,
                "ptm": None,
                "mean_pae": None,
                "pdb_path": None
            }

            try:
                if seq_len > args.max_length:
                    raise ValueError(f"Sequence length {seq_len} exceeds max_length {args.max_length}")

                # Tokenize sequence (simplified, using model's tokenizer)
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(args.model)
                inputs = tokenizer([sequence], return_tensors="pt", add_special_tokens=True)
                inputs = {k: v.to(args.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)

                # Extract metrics
                mean_plddt = float(outputs["aligned_confidence_probs"][0].mean().item()) * 100  # Normalize to 0-100
                ptm = float(outputs["ptm"][0].item())
                mean_pae = float(outputs["predicted_aligned_error"][0].mean().item())

                result["mean_plddt"] = mean_plddt
                result["ptm"] = ptm
                result["mean_pae"] = mean_pae
                result["valid"] = True
                print(f"[seq {seq_idx+1}/{len(sequences)}] length={seq_len} valid={result['valid']} error={result['error']} mean_plddt={mean_plddt:.2f} ptm={ptm:.3f}", flush=True)

            except Exception as e:
                result["error"] = f"{type(e).__name__}: {str(e)}"
                result["valid"] = False
                print(f"[seq {seq_idx+1}/{len(sequences)}] length={seq_len} valid={result['valid']} error={result['error']}", flush=True)

            # Write to output
            out_f.write(json.dumps(result, ensure_ascii=False) + '\n')
            out_f.flush()

    print(f"Processed {len(sequences)} sequences, output written to {args.output}", flush=True)


if __name__ == "__main__":
    main()