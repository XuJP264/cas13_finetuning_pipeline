#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def load_history(output_dir: Path) -> list[dict]:
    state_path = output_dir / "trainer_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"trainer_state.json not found: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    history = state.get("log_history") or []
    if not isinstance(history, list):
        raise ValueError(f"trainer_state.json log_history is not a list: {state_path}")
    return [row for row in history if isinstance(row, dict)]


def write_csv(history: list[dict], output_dir: Path) -> Path:
    out = output_dir / "loss_curve.csv"
    fields = ["step", "epoch", "loss", "eval_loss", "learning_rate"]
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in history:
            if not any(key in row for key in ("loss", "eval_loss", "learning_rate")):
                continue
            writer.writerow({field: row.get(field) for field in fields})
    return out


def plot_curves(history: list[dict], output_dir: Path) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required to write PNG curves. Install with: python3 -m pip install matplotlib"
        ) from exc

    written: list[Path] = []
    train = [(row.get("step"), row.get("loss")) for row in history if row.get("step") is not None and row.get("loss") is not None]
    valid = [(row.get("step"), row.get("eval_loss")) for row in history if row.get("step") is not None and row.get("eval_loss") is not None]
    if train or valid:
        plt.figure(figsize=(8, 5))
        if train:
            plt.plot([x for x, _ in train], [y for _, y in train], label="train_loss")
        if valid:
            plt.plot([x for x, _ in valid], [y for _, y in valid], label="eval_loss")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        out = output_dir / "loss_curve.png"
        plt.tight_layout()
        plt.savefig(out, dpi=160)
        plt.close()
        written.append(out)

    lr = [(row.get("step"), row.get("learning_rate")) for row in history if row.get("step") is not None and row.get("learning_rate") is not None]
    if lr:
        plt.figure(figsize=(8, 4))
        plt.plot([x for x, _ in lr], [y for _, y in lr], label="learning_rate")
        plt.xlabel("step")
        plt.ylabel("learning_rate")
        plt.legend()
        plt.grid(True, alpha=0.3)
        out = output_dir / "lr_curve.png"
        plt.tight_layout()
        plt.savefig(out, dpi=160)
        plt.close()
        written.append(out)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs/cas13_raw_2epoch")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    history = load_history(output_dir)
    csv_path = write_csv(history, output_dir)
    print(f"LOSS_CURVE_CSV={csv_path}")
    try:
        written = plot_curves(history, output_dir)
    except RuntimeError as exc:
        print(f"PLOT_ERROR={exc}", file=sys.stderr)
        raise SystemExit(1) from None
    for path in written:
        print(f"CURVE_PNG={path}")


if __name__ == "__main__":
    main()
