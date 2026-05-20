#!/usr/bin/env python
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path


ASSETS = {
    "checkpoint": "https://zenodo.org/records/15128064/files/progen2-base-ft.ckpt?download=1",
    "config": "https://zenodo.org/records/15128064/files/progen2-base-ft-config.json?download=1",
    "atlas": "https://storage.googleapis.com/crispr-cas-atlas-xy7q13lmk9/crispr-cas-atlas-v1.0.json",
}


def download(url: str, path: Path, force: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        print(f"exists: {path}")
        return
    print(f"downloading {url} -> {path}")
    try:
        urllib.request.urlretrieve(url, path)
    except Exception as exc:
        raise SystemExit(f"Download failed for {url}: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    raw = Path(args.raw_dir)
    download(ASSETS["checkpoint"], raw / "progen2-base-ft.ckpt", args.force)
    download(ASSETS["config"], raw / "progen2-base-ft-config.json", args.force)
    download(ASSETS["atlas"], raw / "crispr-cas-atlas-v1.0.json", args.force)


if __name__ == "__main__":
    main()
