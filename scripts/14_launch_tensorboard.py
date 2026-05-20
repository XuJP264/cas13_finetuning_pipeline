#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="outputs")
    parser.add_argument("--port", type=int, default=6006)
    args = parser.parse_args()
    cmd = ["tensorboard", "--logdir", args.logdir, "--port", str(args.port)]
    print("running:", " ".join(cmd))
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
