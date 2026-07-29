#!/usr/bin/env python3
"""Train clean base/e35 checkpoints for representative validation splits."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_seeds(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def run(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="1890,3716,962,1022,2262")
    ap.add_argument("--root", default="checkpoints/clean_panel")
    ap.add_argument("--base-epochs", type=int, default=180)
    ap.add_argument("--e35-epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    common = [
        "model.n_freq_basis=16",
        "model.n_vis_rank=16",
        "model.scatter_dropout=0.1",
        f"train.epochs={args.base_epochs}",
        f"train.batch_size={args.batch}",
        "train.device=cuda",
    ]
    for split_seed in parse_seeds(args.seeds):
        outdir = Path(args.root) / f"s{split_seed}"
        outdir.mkdir(parents=True, exist_ok=True)
        base = outdir / "base.pt"
        e35 = outdir / "base_e35.pt"
        if args.force or not base.exists():
            run([
                sys.executable,
                "scripts/train.py",
                "--config",
                "configs/round1_scatter.yaml",
                "--set",
                *common,
                f"train.seed={split_seed}",
                f"train.split_seed={split_seed}",
                f"output.ckpt={base}",
            ])
        else:
            print(f"[skip] {base}", flush=True)
        if args.force or not e35.exists():
            run([
                sys.executable,
                "scripts/finetune_epscap.py",
                "--init",
                str(base),
                "--datadir",
                "Round1_Map(2)",
                "--out",
                str(e35),
                "--epochs",
                str(args.e35_epochs),
                "--lr",
                "2e-4",
                "--no-c3",
                "--eps",
                "3.5e-9",
                "--split-seed",
                str(split_seed),
                "--device",
                "cuda",
            ])
        else:
            print(f"[skip] {e35}", flush=True)


if __name__ == "__main__":
    main()
