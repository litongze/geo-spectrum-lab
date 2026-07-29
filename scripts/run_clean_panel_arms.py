#!/usr/bin/env python3
"""Train clean neighbor-attention arms for representative validation splits."""
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
    ap.add_argument("--panel", default="1890,3716,962,1022,2262")
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--root", default="clean_panel")
    ap.add_argument("--only", choices=["both", "pas", "pdp"], default="both")
    ap.add_argument(
        "--feature-set",
        choices=["basic", "geometry"],
        default="basic",
    )
    ap.add_argument(
        "--pas-layout",
        choices=["legacy_hvp", "pvh", "phv"],
        default="legacy_hvp",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    arm_seeds = parse_seeds(args.seeds)
    for split_seed in parse_seeds(args.panel):
        prefix = f"{args.root}/s{split_seed}/nbrattn_clean_k{args.K}"
        outdir = Path("checkpoints") / args.root / f"s{split_seed}"
        tags = ("pas", "pdp") if args.only == "both" else (args.only,)
        expected = [
            outdir / f"nbrattn_clean_k{args.K}_{tag}_k{args.K}s{sd}.pt"
            for sd in arm_seeds
            for tag in tags
        ]
        if not args.force and all(p.exists() for p in expected):
            print(f"[skip] split {split_seed} K{args.K} arms already exist", flush=True)
            continue
        run([
            sys.executable,
            "scripts/train_nbr_arm.py",
            "--K",
            str(args.K),
            "--prefix",
            prefix,
            "--epochs",
            str(args.epochs),
            "--seeds",
            *[str(x) for x in arm_seeds],
            "--split-seed",
            str(split_seed),
            "--clean-holdout",
            "--only",
            args.only,
            "--pas-layout",
            args.pas_layout,
            "--feature-set",
            args.feature_set,
        ])


if __name__ == "__main__":
    main()
