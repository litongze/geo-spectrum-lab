#!/usr/bin/env python3
"""Train the five clean PAS/PDP attention arms used by GS37."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_seeds(raw: str) -> list[int]:
    return [int(value) for value in raw.split(",") if value.strip()]


def run(command: list[str]) -> None:
    print("[run]", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", default="Round1_Map(2)")
    parser.add_argument("--seeds", default="1890,3716,962,1022,2262")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for split_seed in parse_seeds(args.seeds):
        pas_prefix = (
            f"clean_panel_phv_geom_k16/s{split_seed}/"
            "nbrattn_clean_k16"
        )
        pas_path = (
            Path("checkpoints")
            / f"{pas_prefix}_pas_k16s0.pt"
        )
        if args.force or not pas_path.is_file():
            run(
                [
                    sys.executable,
                    "scripts/train_nbr_arm.py",
                    "--datadir",
                    args.datadir,
                    "--device",
                    args.device,
                    "--K",
                    "16",
                    "--prefix",
                    pas_prefix,
                    "--epochs",
                    str(args.epochs),
                    "--seeds",
                    "0",
                    "--batch",
                    str(args.batch_size),
                    "--only",
                    "pas",
                    "--feature-set",
                    "geometry",
                    "--pas-layout",
                    "phv",
                    "--split-seed",
                    str(split_seed),
                    "--clean-holdout",
                ]
            )
        else:
            print(f"[skip] {pas_path}", flush=True)

        pdp_prefix = (
            f"clean_panel/s{split_seed}/nbrattn_clean_k32"
        )
        pdp_path = (
            Path("checkpoints")
            / f"{pdp_prefix}_pdp_k32s0.pt"
        )
        if args.force or not pdp_path.is_file():
            run(
                [
                    sys.executable,
                    "scripts/train_nbr_arm.py",
                    "--datadir",
                    args.datadir,
                    "--device",
                    args.device,
                    "--K",
                    "32",
                    "--prefix",
                    pdp_prefix,
                    "--epochs",
                    str(args.epochs),
                    "--seeds",
                    "0",
                    "--batch",
                    str(args.batch_size),
                    "--only",
                    "pdp",
                    "--split-seed",
                    str(split_seed),
                    "--clean-holdout",
                ]
            )
        else:
            print(f"[skip] {pdp_path}", flush=True)

    print("GS37_MOMENT_ARMS_DONE", flush=True)


if __name__ == "__main__":
    main()
