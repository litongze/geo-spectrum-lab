#!/usr/bin/env python3
"""Write deterministic validation indices for strict holdout training."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from wireless_twin.data import load_round
from wireless_twin.evaluation.validation_splits import (
    build_validation_splits,
    split_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", required=True)
    parser.add_argument(
        "--split",
        required=True,
        choices=(
            "random_interpolation",
            "spatial_block_0",
            "spatial_block_1",
            "spatial_block_2",
            "test_matched",
            "power_stratified",
        ),
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rd = load_round(args.datadir, load_test=True, mmap_channels=True)
    raw_train_pos = rd.train.positions * rd.pos_std + rd.pos_mean
    splits = build_validation_splits(
        raw_train_pos,
        rd.train.channels,
        rd.test_positions,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    if args.split not in splits:
        available = ", ".join(sorted(splits))
        raise ValueError(f"split {args.split!r} is unavailable; choices: {available}")

    split = splits[args.split]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, split.val_idx.astype(np.int64))

    summary_path = Path(args.summary_out) if args.summary_out else out.with_suffix(".md")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = split_summary(args.split, split.val_idx, raw_train_pos, rd.train.channels)
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("# Validation Split\n\n")
        f.write(f"- split: `{args.split}`\n")
        f.write(f"- seed: `{args.seed}`\n")
        f.write(f"- val_fraction: `{args.val_fraction}`\n")
        f.write(f"- n_train: `{len(split.train_idx)}`\n")
        f.write(f"- n_val: `{len(split.val_idx)}`\n")
        f.write(f"- indices: `{out}`\n\n")
        f.write("| key | value |\n")
        f.write("|---|---:|\n")
        for key in sorted(summary):
            f.write(f"| {key} | {summary[key]:.9g} |\n")

    print(f"[make_validation_indices] wrote {out}; n_val={len(split.val_idx)}")
    print(f"[make_validation_indices] wrote {summary_path}")


if __name__ == "__main__":
    main()
