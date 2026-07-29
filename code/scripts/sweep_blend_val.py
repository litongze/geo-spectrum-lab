#!/usr/bin/env python3
"""Evaluate alpha blends of two prediction files on a validation split."""

from __future__ import annotations

import argparse

import numpy as np

import _bootstrap  # noqa: F401
from wireless_twin.data import load_round
from wireless_twin.evaluation.metrics import evaluate_channels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", required=True)
    parser.add_argument("--val-indices", required=True)
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--alphas", default="0,0.5,0.8,0.9,0.95,1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rd = load_round(args.datadir, load_test=False, mmap_channels=True)
    val_idx = np.load(args.val_indices).astype(np.int64)
    gt = np.asarray(rd.train.channels[val_idx])
    a = np.load(args.a)
    b = np.load(args.b)
    if a.shape != b.shape or a.shape != gt.shape:
        raise ValueError(f"shape mismatch: a={a.shape}, b={b.shape}, gt={gt.shape}")
    best = None
    for alpha_text in args.alphas.split(","):
        alpha = float(alpha_text)
        pred = alpha * a + (1.0 - alpha) * b
        scores = evaluate_channels(pred.astype(np.complex64), gt, rd.spec)
        row = (
            alpha,
            scores["C"],
            scores["C1_PAS"],
            scores["C2_PDP"],
            scores["C3_NMSE"],
        )
        if best is None or row[1] > best[1]:
            best = row
        print(
            f"alpha={alpha:.4f} C={row[1]:.9f} "
            f"PAS={row[2]:.9f} PDP={row[3]:.9f} NMSE={row[4]:.9f}")
    assert best is not None
    print(
        f"[sweep_blend_val] best alpha={best[0]:.4f} C={best[1]:.9f} "
        f"PAS={best[2]:.9f} PDP={best[3]:.9f} NMSE={best[4]:.9f}")


if __name__ == "__main__":
    main()
