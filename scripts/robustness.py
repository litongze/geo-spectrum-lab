#!/usr/bin/env python3
"""Robustness to the (undisclosed) metric weights.

C1/C2/C3 are properties of a model's predictions and do NOT depend on the
weights w=(w1,w2,w3); only how they combine into C does.  So we score each
checkpoint's C1/C2/C3 once on the held-out split, then report C over a grid of
plausible weightings — a model that stays on top across the grid is robust to
the unknown official weights.

    python scripts/robustness.py --datadir "Round1_Map(2)" \
        --ckpts checkpoints/round1_best.pt checkpoints/round1_gpu.pt
"""
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401

from wireless_twin.data.channel_dataset import detect_round
from wireless_twin.data.setup_config import load_setup
from wireless_twin.evaluation.metrics import (
    channel_nmse, competition_score, pas_accuracy, pdp_accuracy)
from wireless_twin.evaluation.predictor import (
    load_model_from_checkpoint, predict_test_channels)
from score_holdout import reproduce_val_indices

# plausible weight grid (w3 = 1 - w1 - w2), covers PAS-heavy / PDP-heavy /
# NMSE-heavy / balanced regimes
GRID = [(0.4, 0.4, 0.2), (0.33, 0.33, 0.34), (0.5, 0.3, 0.2),
        (0.3, 0.5, 0.2), (0.45, 0.45, 0.1), (0.3, 0.3, 0.4),
        (0.2, 0.2, 0.6), (0.6, 0.2, 0.2), (0.2, 0.6, 0.2)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datadir", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    d = Path(args.datadir)
    tag = detect_round(args.datadir)
    spec = load_setup(d / f"{tag}_Setup.json")
    pos = np.load(d / f"{tag}_Train_Pos.npy").astype(np.float32)
    ch = np.load(d / f"{tag}_Train_Channel.npy")
    val_idx = reproduce_val_indices(len(pos), args.val_fraction, args.seed)
    val_pos, val_gt = pos[val_idx], ch[val_idx]

    rows = []
    for c in args.ckpts:
        model, meta = load_model_from_checkpoint(c, device=args.device)
        pred = predict_test_channels(model, val_pos, meta, device=args.device)
        c1 = pas_accuracy(pred, val_gt, spec)
        c2 = pdp_accuracy(pred, val_gt, spec)
        c3 = channel_nmse(pred, val_gt)
        rows.append((Path(c).name, c1, c2, c3, meta["model_name"]))

    hdr = "model".ljust(22) + "C1     C2     C3   |  " + \
        "  ".join(f"{w1:.2f}/{w2:.2f}/{w3:.2f}" for w1, w2, w3 in GRID)
    print(hdr)
    print("-" * len(hdr))
    for name, c1, c2, c3, mname in rows:
        cs = [competition_score(c1, c2, c3, w) for w in GRID]
        print(f"{name[:22]:22}{c1:.3f}  {c2:.3f}  {c3:.3f} | " +
              "  ".join(f"{v:.3f}" + " " * 6 for v in cs))
    # winner per weighting
    print("\nbest model per weighting:")
    for j, w in enumerate(GRID):
        vals = [competition_score(r[1], r[2], r[3], w) for r in rows]
        k = int(np.argmax(vals))
        print(f"  w={w}: {rows[k][0]}  (C={vals[k]:.4f})")


if __name__ == "__main__":
    main()
