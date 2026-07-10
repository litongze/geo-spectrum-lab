#!/usr/bin/env python3
"""Score an ENSEMBLE (mean of predicted complex channels) of several checkpoints
on the held-out validation split, with the strict task-book metric.

Averaging H only helps if the models' phases are consistent (physics models
predict physically-scaled channels, so they can be); if phases were random the
mean would cancel. This script measures whether it helps.

    python scripts/score_ensemble.py --datadir "Round1_Map(2)" \
        --ckpts checkpoints/a.pt checkpoints/b.pt ...
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401

from wireless_twin.data.channel_dataset import detect_round
from wireless_twin.data.setup_config import load_setup
from wireless_twin.evaluation.metrics import evaluate_channels
from wireless_twin.evaluation.predictor import (
    load_model_from_checkpoint, predict_test_channels)
from score_holdout import reproduce_val_indices


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

    preds = []
    for c in args.ckpts:
        model, meta = load_model_from_checkpoint(c, device=args.device)
        p = predict_test_channels(model, val_pos, meta, device=args.device)
        s = evaluate_channels(p, val_gt, spec)
        print(f"  single {Path(c).name:28s} C={s['C']:.4f}  "
              f"(PAS {s['C1_PAS']:.4f} PDP {s['C2_PDP']:.4f})")
        preds.append(p)

    ens = np.mean(np.stack(preds, 0), axis=0)
    s = evaluate_channels(ens, val_gt, spec)
    print(f"=== ENSEMBLE of {len(preds)} : C={s['C']:.4f}  "
          f"(PAS {s['C1_PAS']:.4f} PDP {s['C2_PDP']:.4f} NMSE {s['C3_NMSE']:.4f})")


if __name__ == "__main__":
    main()
