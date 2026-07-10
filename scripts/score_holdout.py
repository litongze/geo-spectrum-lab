#!/usr/bin/env python3
"""Score a checkpoint on the held-out validation split using the STRICT task-book
metric (evaluation/metrics.py): C1(PAS), C2(PDP), C3(NMSE global) and
C = w1*C1 + w2*C2 + w3/(1+C3).

The validation split is reproduced exactly from the trainer's seed so the number
is a faithful, self-consistent estimate of the leaderboard formula on unseen
positions (the organiser's hidden test GT is not available locally).

    python scripts/score_holdout.py --ckpt checkpoints/round1_best.pt \
        --datadir "Round1_Map(2)"
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

import _bootstrap  # noqa: F401

from wireless_twin.data.channel_dataset import detect_round
from wireless_twin.data.setup_config import load_setup
from wireless_twin.evaluation.metrics import evaluate_channels
from wireless_twin.evaluation.predictor import (
    load_model_from_checkpoint, predict_test_channels)


def reproduce_val_indices(n: int, val_fraction: float, seed: int):
    """Match torch.utils.data.random_split(seed) used by the Trainer."""
    n_val = int(n * val_fraction)
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen).tolist()
    return perm[n - n_val:]          # random_split gives the tail to the 2nd part


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--datadir", required=True)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from pathlib import Path
    tag = detect_round(args.datadir)
    d = Path(args.datadir)
    spec = load_setup(d / f"{tag}_Setup.json")
    pos = np.load(d / f"{tag}_Train_Pos.npy").astype(np.float32)
    ch = np.load(d / f"{tag}_Train_Channel.npy")

    val_idx = reproduce_val_indices(len(pos), args.val_fraction, args.seed)
    val_pos, val_gt = pos[val_idx], ch[val_idx]

    model, meta = load_model_from_checkpoint(args.ckpt, device=args.device)
    pred = predict_test_channels(model, val_pos, meta, device=args.device)

    scores = evaluate_channels(pred, val_gt, spec)
    print(f"=== STRICT task-book score on {len(val_idx)} held-out positions ===")
    print(f"model    : {meta['model_name']}  ({args.ckpt})")
    print(f"C1 (PAS) : {scores['C1_PAS']:.4f}")
    print(f"C2 (PDP) : {scores['C2_PDP']:.4f}")
    print(f"C3 (NMSE): {scores['C3_NMSE']:.4f}")
    print(f"C (w={spec.metric_weights}) : {scores['C']:.4f}")


if __name__ == "__main__":
    main()
