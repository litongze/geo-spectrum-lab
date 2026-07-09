#!/usr/bin/env python3
"""Score a predicted channel file against ground truth (offline C1/C2/C3/C).

Useful for local validation: hold out part of the training set, predict it, and
check the leaderboard metric before submitting.

Usage
-----
    python scripts/evaluate.py --pred pred.npy --gt gt.npy \
        --setup data/Data1/Round1_Setup.json
"""

from __future__ import annotations

import argparse

import numpy as np

import _bootstrap  # noqa: F401

from wireless_twin.data.setup_config import load_setup
from wireless_twin.evaluation.metrics import evaluate_channels


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred", required=True, help="predicted channels .npy")
    p.add_argument("--gt", required=True, help="ground-truth channels .npy")
    p.add_argument("--setup", required=True, help="RoundX_Setup.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    spec = load_setup(args.setup)
    pred = np.load(args.pred)
    gt = np.load(args.gt)

    scores = evaluate_channels(pred, gt, spec)
    print("=" * 40)
    print(f"C1 (PAS  cosine)  : {scores['C1_PAS']:.4f}")
    print(f"C2 (PDP  cosine)  : {scores['C2_PDP']:.4f}")
    print(f"C3 (NMSE)         : {scores['C3_NMSE']:.4f}")
    print(f"C  (weighted, w={spec.metric_weights}): {scores['C']:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
