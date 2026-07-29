#!/usr/bin/env python3
"""Offline PAS/PDP/NMSE scoring for a prediction file."""

from __future__ import annotations

import argparse
import numpy as np

import _bootstrap  # noqa: F401
from wireless_twin.data.setup_config import load_setup
from wireless_twin.evaluation.metrics import evaluate_channels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--setup", required=True)
    args = parser.parse_args()
    scores = evaluate_channels(
        np.load(args.pred), np.load(args.gt), load_setup(args.setup))
    for key, value in scores.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
