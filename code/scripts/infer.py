#!/usr/bin/env python3
"""Generate ``RoundX_Test_Channel.npy`` from a trained checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from wireless_twin.evaluation.predictor import (
    load_model_from_checkpoint,
    predict_test_channels,
    save_test_channels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--datadir", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datadir = Path(args.datadir)
    model, meta = load_model_from_checkpoint(args.ckpt, device=args.device)
    tag = meta["round_tag"]
    test_path = datadir / f"{tag}_Test_Pos.npy"
    positions = np.load(test_path).astype(np.float32)
    channels = predict_test_channels(
        model,
        positions,
        meta,
        device=args.device,
        batch_size=args.batch_size,
        precision=args.precision,
    )
    out = Path(args.out) if args.out else datadir / f"{tag}_Test_Channel.npy"
    save_test_channels(channels, out)
    print(f"[infer] wrote {out}; shape={channels.shape}, dtype={channels.dtype}")


if __name__ == "__main__":
    main()
