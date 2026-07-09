#!/usr/bin/env python3
"""Generate the submission file ``RoundX_Test_Channel.npy`` from a checkpoint.

Usage
-----
    python scripts/infer.py --ckpt checkpoints/round1_path_field.pt \
        --datadir data/Data1
    # -> writes data/Data1/Round1_Test_Channel.npy  (complex, P_test x M x N x S)
"""

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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True, help="trained checkpoint (.pt)")
    p.add_argument("--datadir", required=True,
                   help="folder containing RoundX_Test_Pos.npy")
    p.add_argument("--out", default=None,
                   help="output path (default: <datadir>/<Round>_Test_Channel.npy)")
    p.add_argument("--device", default=None, help="cuda | cpu | mps")
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    datadir = Path(args.datadir)

    model, meta = load_model_from_checkpoint(args.ckpt, device=args.device)
    tag = meta["round_tag"]

    test_pos_path = datadir / f"{tag}_Test_Pos.npy"
    if not test_pos_path.exists():
        raise SystemExit(f"test positions not found: {test_pos_path}")
    positions = np.load(test_pos_path).astype(np.float32)
    print(f"[infer] {tag}: {len(positions)} test positions")

    channels = predict_test_channels(
        model, positions, meta, device=args.device, batch_size=args.batch_size)

    out = Path(args.out) if args.out else datadir / f"{tag}_Test_Channel.npy"
    save_test_channels(channels, out)
    print(f"[infer] wrote {out}  shape={channels.shape} dtype={channels.dtype}")


if __name__ == "__main__":
    main()
