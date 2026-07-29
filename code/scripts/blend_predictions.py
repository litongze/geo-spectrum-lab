#!/usr/bin/env python3
"""Blend two channel prediction .npy files for submission experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True, help="First prediction file.")
    parser.add_argument("--b", required=True, help="Second prediction file.")
    parser.add_argument(
        "--alpha",
        type=float,
        required=True,
        help="Output = alpha*A + (1-alpha)*B.",
    )
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    a = np.load(args.a)
    b = np.load(args.b)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    pred = (float(args.alpha) * a + (1.0 - float(args.alpha)) * b) * float(args.scale)
    if not np.isfinite(pred.real).all() or not np.isfinite(pred.imag).all():
        raise ValueError("blended prediction contains non-finite values")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, pred.astype(np.complex64))
    rms = float(np.sqrt(np.mean(np.abs(pred.astype(np.complex64)) ** 2)))
    print(
        f"[blend] wrote {out}; shape={pred.shape}, dtype=complex64, "
        f"alpha={args.alpha:g}, scale={args.scale:g}, rms={rms:.9g}"
    )


if __name__ == "__main__":
    main()
