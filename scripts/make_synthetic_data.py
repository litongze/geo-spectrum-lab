#!/usr/bin/env python3
"""Generate a tiny synthetic dataset in the exact competition file format.

Handy for wiring up / smoke-testing the pipeline before the real ``DataX`` is
released.  Channels are a smooth, position-dependent superposition of a few
"paths", so a model can actually learn the mapping.  Dimensions default to small
values so everything runs in seconds on CPU (Windows-friendly).

Usage
-----
    python scripts/make_synthetic_data.py --out data/DataSynth --round Round1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401


def build_signatures(rng, k, m, n, s):
    def cplx(shape):
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(shape[-1])
    return cplx((k, m)), cplx((k, n)), cplx((k, s))


def channels_for(positions, u, v, w, k, freqs):
    """H[p,m,n,s] = sum_k gain_k(pos_p) * u[k,m]*v[k,n]*w[k,s]."""
    # smooth complex gains as functions of position (Fourier of pos)
    phases = positions @ freqs.T                       # (P, k)
    gains = np.exp(1j * phases) * (0.5 + 0.5 * np.cos(phases))
    return np.einsum("pk,km,kn,ks->pmns", gains, u, v, w).astype(np.complex64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/DataSynth")
    ap.add_argument("--round", default="Round1")
    ap.add_argument("--p-train", type=int, default=256)
    ap.add_argument("--p-test", type=int, default=64)
    ap.add_argument("--mh", type=int, default=2)
    ap.add_argument("--mv", type=int, default=2)
    ap.add_argument("--mp", type=int, default=2)
    ap.add_argument("--nh", type=int, default=1)
    ap.add_argument("--nv", type=int, default=1)
    ap.add_argument("--np", type=int, default=2)
    ap.add_argument("--s", type=int, default=16)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = args.round

    m = args.mh * args.mv * args.mp
    n = args.nh * args.nv * args.np

    setup = {
        "P_Train": args.p_train, "P_Test": args.p_test,
        "M": m, "MH": args.mh, "MV": args.mv, "MP": args.mp,
        "N": n, "NH": args.nh, "NV": args.nv, "NP": args.np,
        "S": args.s, "Q": 2,
        "X": [50.0, 0.0, 25.0],
        "w": [0.4, 0.4, 0.2],
    }
    (out / f"{tag}_Setup.json").write_text(json.dumps(setup, indent=2))

    # positions in a 40 x 40 x 5 m volume
    train_pos = rng.uniform([-20, -20, 0], [20, 20, 5], size=(args.p_train, 3)).astype(np.float32)
    test_pos = rng.uniform([-20, -20, 0], [20, 20, 5], size=(args.p_test, 3)).astype(np.float32)

    u, v, w = build_signatures(rng, args.k, m, n, args.s)
    freqs = rng.standard_normal((args.k, 3)) * 0.15    # spatial frequencies

    train_ch = channels_for(train_pos, u, v, w, args.k, freqs)
    test_ch = channels_for(test_pos, u, v, w, args.k, freqs)   # held-out GT

    np.save(out / f"{tag}_Train_Pos.npy", train_pos)
    np.save(out / f"{tag}_Train_Channel.npy", train_ch)
    np.save(out / f"{tag}_Test_Pos.npy", test_pos)
    # ground truth for local scoring only (never provided in the real contest)
    np.save(out / f"{tag}_Test_Channel_GT.npy", test_ch)

    # a trivial point-cloud map so map_loader has something to read
    _write_tiny_ply(out / f"{tag}_Map.ply", rng.uniform(-20, 20, size=(200, 3)))

    print(f"[synth] wrote {tag} dataset to {out}")
    print(f"[synth] M={m} N={n} S={args.s} | train={args.p_train} test={args.p_test}")


def _write_tiny_ply(path: Path, pts: np.ndarray) -> None:
    header = ("ply\nformat ascii 1.0\n"
              f"element vertex {len(pts)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "end_header\n")
    with path.open("w", encoding="utf-8") as f:
        f.write(header)
        for p in pts:
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")


if __name__ == "__main__":
    main()
