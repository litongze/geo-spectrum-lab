#!/usr/bin/env python3
"""Experimentally compare checkpoints under an eps-cosine proxy of the official
grader, each at ITS OWN optimal submission scale.

For every checkpoint and every eps we scan a range of global submission scales
and report the best achievable score + the scale that achieves it.  This is the
honest test of whether magnitude-calibrated models (native scale) beat the
uniform-large-scale baseline under the official's (small, unknown) eps.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import _bootstrap  # noqa: F401

from wireless_twin.data.setup_config import load_setup
from wireless_twin.data.channel_dataset import detect_round
from wireless_twin.evaluation.predictor import (
    load_model_from_checkpoint, predict_test_channels)
from score_holdout import reproduce_val_indices

SCALES = [3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
EPS = [1e-8, 1e-10, 1e-12]


def transforms(h, mh, mv, mp, n, s):
    a = h.reshape(-1, mh, mv, mp, n, s)
    pas = torch.fft.fft2(a, dim=(1, 2), norm="ortho").abs().square().sum(3)
    pas = pas.reshape(-1, mh * mv, n, s)
    pdp = torch.fft.ifft(h, dim=-1, norm="ortho").abs().square()
    return pas, pdp


def score(pred, gt, spec, eps):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    P = torch.tensor(pred, dtype=torch.complex64, device=dev)
    G = torch.tensor(gt, dtype=torch.complex64, device=dev)
    pp, pd = transforms(P, spec.mh, spec.mv, spec.mp, spec.n, spec.s)
    gp, gd = transforms(G, spec.mh, spec.mv, spec.mp, spec.n, spec.s)
    c1 = F.cosine_similarity(pp, gp, dim=1, eps=eps).mean()
    c2 = F.cosine_similarity(pd, gd, dim=-1, eps=eps).mean()
    nm = (P - G).abs().square().sum() / G.abs().square().sum()
    w = spec.metric_weights
    return float((w[0]*c1 + w[1]*c2 + w[2]/(1+nm)) / sum(w))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    d = Path(args.datadir); tag = detect_round(d)
    spec = load_setup(d / f"{tag}_Setup.json")
    pos = np.load(d / f"{tag}_Train_Pos.npy").astype(np.float32)
    ch = np.load(d / f"{tag}_Train_Channel.npy")
    vi = reproduce_val_indices(len(pos), 0.1, 0)
    vp, vg = pos[vi], ch[vi]

    print(f"{'model':22} " + " ".join(f"eps={e:.0e}(best@scale)" for e in EPS))
    for c in args.ckpts:
        model, meta = load_model_from_checkpoint(c, device=args.device)
        pred = predict_test_channels(model, vp, meta, device=args.device)
        pred = pred / np.sqrt(np.mean(np.abs(pred) ** 2))     # RMS=1 base
        row = [Path(c).stem[:22].ljust(22)]
        for eps in EPS:
            best, bscale = -1, None
            for sc in SCALES:
                v = score(pred * sc, vg, spec, eps)
                if v > best:
                    best, bscale = v, sc
            row.append(f"{best:.4f}@{bscale:.0e}   ")
        print(" ".join(row))


if __name__ == "__main__":
    main()
