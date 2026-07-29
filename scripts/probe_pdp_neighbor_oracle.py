#!/usr/bin/env python3
"""Measure clean PDP ceilings from spatial-neighbor selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from _bootstrap import ROOT  # noqa: F401
from score_holdout import reproduce_val_indices


def stable_unit(value: torch.Tensor) -> torch.Tensor:
    maximum = value.amax(dim=-1, keepdim=True)
    scaled = value / maximum.clamp_min(torch.finfo(value.dtype).tiny)
    norm = scaled.norm(dim=-1, keepdim=True)
    return torch.where(norm > 0, scaled / norm.clamp_min(1e-30), 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", default="Round1_Map(2)")
    parser.add_argument("--cache-dir", default="cache/teammate_knn_hvp")
    parser.add_argument("--split-seed", type=int, default=1890)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out", default="docs/pdp_neighbor_oracle_s1890/result.json"
    )
    args = parser.parse_args()

    positions = np.load(
        Path(args.datadir) / "Round1_Train_Pos.npy"
    ).astype(np.float32)
    val_idx = np.asarray(
        sorted(
            reproduce_val_indices(
                len(positions), 0.1, args.split_seed
            )
        ),
        dtype=np.int64,
    )
    pool_idx = np.setdiff1d(np.arange(len(positions)), val_idx)
    distance, local = cKDTree(positions[pool_idx, :2]).query(
        positions[val_idx, :2], k=args.k
    )
    neighbors = pool_idx[local]
    device = torch.device(args.device)
    spectra = stable_unit(
        torch.as_tensor(
            np.array(
                np.load(
                    Path(args.cache_dir) / "train_pdp.npy", mmap_mode="r"
                ),
                copy=True,
            ),
            dtype=torch.float32,
            device=device,
        )
    ).reshape(len(positions), -1, 192)
    truth = spectra[
        torch.as_tensor(val_idx, dtype=torch.long, device=device)
    ]
    neighbor_t = torch.as_tensor(
        neighbors, dtype=torch.long, device=device
    )
    distance_t = torch.as_tensor(distance, device=device).clamp_min(0.3)
    rows = []
    for k in (1, 2, 4, 8, 16, args.k):
        values = spectra[neighbor_t[:, :k]]
        cosine = (values * truth[:, None]).sum(-1)
        weight = distance_t[:, :k].pow(-2)
        weight /= weight.sum(dim=1, keepdim=True)
        idw = stable_unit(
            (weight[:, :, None, None] * values).sum(dim=1)
        )
        query_choice = cosine.mean(dim=-1).amax(dim=1).mean()
        rows.append(
            {
                "k": k,
                "idw": float((idw * truth).sum(-1).mean()),
                "nearest_oracle_per_slice": float(
                    cosine.amax(dim=1).mean()
                ),
                "nearest_oracle_per_query": float(query_choice),
            }
        )
    result = {
        "split_seed": args.split_seed,
        "rows": rows,
        "invalid_truth_fraction": float(
            (truth.norm(dim=-1) == 0).float().mean()
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"PDP_NEIGHBOR_ORACLE_DONE out={out}", flush=True)


if __name__ == "__main__":
    main()
