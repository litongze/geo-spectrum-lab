#!/usr/bin/env python3
"""Use a clean predicted PAS as a label-free selector for PDP neighbors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from _bootstrap import ROOT  # noqa: F401
from score_holdout import reproduce_val_indices
from wireless_twin.data.setup_config import load_setup
from wireless_twin.signal import (
    pas_spectrum,
    pas_spectrum_phv,
    pas_spectrum_pvh,
    pdp_spectrum,
)


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
    parser.add_argument("--baseline-val", required=True)
    parser.add_argument(
        "--pas-layout", choices=("hvp", "pvh", "phv"), default="phv"
    )
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--distance-power", type=float, default=2.0)
    parser.add_argument("--gamma-grid", default="0,2,5,10,20,40,80")
    parser.add_argument("--blend-grid", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out", default="docs/crossdomain_pdp_s1890/result.json"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    datadir = Path(args.datadir)
    spec = load_setup(datadir / "Round1_Setup.json")
    positions = np.load(datadir / "Round1_Train_Pos.npy").astype(np.float32)
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
    cache_dir = Path(args.cache_dir)
    all_pas = stable_unit(
        torch.as_tensor(
            np.array(
                np.load(
                    cache_dir / f"train_pas_{args.pas_layout}.npy",
                    mmap_mode="r",
                ),
                copy=True,
            ),
            dtype=torch.float32,
            device=device,
        )
    )
    all_pdp = stable_unit(
        torch.as_tensor(
            np.array(
                np.load(cache_dir / "train_pdp.npy", mmap_mode="r"),
                copy=True,
            ),
            dtype=torch.float32,
            device=device,
        )
    )
    baseline_channel = torch.as_tensor(
        np.array(np.load(args.baseline_val, mmap_mode="r"), copy=True),
        dtype=torch.complex64,
        device=device,
    )
    transform = {
        "hvp": pas_spectrum,
        "pvh": pas_spectrum_pvh,
        "phv": pas_spectrum_phv,
    }[args.pas_layout]
    query_pas = stable_unit(transform(baseline_channel, spec))
    baseline_pdp = stable_unit(pdp_spectrum(baseline_channel, spec))
    del baseline_channel
    truth = all_pdp[
        torch.as_tensor(val_idx, dtype=torch.long, device=device)
    ]
    gammas = [float(value) for value in args.gamma_grid.split(",")]
    blends = [float(value) for value in args.blend_grid.split(",")]
    totals = torch.zeros(
        len(gammas), len(blends), device=device
    )
    agreement_summary = []
    count = 0
    for start in range(0, len(val_idx), args.batch_size):
        stop = min(start + args.batch_size, len(val_idx))
        neighbor_t = torch.as_tensor(
            neighbors[start:stop], dtype=torch.long, device=device
        )
        neighbor_pas = all_pas[neighbor_t]
        agreement = (
            neighbor_pas * query_pas[start:stop, None]
        ).sum(dim=-1).mean(dim=-1)
        agreement_summary.append(agreement.detach().cpu())
        log_distance = torch.as_tensor(
            np.log(np.maximum(distance[start:stop], 0.3)),
            dtype=torch.float32,
            device=device,
        )
        neighbor_pdp = all_pdp[neighbor_t]
        target = truth[start:stop]
        for gamma_index, gamma in enumerate(gammas):
            weight = torch.softmax(
                -args.distance_power * log_distance[:, :, None]
                + gamma * agreement,
                dim=1,
            )
            selected = stable_unit(
                torch.einsum(
                    "bkn,bkmns->bmns", weight, neighbor_pdp
                )
            )
            for blend_index, beta in enumerate(blends):
                prediction = stable_unit(
                    (1.0 - beta) * baseline_pdp[start:stop]
                    + beta * selected
                )
                totals[gamma_index, blend_index] += (
                    prediction * target
                ).sum()
        count += target.shape[0] * target.shape[1] * target.shape[2]
    scores = totals / count
    rows = []
    for gamma_index, gamma in enumerate(gammas):
        for blend_index, beta in enumerate(blends):
            rows.append(
                {
                    "gamma": gamma,
                    "blend_beta": beta,
                    "score": float(scores[gamma_index, blend_index]),
                }
            )
    rows.sort(key=lambda row: row["score"], reverse=True)
    agreement_all = torch.cat(agreement_summary)
    result = {
        "split_seed": args.split_seed,
        "pas_layout": args.pas_layout,
        "k": args.k,
        "distance_power": args.distance_power,
        "baseline_score": next(
            row["score"]
            for row in rows
            if row["blend_beta"] == 0.0
        ),
        "best": rows[0],
        "rows": rows,
        "agreement_mean": float(agreement_all.mean()),
        "agreement_std": float(agreement_all.std()),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"CROSSDOMAIN_PDP_DONE out={out}", flush=True)


if __name__ == "__main__":
    main()
