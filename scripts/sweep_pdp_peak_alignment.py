#!/usr/bin/env python3
"""Sweep label-free local peak alignment before PDP neighbor interpolation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from _bootstrap import ROOT  # noqa: F401
from score_holdout import reproduce_val_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", default="Round1_Map(2)")
    parser.add_argument(
        "--cache", default="cache/teammate_knn_hvp/train_pdp.npy"
    )
    parser.add_argument("--tune-seeds", default="1890,3716,962,1022")
    parser.add_argument("--audit-seed", type=int, default=2262)
    parser.add_argument(
        "--testmatched",
        default="handoff_to_teammate_20260727/splits/"
        "test_matched_seed2026_val.npy",
    )
    parser.add_argument("--k-grid", default="3,8,16")
    parser.add_argument("--max-shift-grid", default="1,2,3")
    parser.add_argument("--beta-grid", default="0.25,0.5,0.75,1")
    parser.add_argument("--distance-power", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out", default="docs/pdp_peak_alignment/result.json"
    )
    return parser.parse_args()


def stable_unit(value: torch.Tensor) -> torch.Tensor:
    scale = value.amax(dim=-1, keepdim=True)
    scaled = torch.where(
        scale > 0,
        value / scale.clamp_min(torch.finfo(value.dtype).tiny),
        torch.zeros_like(value),
    )
    return scaled / scaled.norm(dim=-1, keepdim=True).clamp_min(
        torch.finfo(value.dtype).tiny
    )


def circular_delta(left: torch.Tensor, right: torch.Tensor, size: int) -> torch.Tensor:
    return torch.remainder(left - right + size // 2, size) - size // 2


def align_to_reference(
    spectra: torch.Tensor,
    peaks: torch.Tensor,
    reference: torch.Tensor,
    max_shift: int,
) -> torch.Tensor:
    shift = circular_delta(reference[:, None], peaks, spectra.shape[-1])
    shift = torch.where(
        shift.abs() <= max_shift, shift, torch.zeros_like(shift)
    )
    delay = torch.arange(spectra.shape[-1], device=spectra.device)
    source = torch.remainder(
        delay[None, None, None, :] - shift[..., None],
        spectra.shape[-1],
    )
    return torch.gather(spectra, -1, source)


def consensus_reference(
    peaks: torch.Tensor,
    weights: torch.Tensor,
    tolerance: int,
    size: int,
) -> torch.Tensor:
    candidate_delta = circular_delta(
        peaks[:, :, None], peaks[:, None, :], size
    )
    support = (
        (candidate_delta.abs() <= tolerance).float()
        * weights[:, None, :, None]
    ).sum(dim=2)
    selected = support.argmax(dim=1, keepdim=True)
    return torch.gather(peaks, 1, selected).squeeze(1)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    tune_seeds = [
        int(value) for value in args.tune_seeds.split(",") if value.strip()
    ]
    k_grid = [int(value) for value in args.k_grid.split(",") if value.strip()]
    max_shift_grid = [
        int(value)
        for value in args.max_shift_grid.split(",")
        if value.strip()
    ]
    beta_grid = [
        float(value) for value in args.beta_grid.split(",") if value.strip()
    ]
    configs = [
        {
            "name": f"k{k}_{reference}_s{max_shift}_b{beta:g}",
            "k": k,
            "reference": reference,
            "max_shift": max_shift,
            "beta": beta,
        }
        for k in k_grid
        for reference in ("first", "consensus")
        for max_shift in max_shift_grid
        for beta in beta_grid
    ]
    datadir = Path(args.datadir)
    positions = np.load(datadir / "Round1_Train_Pos.npy").astype(np.float32)
    spectra = np.load(args.cache, mmap_mode="r")
    all_idx = np.arange(len(positions), dtype=np.int64)
    splits = {
        str(seed): np.asarray(
            sorted(reproduce_val_indices(len(positions), 0.1, seed)),
            dtype=np.int64,
        )
        for seed in tune_seeds + [args.audit_seed]
    }
    splits["testmatched"] = np.load(args.testmatched).astype(np.int64)
    split_scores: dict[str, list[float]] = {}

    for split_name, val_idx in splits.items():
        pool_idx = np.setdiff1d(all_idx, val_idx, assume_unique=False)
        distance, local = cKDTree(positions[pool_idx, :2]).query(
            positions[val_idx, :2], k=max(k_grid)
        )
        neighbors = pool_idx[local]
        totals = torch.zeros(len(configs), device=device)
        count = 0
        for start in range(0, len(val_idx), args.batch_size):
            stop = min(start + args.batch_size, len(val_idx))
            truth = stable_unit(
                torch.as_tensor(
                    np.array(spectra[val_idx[start:stop]], copy=True),
                    device=device,
                ).reshape(stop - start, -1, spectra.shape[-1])
            )
            neighbor = stable_unit(
                torch.as_tensor(
                    np.array(spectra[neighbors[start:stop]], copy=True),
                    device=device,
                ).reshape(
                    stop - start,
                    max(k_grid),
                    -1,
                    spectra.shape[-1],
                )
            )
            batch_distance = torch.as_tensor(
                distance[start:stop], dtype=torch.float32, device=device
            ).clamp_min(1e-3)
            peaks = neighbor.argmax(dim=-1)
            cache: dict[tuple[int, str, int], tuple[torch.Tensor, torch.Tensor]] = {}
            for config_idx, config in enumerate(configs):
                k = config["k"]
                weights = batch_distance[:, :k].pow(-args.distance_power)
                weights = weights / weights.sum(dim=1, keepdim=True)
                baseline = (
                    weights[:, :, None, None] * neighbor[:, :k]
                ).sum(dim=1)
                key = (k, config["reference"], config["max_shift"])
                if key not in cache:
                    if config["reference"] == "first":
                        reference = peaks[:, 0]
                    else:
                        reference = consensus_reference(
                            peaks[:, :k],
                            weights,
                            tolerance=1,
                            size=spectra.shape[-1],
                        )
                    aligned = align_to_reference(
                        neighbor[:, :k],
                        peaks[:, :k],
                        reference,
                        config["max_shift"],
                    )
                    aligned = (
                        weights[:, :, None, None] * aligned
                    ).sum(dim=1)
                    cache[key] = baseline, aligned
                baseline, aligned = cache[key]
                prediction = stable_unit(
                    (1.0 - config["beta"]) * baseline
                    + config["beta"] * aligned
                )
                totals[config_idx] += (prediction * truth).sum()
            count += truth.numel() // spectra.shape[-1]
        values = (totals / count).cpu().numpy().tolist()
        split_scores[split_name] = [float(value) for value in values]
        print(
            f"[pdp-align] split={split_name} best={max(values):.6f}",
            flush=True,
        )

    ranked = []
    for config_idx, config in enumerate(configs):
        tune = [
            split_scores[str(seed)][config_idx] for seed in tune_seeds
        ]
        ranked.append(
            {
                **config,
                "tune_median": float(np.median(tune)),
                "tune_mean": float(np.mean(tune)),
                "tune_worst": float(np.min(tune)),
                "audit": split_scores[str(args.audit_seed)][config_idx],
                "testmatched": split_scores["testmatched"][config_idx],
                "scores": {
                    name: values[config_idx]
                    for name, values in split_scores.items()
                },
            }
        )
    ranked.sort(
        key=lambda row: (
            row["tune_median"],
            row["tune_mean"],
            row["tune_worst"],
        ),
        reverse=True,
    )
    payload = {
        "selection_policy": "ranked only on tune seeds",
        "tune_seeds": tune_seeds,
        "audit_seed": args.audit_seed,
        "ranked": ranked,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("[PDP TOP]", json.dumps(ranked[:8]), flush=True)
    print(f"PDP_ALIGNMENT_DONE out={out}", flush=True)


if __name__ == "__main__":
    main()
