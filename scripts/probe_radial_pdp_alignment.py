#!/usr/bin/env python3
"""Probe source-only peak alignment inside radial-neighbor PDP targets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from _bootstrap import ROOT  # noqa: F401
from score_holdout import reproduce_val_indices
from sweep_rayprofile_spectrum_knn import local_radial_profiles
from validate_moment_projection import stable_unit
from wireless_twin.data.map_loader import load_point_cloud
from wireless_twin.data.setup_config import load_setup
from wireless_twin.models.raytrace2 import build_heightmap
from wireless_twin.signal import pdp_spectrum


def parse_grid(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item]


def circular_delta(
    left: torch.Tensor, right: torch.Tensor, size: int
) -> torch.Tensor:
    return (
        torch.remainder(left - right + size // 2, size)
        - size // 2
    )


def align_to_nearest(
    values: torch.Tensor, max_shift: int
) -> torch.Tensor:
    peaks = values.argmax(dim=-1)
    shift = circular_delta(
        peaks[:, :1], peaks, values.shape[-1]
    )
    shift = torch.where(
        shift.abs() <= max_shift,
        shift,
        torch.zeros_like(shift),
    )
    delay = torch.arange(
        values.shape[-1], device=values.device
    )
    source = torch.remainder(
        delay[None, None, None, None, :]
        - shift[..., None],
        values.shape[-1],
    )
    return torch.gather(values, -1, source)


def radial_neighbors(
    query_idx: np.ndarray,
    pool_idx: np.ndarray,
    positions: np.ndarray,
    profiles: np.ndarray,
    profile_lambda: float,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    xy_delta = (
        positions[query_idx, None, :2]
        - positions[pool_idx][None, :, :2]
    )
    spatial_distance2 = np.square(xy_delta).sum(axis=-1)
    scale = profiles[pool_idx].std(axis=0).clip(0.05)
    profile_delta = (
        profiles[query_idx, None] - profiles[pool_idx][None]
    ) / scale
    effective_distance2 = (
        spatial_distance2
        + profile_lambda**2
        * np.square(profile_delta).mean(axis=-1)
    )
    local = np.argpartition(
        effective_distance2, kth=k - 1, axis=1
    )[:, :k]
    selected_effective = np.take_along_axis(
        effective_distance2, local, axis=1
    )
    order = np.argsort(selected_effective, axis=1)
    local = np.take_along_axis(local, order, axis=1)
    distance = np.sqrt(
        np.maximum(
            np.take_along_axis(spatial_distance2, local, axis=1),
            1e-6,
        )
    ).astype(np.float32)
    return pool_idx[local], distance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", default="Round1_Map(2)")
    parser.add_argument(
        "--cache-dir", default="cache/teammate_knn_hvp"
    )
    parser.add_argument(
        "--prediction-template",
        default=(
            "cache/dual_gate50_10_projection/"
            "split_s{seed}_prediction.npy"
        ),
    )
    parser.add_argument(
        "--tune-seeds", default="1890,3716,962,1022"
    )
    parser.add_argument("--audit-seed", default="2262")
    parser.add_argument("--profile-lambda", type=float, default=24.0)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--distance-power", type=float, default=0.5)
    parser.add_argument("--anchor-strength", type=float, default=4.0)
    parser.add_argument("--max-shift-grid", default="1,2")
    parser.add_argument(
        "--alignment-blend-grid", default="0,0.1,0.2,0.3,0.5"
    )
    parser.add_argument(
        "--target-beta-grid", default="0.2,0.25,0.3,0.35,0.4"
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out", default="docs/radial_pdp_alignment/result.json"
    )
    args = parser.parse_args()

    tune = [item for item in args.tune_seeds.split(",") if item]
    names = [*tune, args.audit_seed]
    max_shifts = [
        int(item) for item in args.max_shift_grid.split(",") if item
    ]
    alignment_blends = parse_grid(args.alignment_blend_grid)
    target_betas = parse_grid(args.target_beta_grid)
    configs = [
        (max_shift, alignment_blend, target_beta)
        for max_shift in max_shifts
        for alignment_blend in alignment_blends
        for target_beta in target_betas
    ]

    datadir = Path(args.datadir)
    spec = load_setup(datadir / "Round1_Setup.json")
    positions = np.load(
        datadir / "Round1_Train_Pos.npy"
    ).astype(np.float32)
    points = load_point_cloud(datadir / "Round1_Map.ply")
    heightmap, x0, y0, resolution = build_heightmap(points)
    profiles = local_radial_profiles(
        positions,
        np.asarray(spec.bs_position, dtype=np.float32),
        heightmap,
        x0,
        y0,
        resolution,
    )["radial_height"]
    del points, heightmap
    device = torch.device(args.device)
    cached = np.load(
        Path(args.cache_dir) / "train_pdp.npy", mmap_mode="r"
    )
    spectra = stable_unit(
        torch.as_tensor(
            np.array(cached, copy=True),
            dtype=torch.float32,
            device=device,
        )
    )
    del cached
    all_idx = np.arange(len(positions), dtype=np.int64)
    tune_union = {
        int(index)
        for name in tune
        for index in reproduce_val_indices(
            len(positions), 0.1, int(name)
        )
    }
    scores = {}
    strict_scores = {}

    with torch.inference_mode():
        for name in names:
            val_idx = np.asarray(
                sorted(
                    reproduce_val_indices(
                        len(positions), 0.1, int(name)
                    )
                ),
                dtype=np.int64,
            )
            pool_idx = np.setdiff1d(all_idx, val_idx)
            neighbors, distance = radial_neighbors(
                val_idx,
                pool_idx,
                positions,
                profiles,
                args.profile_lambda,
                args.k,
            )
            prediction = np.load(
                args.prediction_template.format(seed=name),
                mmap_mode="r",
            )
            totals = {
                config: 0.0 for config in configs
            }
            strict_totals = {
                config: 0.0 for config in configs
            }
            count = 0
            strict_count = 0
            strict_query = np.asarray(
                [
                    int(index) not in tune_union
                    for index in val_idx
                ],
                dtype=bool,
            )
            for start in range(0, len(val_idx), args.batch_size):
                stop = min(start + args.batch_size, len(val_idx))
                predicted = torch.as_tensor(
                    np.array(prediction[start:stop], copy=True),
                    dtype=torch.complex64,
                    device=device,
                )
                baseline = stable_unit(
                    pdp_spectrum(predicted, spec)
                )
                truth = spectra[
                    torch.as_tensor(
                        val_idx[start:stop],
                        dtype=torch.long,
                        device=device,
                    )
                ]
                neighbor_t = torch.as_tensor(
                    neighbors[start:stop],
                    dtype=torch.long,
                    device=device,
                )
                values = spectra[neighbor_t]
                nearest_global = (
                    values * values[:, :1]
                ).sum(dim=-1).mean(dim=(2, 3))
                distance_t = torch.as_tensor(
                    distance[start:stop],
                    dtype=torch.float32,
                    device=device,
                ).clamp_min(0.3)
                weight = torch.softmax(
                    -args.distance_power * distance_t.log()
                    + args.anchor_strength * nearest_global,
                    dim=1,
                )
                unaligned = stable_unit(
                    torch.einsum(
                        "bk,bkmns->bmns", weight, values
                    )
                )
                aligned_bank = {
                    max_shift: stable_unit(
                        torch.einsum(
                            "bk,bkmns->bmns",
                            weight,
                            align_to_nearest(values, max_shift),
                        )
                    )
                    for max_shift in max_shifts
                }
                keep = torch.as_tensor(
                    strict_query[start:stop],
                    dtype=torch.bool,
                    device=device,
                )
                for config in configs:
                    max_shift, alignment_blend, target_beta = config
                    selected = stable_unit(
                        (1.0 - alignment_blend) * unaligned
                        + alignment_blend * aligned_bank[max_shift]
                    )
                    output = stable_unit(
                        (1.0 - target_beta) * baseline
                        + target_beta * selected
                    )
                    cosine = (output * truth).sum(dim=-1)
                    totals[config] += float(cosine.sum())
                    if name == args.audit_seed and keep.any():
                        strict_totals[config] += float(
                            cosine[keep].sum()
                        )
                count += baseline.numel() // baseline.shape[-1]
                if name == args.audit_seed:
                    strict_count += (
                        int(keep.sum())
                        * int(np.prod(baseline.shape[1:-1]))
                    )
            scores[name] = {
                config: total / count
                for config, total in totals.items()
            }
            if name == args.audit_seed:
                strict_scores = {
                    config: total / strict_count
                    for config, total in strict_totals.items()
                }
            print(
                f"[radial-pdp-align] split={name} done",
                flush=True,
            )

    baseline_config = (max_shifts[0], 0.0, 0.35)
    rows = []
    for config in configs:
        tune_delta = {
            name: scores[name][config] - scores[name][baseline_config]
            for name in tune
        }
        rows.append(
            {
                "max_shift": config[0],
                "alignment_blend": config[1],
                "target_beta": config[2],
                "tune_delta": tune_delta,
                "tune_delta_mean": float(
                    np.mean(list(tune_delta.values()))
                ),
                "tune_delta_worst": float(
                    np.min(list(tune_delta.values()))
                ),
                "audit_delta": float(
                    scores[args.audit_seed][config]
                    - scores[args.audit_seed][baseline_config]
                ),
                "strict_audit_delta": float(
                    strict_scores[config]
                    - strict_scores[baseline_config]
                ),
                "scores": {
                    name: scores[name][config] for name in names
                },
            }
        )
    rows.sort(
        key=lambda row: (
            row["tune_delta_mean"],
            row["tune_delta_worst"],
        ),
        reverse=True,
    )
    robust = [
        row
        for row in rows
        if row["tune_delta_worst"] >= 0.0
        and row["audit_delta"] >= 0.0
        and row["strict_audit_delta"] >= 0.0
    ]
    payload = {
        "selection_policy": (
            "peak alignment and target strengths selected on four "
            "tune folds; audit and strict audit are diagnostics"
        ),
        "baseline": {
            "max_shift": baseline_config[0],
            "alignment_blend": baseline_config[1],
            "target_beta": baseline_config[2],
        },
        "best": rows[0],
        "best_robust": robust[0] if robust else None,
        "ranked": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    selected = robust[0] if robust else rows[0]
    print(
        "RADIAL_PDP_ALIGNMENT_DONE "
        f"shift={selected['max_shift']} "
        f"align={selected['alignment_blend']:g} "
        f"beta={selected['target_beta']:g} "
        f"tune={selected['tune_delta_mean']:+.6f} "
        f"worst={selected['tune_delta_worst']:+.6f} "
        f"audit={selected['audit_delta']:+.6f} "
        f"strict={selected['strict_audit_delta']:+.6f} "
        f"out={out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
