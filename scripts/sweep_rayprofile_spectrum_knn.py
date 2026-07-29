#!/usr/bin/env python3
"""Sweep full BS-to-UE ray-profile distances for spectrum neighbors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from _bootstrap import ROOT  # noqa: F401
from score_holdout import reproduce_val_indices
from wireless_twin.data.map_loader import load_point_cloud
from wireless_twin.data.setup_config import load_setup
from wireless_twin.models.raytrace2 import build_heightmap


def stable_unit(value: torch.Tensor) -> torch.Tensor:
    maximum = value.amax(dim=-1, keepdim=True)
    scaled = torch.where(
        maximum > 0,
        value / maximum.clamp_min(torch.finfo(value.dtype).tiny),
        torch.zeros_like(value),
    )
    return scaled / scaled.norm(dim=-1, keepdim=True).clamp_min(
        torch.finfo(value.dtype).tiny
    )


def ray_profiles(
    positions: np.ndarray,
    bs_position: np.ndarray,
    heightmap: np.ndarray,
    x0: float,
    y0: float,
    resolution: float,
    samples: int,
) -> dict[str, np.ndarray]:
    fraction = np.linspace(0.01, 0.99, samples, dtype=np.float32)
    direction = positions - bs_position
    points = (
        bs_position[None, None]
        + direction[:, None] * fraction[None, :, None]
    )
    gx = np.clip(
        ((points[..., 0] - x0) / resolution).astype(np.int64),
        0,
        heightmap.shape[0] - 1,
    )
    gy = np.clip(
        ((points[..., 1] - y0) / resolution).astype(np.int64),
        0,
        heightmap.shape[1] - 1,
    )
    sampled_height = heightmap[gx, gy].astype(np.float32)
    excess = sampled_height - points[..., 2]
    positive = np.maximum(excess, 0.0)
    positive_scale = np.quantile(
        positive[positive > 0], 0.75
    ) if np.any(positive > 0) else 1.0
    height_scale = np.quantile(
        sampled_height[sampled_height > 0], 0.75
    ) if np.any(sampled_height > 0) else 1.0
    blocked = (excess > 0.5).astype(np.float32)
    positive = np.clip(positive / max(positive_scale, 1e-3), 0, 4)
    height = np.clip(
        sampled_height / max(height_scale, 1e-3), 0, 4
    )
    return {
        "blocked": blocked,
        "positive": positive.astype(np.float32),
        "height": height.astype(np.float32),
        "combined": np.concatenate(
            [blocked, positive, height], axis=1
        ).astype(np.float32),
    }


def local_radial_profiles(
    positions: np.ndarray,
    bs_position: np.ndarray,
    heightmap: np.ndarray,
    x0: float,
    y0: float,
    resolution: float,
    directions: int = 16,
    radius: int = 32,
) -> dict[str, np.ndarray]:
    base_angle = np.arctan2(
        positions[:, 1] - bs_position[1],
        positions[:, 0] - bs_position[0],
    )
    angle = base_angle[:, None] + np.linspace(
        0.0, 2.0 * np.pi, directions, endpoint=False
    )[None]
    distance = np.arange(1, radius + 1, dtype=np.float32)
    px = (
        positions[:, None, None, 0]
        + np.cos(angle)[:, :, None] * distance[None, None]
    )
    py = (
        positions[:, None, None, 1]
        + np.sin(angle)[:, :, None] * distance[None, None]
    )
    gx = np.clip(
        ((px - x0) / resolution).astype(np.int64),
        0,
        heightmap.shape[0] - 1,
    )
    gy = np.clip(
        ((py - y0) / resolution).astype(np.int64),
        0,
        heightmap.shape[1] - 1,
    )
    sampled_height = heightmap[gx, gy].astype(np.float32)
    sampled_occupied = sampled_height > 2.0
    center_gx = np.clip(
        ((positions[:, 0] - x0) / resolution).astype(np.int64),
        0,
        heightmap.shape[0] - 1,
    )
    center_gy = np.clip(
        ((positions[:, 1] - y0) / resolution).astype(np.int64),
        0,
        heightmap.shape[1] - 1,
    )
    center_occupied = (
        heightmap[center_gx, center_gy] > 2.0
    )[:, None, None]
    changed = sampled_occupied != center_occupied
    any_changed = changed.any(axis=-1)
    first_change = np.where(
        any_changed, np.argmax(changed, axis=-1) + 1, radius + 1
    ).astype(np.float32)
    boundary = first_change / (radius + 1)
    occupancy = sampled_occupied.mean(axis=-1).astype(np.float32)
    height_scale = np.quantile(
        sampled_height[sampled_height > 0], 0.75
    ) if np.any(sampled_height > 0) else 1.0
    maximum_height = np.clip(
        sampled_height.max(axis=-1) / max(height_scale, 1e-3),
        0,
        4,
    ).astype(np.float32)
    return {
        "radial_boundary": boundary,
        "radial_occupancy": occupancy,
        "radial_height": maximum_height,
        "radial_combined": np.concatenate(
            [boundary, occupancy, maximum_height], axis=1
        ).astype(np.float32),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", default="Round1_Map(2)")
    parser.add_argument("--cache-dir", default="cache/teammate_knn_hvp")
    parser.add_argument("--tune-seeds", default="1890,3716,962,1022")
    parser.add_argument("--audit-seed", type=int, default=2262)
    parser.add_argument(
        "--testmatched",
        default="handoff_to_teammate_20260727/splits/"
        "test_matched_seed2026_val.npy",
    )
    parser.add_argument(
        "--profile-modes", default="blocked,positive,height,combined"
    )
    parser.add_argument("--lambda-grid", default="0.5,1,2,4,8,16,32")
    parser.add_argument("--ray-samples", type=int, default=128)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--distance-power", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out", default="docs/rayprofile_spectrum_knn/result.json"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    datadir = Path(args.datadir)
    spec = load_setup(datadir / "Round1_Setup.json")
    positions = np.load(
        datadir / "Round1_Train_Pos.npy"
    ).astype(np.float32)
    points = load_point_cloud(datadir / "Round1_Map.ply")
    heightmap, x0, y0, resolution = build_heightmap(points)
    profiles = ray_profiles(
        positions,
        np.asarray(spec.bs_position, dtype=np.float32),
        heightmap,
        x0,
        y0,
        resolution,
        args.ray_samples,
    )
    profiles.update(
        local_radial_profiles(
            positions,
            np.asarray(spec.bs_position, dtype=np.float32),
            heightmap,
            x0,
            y0,
            resolution,
        )
    )
    requested_modes = [
        value.strip()
        for value in args.profile_modes.split(",")
        if value.strip()
    ]
    unknown = set(requested_modes) - set(profiles)
    if unknown:
        raise ValueError(f"unknown profile modes: {sorted(unknown)}")
    lambdas = [
        float(value) for value in args.lambda_grid.split(",") if value
    ]
    configs = [{"mode": "none", "lambda": 0.0}]
    configs.extend(
        {"mode": mode, "lambda": metric_lambda}
        for mode in requested_modes
        for metric_lambda in lambdas
    )

    cache_dir = Path(args.cache_dir)
    pas = stable_unit(
        torch.as_tensor(
            np.array(
                np.load(
                    cache_dir / "train_pas_phv.npy", mmap_mode="r"
                ),
                copy=True,
            ),
            dtype=torch.float32,
            device=device,
        )
    )
    pdp = stable_unit(
        torch.as_tensor(
            np.array(
                np.load(cache_dir / "train_pdp.npy", mmap_mode="r"),
                copy=True,
            ),
            dtype=torch.float32,
            device=device,
        )
    )

    tune_seeds = [
        int(value) for value in args.tune_seeds.split(",") if value
    ]
    split_indices = {
        str(seed): np.asarray(
            sorted(reproduce_val_indices(len(positions), 0.1, seed)),
            dtype=np.int64,
        )
        for seed in tune_seeds + [args.audit_seed]
    }
    split_indices["testmatched"] = np.load(args.testmatched).astype(
        np.int64
    )
    all_indices = np.arange(len(positions), dtype=np.int64)
    scores = {
        domain: {index: {} for index in range(len(configs))}
        for domain in ("pas", "pdp")
    }

    def score(
        spectra: torch.Tensor,
        truth_idx: np.ndarray,
        neighbors: np.ndarray,
        distance: np.ndarray,
    ) -> float:
        total = 0.0
        count = 0
        truth_t = torch.as_tensor(
            truth_idx, dtype=torch.long, device=device
        )
        for start in range(0, len(truth_idx), args.batch_size):
            stop = min(start + args.batch_size, len(truth_idx))
            index = torch.as_tensor(
                neighbors[start:stop],
                dtype=torch.long,
                device=device,
            )
            weight = torch.as_tensor(
                distance[start:stop],
                dtype=torch.float32,
                device=device,
            ).clamp_min(1e-3).pow(-args.distance_power)
            weight = weight / weight.sum(dim=1, keepdim=True)
            prediction = stable_unit(
                torch.einsum(
                    "bk,bk...->b...", weight, spectra[index]
                )
            )
            truth = spectra[truth_t[start:stop]]
            cosine = (prediction * truth).sum(dim=-1)
            total += float(cosine.sum())
            count += cosine.numel()
        return total / count

    for split_name, val_idx in split_indices.items():
        pool_idx = np.setdiff1d(all_indices, val_idx)
        xy_delta = (
            positions[val_idx, None, :2]
            - positions[pool_idx][None, :, :2]
        )
        xy_distance2 = np.square(xy_delta).sum(axis=-1)
        profile_distance = {}
        for mode in requested_modes:
            delta = (
                profiles[mode][val_idx, None]
                - profiles[mode][pool_idx][None]
            )
            profile_distance[mode] = np.square(delta).mean(
                axis=-1
            ).astype(np.float32)
        for config_index, config in enumerate(configs):
            distance2 = xy_distance2.copy()
            if config["mode"] != "none":
                distance2 += (
                    config["lambda"] ** 2
                    * profile_distance[config["mode"]]
                )
            local = np.argpartition(
                distance2, kth=args.k - 1, axis=1
            )[:, : args.k]
            selected = np.take_along_axis(distance2, local, axis=1)
            order = np.argsort(selected, axis=1)
            local = np.take_along_axis(local, order, axis=1)
            selected = np.take_along_axis(selected, order, axis=1)
            neighbors = pool_idx[local]
            distance = np.sqrt(np.maximum(selected, 1e-6))
            scores["pas"][config_index][split_name] = score(
                pas, val_idx, neighbors, distance
            )
            scores["pdp"][config_index][split_name] = score(
                pdp, val_idx, neighbors, distance
            )
        print(f"[ray-profile] split={split_name} done", flush=True)

    def ranked(domain: str) -> list[dict[str, object]]:
        rows = []
        for config_index, config in enumerate(configs):
            split_score = scores[domain][config_index]
            tune = [split_score[str(seed)] for seed in tune_seeds]
            rows.append(
                {
                    **config,
                    "tune_median": float(np.median(tune)),
                    "tune_mean": float(np.mean(tune)),
                    "tune_worst": float(np.min(tune)),
                    "audit": split_score[str(args.audit_seed)],
                    "testmatched": split_score["testmatched"],
                    "scores": split_score,
                }
            )
        return sorted(
            rows,
            key=lambda row: (
                row["tune_median"],
                row["tune_mean"],
                row["tune_worst"],
            ),
            reverse=True,
        )

    result = {
        "selection_policy": "tune splits only",
        "tune_seeds": tune_seeds,
        "audit_seed": args.audit_seed,
        "k": args.k,
        "distance_power": args.distance_power,
        "ray_samples": args.ray_samples,
        "pas_ranked": ranked("pas"),
        "pdp_ranked": ranked("pdp"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("[PAS TOP]", json.dumps(result["pas_ranked"][:8]), flush=True)
    print("[PDP TOP]", json.dumps(result["pdp_ranked"][:8]), flush=True)
    print(f"RAY_PROFILE_SWEEP_DONE out={out}", flush=True)


if __name__ == "__main__":
    main()
