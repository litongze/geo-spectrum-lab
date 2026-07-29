#!/usr/bin/env python3
"""Sweep map-aware neighbor metrics for PHV PAS and PDP interpolation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from _bootstrap import ROOT  # noqa: F401
from score_holdout import reproduce_val_indices
from wireless_twin.data.map_loader import load_point_cloud
from wireless_twin.data.setup_config import load_setup
from wireless_twin.models.raytrace2 import build_heightmap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", default="Round1_Map(2)")
    parser.add_argument("--cache-dir", default="cache/teammate_knn_hvp")
    parser.add_argument("--panel", default="1890,3716,962,1022,2262")
    parser.add_argument(
        "--testmatched",
        default="handoff_to_teammate_20260727/splits/"
        "test_matched_seed2026_val.npy",
    )
    parser.add_argument("--feature-modes", default="none,los,local,all")
    parser.add_argument("--lambda-grid", default="0.5,1,2,4,8,16")
    parser.add_argument("--k-grid", default="2,3,4,6,8,12")
    parser.add_argument("--power-grid", default="2,3,4,5")
    parser.add_argument("--ray-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out", default="docs/mapaware_spectrum_knn/result.json"
    )
    return parser.parse_args()


def map_features(
    positions: np.ndarray,
    bs_position: np.ndarray,
    heightmap: np.ndarray,
    x0: float,
    y0: float,
    resolution: float,
    ray_samples: int,
) -> tuple[np.ndarray, list[str]]:
    t = np.linspace(0.02, 0.98, ray_samples, dtype=np.float32)
    start = bs_position.astype(np.float32)
    direction = positions.astype(np.float32) - start
    points = start[None, None, :] + direction[:, None, :] * t[None, :, None]
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
    ray_height = heightmap[gx, gy]
    excess = ray_height - points[..., 2]
    blocked = excess > 0.5
    any_blocked = blocked.any(axis=1)
    first = np.where(
        any_blocked,
        np.argmax(blocked, axis=1) / max(ray_samples - 1, 1),
        1.0,
    )
    last = np.where(
        any_blocked,
        1.0
        - np.argmax(blocked[:, ::-1], axis=1)
        / max(ray_samples - 1, 1),
        0.0,
    )
    positive = np.maximum(excess, 0.0)

    occupancy = (heightmap > 0.5).astype(np.float32)
    occ3 = ndimage.uniform_filter(occupancy, size=7, mode="nearest")
    occ8 = ndimage.uniform_filter(occupancy, size=17, mode="nearest")
    max8 = ndimage.maximum_filter(heightmap, size=17, mode="nearest")
    target_gx = np.clip(
        ((positions[:, 0] - x0) / resolution).astype(np.int64),
        0,
        heightmap.shape[0] - 1,
    )
    target_gy = np.clip(
        ((positions[:, 1] - y0) / resolution).astype(np.int64),
        0,
        heightmap.shape[1] - 1,
    )
    local_height = heightmap[target_gx, target_gy]
    values = np.column_stack(
        [
            any_blocked.astype(np.float32),
            np.maximum(excess.max(axis=1), 0.0),
            positive.mean(axis=1),
            blocked.mean(axis=1),
            first.astype(np.float32),
            last.astype(np.float32),
            local_height,
            occ3[target_gx, target_gy],
            occ8[target_gx, target_gy],
            max8[target_gx, target_gy],
        ]
    ).astype(np.float32)
    names = [
        "los_blocked",
        "los_max_excess",
        "los_positive_mean",
        "los_blocked_fraction",
        "los_first_block",
        "los_last_block",
        "local_height",
        "local_occ_3m",
        "local_occ_8m",
        "local_max_height_8m",
    ]
    return values, names


def robust_standardize(
    train: np.ndarray, all_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.median(train, axis=0)
    q25, q75 = np.quantile(train, [0.25, 0.75], axis=0)
    scale = q75 - q25
    std = train.std(axis=0)
    scale = np.where(scale > 1e-5, scale, np.maximum(std, 1e-3))
    return (
        ((all_values - center) / scale).astype(np.float32),
        center.astype(np.float32),
        scale.astype(np.float32),
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    datadir = Path(args.datadir)
    spec = load_setup(datadir / "Round1_Setup.json")
    train_pos = np.load(
        datadir / "Round1_Train_Pos.npy"
    ).astype(np.float32)
    test_pos = np.load(
        datadir / "Round1_Test_Pos.npy"
    ).astype(np.float32)
    all_pos = np.concatenate([train_pos, test_pos], axis=0)
    points = load_point_cloud(datadir / "Round1_Map.ply")
    heightmap, x0, y0, resolution = build_heightmap(points)
    raw_features, feature_names = map_features(
        all_pos,
        np.asarray(spec.bs_position, dtype=np.float32),
        heightmap,
        x0,
        y0,
        resolution,
        args.ray_samples,
    )
    features, feature_center, feature_scale = robust_standardize(
        raw_features[: len(train_pos)], raw_features
    )
    mode_columns = {
        "none": [],
        "los": list(range(6)),
        "local": list(range(6, 10)),
        "all": list(range(10)),
    }
    requested_modes = [
        value.strip() for value in args.feature_modes.split(",") if value
    ]
    unknown = set(requested_modes) - set(mode_columns)
    if unknown:
        raise ValueError(f"unknown feature modes: {sorted(unknown)}")
    lambdas = [
        float(value) for value in args.lambda_grid.split(",") if value
    ]
    metric_configs = [{"mode": "none", "lambda": 0.0}]
    metric_configs.extend(
        {"mode": mode, "lambda": value}
        for mode in requested_modes
        if mode != "none"
        for value in lambdas
    )
    k_grid = [int(value) for value in args.k_grid.split(",") if value]
    powers = [
        float(value) for value in args.power_grid.split(",") if value
    ]
    max_k = max(k_grid)

    cache_dir = Path(args.cache_dir)
    print("[map-knn] loading PHV spectra", flush=True)
    pas = torch.as_tensor(
        np.array(
            np.load(cache_dir / "train_pas_phv.npy", mmap_mode="r"),
            copy=True,
        ),
        dtype=torch.float32,
        device=device,
    )
    pdp = torch.as_tensor(
        np.array(
            np.load(cache_dir / "train_pdp.npy", mmap_mode="r"),
            copy=True,
        ),
        dtype=torch.float32,
        device=device,
    )
    pas = pas / pas.norm(dim=-1, keepdim=True).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    pdp = pdp / pdp.norm(dim=-1, keepdim=True).clamp_min(
        torch.finfo(torch.float32).tiny
    )

    seeds = [int(value) for value in args.panel.split(",") if value]
    tune_seeds, audit_seed = seeds[:-1], seeds[-1]
    splits = {
        str(seed): np.asarray(
            sorted(reproduce_val_indices(len(train_pos), 0.1, seed)),
            dtype=np.int64,
        )
        for seed in seeds
    }
    splits["testmatched"] = np.load(args.testmatched).astype(np.int64)
    all_train_idx = np.arange(len(train_pos), dtype=np.int64)
    scores = {
        domain: {
            (
                metric["mode"],
                metric["lambda"],
                k,
                power,
            ): {}
            for metric in metric_configs
            for k in k_grid
            for power in powers
        }
        for domain in ("pas", "pdp")
    }

    def neighbors_for_metric(
        val_idx: np.ndarray,
        pool_idx: np.ndarray,
        metric: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        xy_diff = (
            train_pos[val_idx, None, :2]
            - train_pos[pool_idx][None, :, :2]
        )
        distance2 = np.square(xy_diff).sum(axis=-1)
        columns = mode_columns[metric["mode"]]
        if columns:
            feature_diff = (
                features[val_idx][:, columns][:, None, :]
                - features[pool_idx][:, columns][None, :, :]
            )
            feature_distance = np.square(feature_diff).mean(axis=-1)
            distance2 += metric["lambda"] ** 2 * feature_distance
        local = np.argpartition(
            distance2, kth=max_k - 1, axis=1
        )[:, :max_k]
        selected_distance = np.take_along_axis(distance2, local, axis=1)
        order = np.argsort(selected_distance, axis=1)
        local = np.take_along_axis(local, order, axis=1)
        selected_distance = np.take_along_axis(
            selected_distance, order, axis=1
        )
        return pool_idx[local], np.sqrt(
            np.maximum(selected_distance, 1e-6)
        ).astype(np.float32)

    def score_domain(
        values: torch.Tensor,
        truth_idx: torch.Tensor,
        neighbors: np.ndarray,
        distance: np.ndarray,
        k: int,
        power: float,
    ) -> float:
        total = 0.0
        count = 0
        for start in range(0, len(neighbors), args.batch_size):
            stop = min(start + args.batch_size, len(neighbors))
            index = torch.as_tensor(
                neighbors[start:stop, :k],
                dtype=torch.long,
                device=device,
            )
            weight = torch.as_tensor(
                distance[start:stop, :k],
                device=device,
            ).clamp_min(1e-3).pow(-power)
            weight /= weight.sum(dim=1, keepdim=True)
            prediction = torch.einsum(
                "bk,bk...->b...", weight, values[index]
            )
            prediction /= prediction.norm(
                dim=-1, keepdim=True
            ).clamp_min(torch.finfo(torch.float32).tiny)
            truth = values[truth_idx[start:stop]]
            cosine = (prediction * truth).sum(dim=-1).clamp(-1, 1)
            total += float(cosine.sum())
            count += cosine.numel()
        return total / count

    for split_name, val_idx in splits.items():
        pool_idx = np.setdiff1d(all_train_idx, val_idx)
        truth_idx = torch.as_tensor(
            val_idx, dtype=torch.long, device=device
        )
        for metric_index, metric in enumerate(metric_configs):
            neighbors, distance = neighbors_for_metric(
                val_idx, pool_idx, metric
            )
            for k in k_grid:
                for power in powers:
                    key = (metric["mode"], metric["lambda"], k, power)
                    scores["pas"][key][split_name] = score_domain(
                        pas, truth_idx, neighbors, distance, k, power
                    )
                    scores["pdp"][key][split_name] = score_domain(
                        pdp, truth_idx, neighbors, distance, k, power
                    )
            if metric_index % 4 == 0:
                print(
                    f"[map-knn] split={split_name} "
                    f"metric={metric_index + 1}/{len(metric_configs)}",
                    flush=True,
                )

    def rank(domain: str) -> list[dict]:
        rows = []
        for key, split_scores in scores[domain].items():
            mode, metric_lambda, k, power = key
            tune = [split_scores[str(seed)] for seed in tune_seeds]
            rows.append(
                {
                    "feature_mode": mode,
                    "feature_lambda": metric_lambda,
                    "k": k,
                    "distance_power": power,
                    "tune_median": float(np.median(tune)),
                    "tune_mean": float(np.mean(tune)),
                    "tune_worst": float(np.min(tune)),
                    "audit": split_scores[str(audit_seed)],
                    "testmatched": split_scores["testmatched"],
                    "scores": split_scores,
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

    payload = {
        "selection_policy": "tune panel only",
        "pas_layout": "phv",
        "panel": seeds,
        "feature_names": feature_names,
        "feature_center": feature_center.tolist(),
        "feature_scale": feature_scale.tolist(),
        "raw_feature_summary": {
            name: {
                "train_mean": float(raw_features[: len(train_pos), i].mean()),
                "test_mean": float(raw_features[len(train_pos) :, i].mean()),
                "train_std": float(raw_features[: len(train_pos), i].std()),
                "test_std": float(raw_features[len(train_pos) :, i].std()),
            }
            for i, name in enumerate(feature_names)
        },
        "pas_ranked": rank("pas"),
        "pdp_ranked": rank("pdp"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("[PAS TOP]", json.dumps(payload["pas_ranked"][:5]), flush=True)
    print("[PDP TOP]", json.dumps(payload["pdp_ranked"][:5]), flush=True)
    print(f"MAPAWARE_SWEEP_DONE out={out}", flush=True)


if __name__ == "__main__":
    main()
