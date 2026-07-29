#!/usr/bin/env python3
"""Search label-free map-aware KNN metrics for PAS and PDP interpolation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import uniform_filter

from _bootstrap import ROOT  # noqa: F401
from score_holdout import reproduce_val_indices
from wireless_twin.data.map_loader import load_point_cloud
from wireless_twin.data.setup_config import load_setup
from wireless_twin.models.raytrace2 import build_heightmap


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
    parser.add_argument("--pas-k", type=int, default=2)
    parser.add_argument("--pas-distance-power", type=float, default=4.0)
    parser.add_argument("--pdp-k", type=int, default=4)
    parser.add_argument("--pdp-distance-power", type=float, default=5.0)
    parser.add_argument("--ray-samples", type=int, default=192)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="docs/map_knn_sweep/result.json")
    return parser.parse_args()


def longest_run(mask: np.ndarray) -> np.ndarray:
    result = np.zeros(len(mask), dtype=np.float32)
    for row, values in enumerate(mask):
        padded = np.concatenate([[False], values, [False]])
        edges = np.flatnonzero(padded[1:] != padded[:-1])
        if len(edges) >= 2:
            result[row] = float(np.max(edges[1::2] - edges[::2]))
    return result


def map_features(
    positions: np.ndarray,
    bs_position: np.ndarray,
    heightmap: np.ndarray,
    x0: float,
    y0: float,
    resolution: float,
    ray_samples: int,
) -> dict[str, np.ndarray]:
    gx = np.clip(
        ((positions[:, 0] - x0) / resolution).astype(np.int64),
        0,
        heightmap.shape[0] - 1,
    )
    gy = np.clip(
        ((positions[:, 1] - y0) / resolution).astype(np.int64),
        0,
        heightmap.shape[1] - 1,
    )
    indoor = (heightmap[gx, gy] > positions[:, 2] + 0.5).astype(
        np.float32
    )

    ts = np.linspace(0.02, 0.98, ray_samples, dtype=np.float32)
    samples = (
        bs_position[None, None, :]
        + ts[None, :, None]
        * (positions[:, None, :] - bs_position[None, None, :])
    )
    sx = np.clip(
        ((samples[..., 0] - x0) / resolution).astype(np.int64),
        0,
        heightmap.shape[0] - 1,
    )
    sy = np.clip(
        ((samples[..., 1] - y0) / resolution).astype(np.int64),
        0,
        heightmap.shape[1] - 1,
    )
    ray_height = heightmap[sx, sy]
    blocked = ray_height > samples[..., 2] + 0.5
    entering = blocked[:, 0].astype(np.int32) + (
        blocked[:, 1:] & ~blocked[:, :-1]
    ).sum(axis=1)
    blocked_fraction = blocked.mean(axis=1)
    first = np.where(
        blocked.any(axis=1),
        blocked.argmax(axis=1) / max(ray_samples - 1, 1),
        1.0,
    )
    run = longest_run(blocked) / ray_samples
    clearance = np.maximum(
        ray_height - samples[..., 2], 0.0
    ).max(axis=1)
    path = np.stack(
        [
            np.clip(entering, 0, 8) / 8.0,
            blocked_fraction,
            first,
            run,
            np.clip(clearance, 0, 40) / 40.0,
        ],
        axis=1,
    ).astype(np.float32)

    occupied = (heightmap > 2.0).astype(np.float32)
    local_columns = []
    for radius in (3, 8, 16, 32):
        size = 2 * radius + 1
        density = uniform_filter(occupied, size=size, mode="nearest")
        mean_height = uniform_filter(heightmap, size=size, mode="nearest")
        local_columns.extend(
            [density[gx, gy], np.clip(mean_height[gx, gy], 0, 40) / 40.0]
        )
    local = np.stack(local_columns, axis=1).astype(np.float32)
    return {"indoor": indoor[:, None], "path": path, "local": local}


def metric_grid() -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for group in ("indoor", "path", "local"):
        for value in (2.0, 5.0, 10.0, 20.0):
            row = {"indoor": 0.0, "path": 0.0, "local": 0.0}
            row[group] = value
            rows.append(row)
    for indoor in (0.0, 5.0, 15.0):
        for path in (0.0, 5.0, 15.0):
            for local in (0.0, 5.0, 15.0):
                rows.append(
                    {"indoor": indoor, "path": path, "local": local}
                )
    unique = {}
    for row in rows:
        name = (
            f"i{row['indoor']:g}_p{row['path']:g}_l{row['local']:g}"
        )
        unique[name] = {"name": name, **row}
    return list(unique.values())


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    tune_seeds = [
        int(value) for value in args.tune_seeds.split(",") if value.strip()
    ]
    datadir = Path(args.datadir)
    setup_path = next(datadir.glob("Round*_Setup.json"))
    tag = setup_path.name.removesuffix("_Setup.json")
    spec = load_setup(setup_path)
    train_pos = np.load(datadir / f"{tag}_Train_Pos.npy").astype(np.float32)
    test_pos = np.load(datadir / f"{tag}_Test_Pos.npy").astype(np.float32)
    points = load_point_cloud(datadir / f"{tag}_Map.ply")
    heightmap, x0, y0, resolution = build_heightmap(points)
    all_pos = np.concatenate([train_pos, test_pos], axis=0)
    feature_groups = map_features(
        all_pos,
        np.asarray(spec.bs_position, dtype=np.float32),
        heightmap,
        x0,
        y0,
        resolution,
        args.ray_samples,
    )
    del points, heightmap
    train_features = {
        key: value[: len(train_pos)] for key, value in feature_groups.items()
    }
    test_features = {
        key: value[len(train_pos) :] for key, value in feature_groups.items()
    }
    print(
        "[features] train indoor=%.4f test indoor=%.4f"
        % (
            train_features["indoor"].mean(),
            test_features["indoor"].mean(),
        ),
        flush=True,
    )

    cache_dir = Path(args.cache_dir)
    pas = torch.from_numpy(
        np.array(
            np.load(cache_dir / "train_pas_hvp.npy", mmap_mode="r"),
            copy=True,
        )
    ).to(device)
    pdp = torch.from_numpy(
        np.array(
            np.load(cache_dir / "train_pdp.npy", mmap_mode="r"),
            copy=True,
        )
    ).to(device)
    all_idx = np.arange(len(train_pos), dtype=np.int64)
    splits = {
        str(seed): np.asarray(
            sorted(reproduce_val_indices(len(train_pos), 0.1, seed)),
            dtype=np.int64,
        )
        for seed in tune_seeds + [args.audit_seed]
    }
    splits["testmatched"] = np.load(args.testmatched).astype(np.int64)
    configs = metric_grid()
    scores = {
        domain: {
            config["name"]: {} for config in configs
        }
        for domain in ("pas", "pdp")
    }

    for split_name, val_idx in splits.items():
        print(f"[split] {split_name}", flush=True)
        pool_idx = np.setdiff1d(all_idx, val_idx, assume_unique=False)
        base_dist2 = (
            (
                train_pos[val_idx, None, :2]
                - train_pos[pool_idx][None, :, :2]
            )
            ** 2
        ).sum(axis=-1)
        feature_dist2 = {}
        for group in train_features:
            delta = (
                train_features[group][val_idx, None, :]
                - train_features[group][pool_idx][None, :, :]
            )
            feature_dist2[group] = (delta * delta).sum(axis=-1)

        for config in configs:
            dist2 = base_dist2.copy()
            for group in feature_dist2:
                dist2 += config[group] ** 2 * feature_dist2[group]
            max_k = max(args.pas_k, args.pdp_k)
            neighbors_local = np.argpartition(
                dist2, kth=max_k - 1, axis=1
            )[:, :max_k]
            selected_dist2 = np.take_along_axis(
                dist2, neighbors_local, axis=1
            )
            order = np.argsort(selected_dist2, axis=1)
            neighbors_local = np.take_along_axis(
                neighbors_local, order, axis=1
            )
            selected_dist2 = np.take_along_axis(
                selected_dist2, order, axis=1
            )
            neighbors = torch.as_tensor(
                pool_idx[neighbors_local], dtype=torch.long, device=device
            )
            distance = torch.as_tensor(
                np.sqrt(selected_dist2 + 1e-6),
                dtype=torch.float32,
                device=device,
            )
            target_idx = torch.as_tensor(
                val_idx, dtype=torch.long, device=device
            )
            for domain, spectra, k, power in (
                ("pas", pas, args.pas_k, args.pas_distance_power),
                ("pdp", pdp, args.pdp_k, args.pdp_distance_power),
            ):
                weight = distance[:, :k].pow(-power)
                weight /= weight.sum(dim=1, keepdim=True)
                prediction = torch.einsum(
                    "bk,bk...->b...", weight, spectra[neighbors[:, :k]]
                )
                vector_dim = -1
                pred_norm = prediction.norm(
                    dim=vector_dim, keepdim=True
                ).clamp_min(torch.finfo(torch.float32).tiny)
                truth = spectra[target_idx]
                truth_norm = truth.norm(
                    dim=vector_dim, keepdim=True
                ).clamp_min(torch.finfo(torch.float32).tiny)
                cosine = (
                    (prediction / pred_norm) * (truth / truth_norm)
                ).sum(dim=vector_dim).mean()
                scores[domain][config["name"]][split_name] = float(cosine)
                del prediction, truth, weight
            del neighbors, distance

    def rank(domain: str) -> list[dict]:
        rows = []
        config_by_name = {config["name"]: config for config in configs}
        for name, split_scores in scores[domain].items():
            tune = [split_scores[str(seed)] for seed in tune_seeds]
            rows.append(
                {
                    **config_by_name[name],
                    "tune_median": float(np.median(tune)),
                    "tune_mean": float(np.mean(tune)),
                    "tune_worst": float(np.min(tune)),
                    "audit": split_scores[str(args.audit_seed)],
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
        "selection_policy": (
            "metric weights ranked only on tune seeds; map features use no labels"
        ),
        "tune_seeds": tune_seeds,
        "audit_seed": args.audit_seed,
        "feature_summary": {
            "train_indoor": float(train_features["indoor"].mean()),
            "test_indoor": float(test_features["indoor"].mean()),
        },
        "pas_ranked": rank("pas"),
        "pdp_ranked": rank("pdp"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("[PAS TOP]", json.dumps(payload["pas_ranked"][:8]), flush=True)
    print("[PDP TOP]", json.dumps(payload["pdp_ranked"][:8]), flush=True)
    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
