#!/usr/bin/env python3
"""Probe whether query-to-neighbor building barriers predict spectral quality."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from _bootstrap import ROOT  # noqa: F401
from score_holdout import reproduce_val_indices
from wireless_twin.data.map_loader import load_point_cloud
from wireless_twin.models.raytrace2 import build_heightmap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", default="Round1_Map(2)")
    parser.add_argument("--cache-dir", default="cache/teammate_knn_hvp")
    parser.add_argument("--tune-seeds", default="1890,3716,962,1022")
    parser.add_argument("--audit-seed", type=int, default=2262)
    parser.add_argument(
        "--testmatched",
        default=(
            "handoff_to_teammate_20260727/splits/"
            "test_matched_seed2026_val.npy"
        ),
    )
    parser.add_argument("--candidate-k", type=int, default=64)
    parser.add_argument("--path-samples", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out", default="docs/barrier_neighbor_signal/result.json"
    )
    return parser.parse_args()


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


def path_features(
    query: np.ndarray,
    candidate: np.ndarray,
    heightmap: np.ndarray,
    x0: float,
    y0: float,
    resolution: float,
    samples: int,
) -> dict[str, np.ndarray]:
    """Return pair features that distinguish same-side from cross-wall pairs."""
    fraction = np.linspace(0.0, 1.0, samples, dtype=np.float32)
    points = (
        query[:, None, None, :2]
        + (
            candidate[:, :, None, :2]
            - query[:, None, None, :2]
        )
        * fraction[None, None, :, None]
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
    occupied = heightmap[gx, gy] > 2.0
    query_inside = occupied[..., :1]
    candidate_inside = occupied[..., -1:]
    endpoint_mismatch = np.logical_xor(
        query_inside, candidate_inside
    )[..., 0]
    same_endpoint = ~endpoint_mismatch

    # A useful path stays in the endpoint's medium. This penalizes a building
    # crossing between two outdoor points and an outdoor gap between points
    # that appear to be in the same building.
    medium_disagreement = (
        occupied != query_inside
    ).mean(axis=-1).astype(np.float32)
    medium_disagreement *= same_endpoint
    transitions = np.count_nonzero(
        occupied[..., 1:] != occupied[..., :-1], axis=-1
    ).astype(np.float32)
    crossing = (transitions > 0).astype(np.float32)
    return {
        "medium_disagreement": medium_disagreement,
        "endpoint_mismatch": endpoint_mismatch.astype(np.float32),
        "crossing": crossing,
        "transitions": transitions,
        "query_inside": query_inside[..., 0].astype(np.float32),
    }


def candidate_similarity(
    spectra: torch.Tensor,
    truth_idx: np.ndarray,
    neighbors: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    rows = []
    truth_t = torch.as_tensor(
        truth_idx, dtype=torch.long, device=device
    )
    with torch.inference_mode():
        for start in range(0, len(truth_idx), batch_size):
            stop = min(start + batch_size, len(truth_idx))
            index = torch.as_tensor(
                neighbors[start:stop], dtype=torch.long, device=device
            )
            candidate = spectra[index]
            truth = spectra[truth_t[start:stop]][:, None]
            cosine = (candidate * truth).sum(dim=-1)
            cosine = cosine.reshape(
                stop - start, neighbors.shape[1], -1
            ).mean(dim=-1)
            rows.append(cosine.cpu().numpy())
    return np.concatenate(rows).astype(np.float32)


def weighted_proxy(
    similarity: np.ndarray,
    distance: np.ndarray,
    features: dict[str, np.ndarray],
    config: tuple[int, float, float, float, float],
) -> float:
    k, power, path_lambda, mismatch_lambda, crossing_lambda = config
    log_weight = (
        -power * np.log(np.maximum(distance, 1e-3))
        - path_lambda * features["medium_disagreement"]
        - mismatch_lambda * features["endpoint_mismatch"]
        - crossing_lambda * features["crossing"]
    )
    selected = np.argpartition(-log_weight, kth=k - 1, axis=1)[
        :, :k
    ]
    selected_log = np.take_along_axis(log_weight, selected, axis=1)
    selected_log -= selected_log.max(axis=1, keepdims=True)
    weight = np.exp(selected_log)
    weight /= np.maximum(weight.sum(axis=1, keepdims=True), 1e-30)
    selected_similarity = np.take_along_axis(
        similarity, selected, axis=1
    )
    return float(np.mean(np.sum(weight * selected_similarity, axis=1)))


def summarize_feature(
    similarity: np.ndarray, feature: np.ndarray
) -> dict[str, float]:
    flat_similarity = similarity.reshape(-1)
    flat_feature = feature.reshape(-1)
    active = flat_feature > 0
    inactive = ~active
    correlation = (
        float(np.corrcoef(flat_similarity, flat_feature)[0, 1])
        if np.std(flat_feature) > 0
        else 0.0
    )
    return {
        "active_fraction": float(active.mean()),
        "similarity_active": (
            float(flat_similarity[active].mean())
            if np.any(active)
            else 0.0
        ),
        "similarity_inactive": (
            float(flat_similarity[inactive].mean())
            if np.any(inactive)
            else 0.0
        ),
        "correlation": correlation,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    datadir = Path(args.datadir)
    positions = np.load(
        datadir / "Round1_Train_Pos.npy"
    ).astype(np.float32)
    points = load_point_cloud(datadir / "Round1_Map.ply")
    heightmap, x0, y0, resolution = build_heightmap(points)

    tune_seeds = [
        int(value) for value in args.tune_seeds.split(",") if value
    ]
    split_indices = {
        str(seed): np.asarray(
            sorted(
                reproduce_val_indices(len(positions), 0.1, seed)
            ),
            dtype=np.int64,
        )
        for seed in tune_seeds + [args.audit_seed]
    }
    split_indices["testmatched"] = np.load(args.testmatched).astype(
        np.int64
    )
    all_indices = np.arange(len(positions), dtype=np.int64)
    split_data: dict[str, dict] = {}
    for name, val_idx in split_indices.items():
        pool_idx = np.setdiff1d(all_indices, val_idx)
        distance, local = cKDTree(positions[pool_idx, :2]).query(
            positions[val_idx, :2], k=args.candidate_k
        )
        neighbors = pool_idx[np.asarray(local)]
        split_data[name] = {
            "indices": val_idx,
            "neighbors": neighbors,
            "distance": np.asarray(distance, dtype=np.float32),
            "features": path_features(
                positions[val_idx],
                positions[neighbors],
                heightmap,
                x0,
                y0,
                resolution,
                args.path_samples,
            ),
        }

    configs = [
        (k, power, path_lambda, mismatch_lambda, crossing_lambda)
        for k in (4, 8, 16, 32)
        for power in (1.0, 2.0, 3.0)
        for path_lambda in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
        for mismatch_lambda in (0.0, 1.0, 2.0, 4.0)
        for crossing_lambda in (0.0, 0.5, 1.0, 2.0)
    ]
    cache_dir = Path(args.cache_dir)
    payload = {
        "selection_policy": (
            "rank path penalties on four tune folds; audit and "
            "test-matched splits are never used for selection"
        ),
        "candidate_k": args.candidate_k,
        "path_samples": args.path_samples,
        "tune_seeds": tune_seeds,
        "audit_seed": args.audit_seed,
        "domains": {},
    }
    for domain, cache_name in (
        ("pas", "train_pas_phv.npy"),
        ("pdp", "train_pdp.npy"),
    ):
        print(f"[barrier] loading {domain}", flush=True)
        spectra = stable_unit(
            torch.as_tensor(
                np.array(
                    np.load(cache_dir / cache_name, mmap_mode="r"),
                    copy=True,
                ),
                dtype=torch.float32,
                device=device,
            )
        )
        similarities = {}
        diagnostics = {}
        for split_name, data in split_data.items():
            similarity = candidate_similarity(
                spectra,
                data["indices"],
                data["neighbors"],
                args.batch_size,
                device,
            )
            similarities[split_name] = similarity
            diagnostics[split_name] = {
                key: summarize_feature(similarity, value)
                for key, value in data["features"].items()
                if key != "query_inside"
            }
            print(
                f"[barrier] domain={domain} split={split_name} "
                f"nearest={similarity[:, 0].mean():.6f}",
                flush=True,
            )

        rows = []
        for config in configs:
            scores = {
                split_name: weighted_proxy(
                    similarities[split_name],
                    data["distance"],
                    data["features"],
                    config,
                )
                for split_name, data in split_data.items()
            }
            tune = [scores[str(seed)] for seed in tune_seeds]
            rows.append(
                {
                    "k": config[0],
                    "distance_power": config[1],
                    "path_lambda": config[2],
                    "mismatch_lambda": config[3],
                    "crossing_lambda": config[4],
                    "tune_median": float(np.median(tune)),
                    "tune_mean": float(np.mean(tune)),
                    "tune_worst": float(np.min(tune)),
                    "audit": scores[str(args.audit_seed)],
                    "testmatched": scores["testmatched"],
                    "scores": scores,
                }
            )
        ranked = sorted(
            rows,
            key=lambda row: (
                row["tune_median"],
                row["tune_mean"],
                row["tune_worst"],
            ),
            reverse=True,
        )
        payload["domains"][domain] = {
            "diagnostics": diagnostics,
            "ranked": ranked[:100],
        }
        del spectra
        torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n"
    )
    for domain in ("pas", "pdp"):
        best = payload["domains"][domain]["ranked"][0]
        print(
            f"BARRIER_SIGNAL domain={domain} "
            f"cfg=k{best['k']}/p{best['distance_power']}/"
            f"path{best['path_lambda']}/"
            f"mismatch{best['mismatch_lambda']}/"
            f"cross{best['crossing_lambda']} "
            f"tune={best['tune_mean']:.6f} "
            f"audit={best['audit']:.6f} "
            f"testmatched={best['testmatched']:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
