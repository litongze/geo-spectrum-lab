#!/usr/bin/env python3
"""Sweep local polynomial spatial interpolation for PAS and PDP spectra."""
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
    parser.add_argument("--cache-dir", default="cache/teammate_knn_hvp")
    parser.add_argument(
        "--pas-layout",
        choices=("hvp", "pvh", "phv"),
        default="hvp",
    )
    parser.add_argument("--tune-seeds", default="1890,3716,962,1022")
    parser.add_argument("--audit-seed", type=int, default=2262)
    parser.add_argument(
        "--testmatched",
        default="handoff_to_teammate_20260727/splits/"
        "test_matched_seed2026_val.npy",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out", default="docs/local_regression_sweep/result.json"
    )
    return parser.parse_args()


def configurations() -> list[dict]:
    rows = []
    for degree in (1, 2):
        k_values = (4, 6, 8, 12, 16) if degree == 1 else (8, 12, 16, 24)
        for k in k_values:
            for power in (0.0, 1.0, 2.0):
                for ridge in (0.01, 0.1, 1.0):
                    for clip in (False, True):
                        rows.append(
                            {
                                "name": (
                                    f"ll{degree}_k{k}_dp{power:g}_"
                                    f"r{ridge:g}_clip{int(clip)}"
                                ),
                                "degree": degree,
                                "k": k,
                                "power": power,
                                "ridge": ridge,
                                "clip": clip,
                            }
                        )
    for k, power in ((2, 4.0), (3, 2.0), (4, 5.0), (6, 2.0)):
        rows.append(
            {
                "name": f"idw_k{k}_dp{power:g}",
                "degree": 0,
                "k": k,
                "power": power,
                "ridge": 0.0,
                "clip": True,
            }
        )
    return rows


def polynomial_design(delta: np.ndarray, degree: int) -> np.ndarray:
    dx, dy = delta[:, 0], delta[:, 1]
    columns = [np.ones(len(delta)), dx, dy]
    if degree == 2:
        columns.extend([dx * dx, dx * dy, dy * dy])
    return np.stack(columns, axis=1)


def local_weights(
    query: np.ndarray,
    train: np.ndarray,
    neighbors: np.ndarray,
    distance: np.ndarray,
    config: dict,
) -> np.ndarray:
    k = config["k"]
    selected = neighbors[:, :k]
    if config["degree"] == 0:
        weight = np.power(
            np.maximum(distance[:, :k], 1e-3), -config["power"]
        )
        return (weight / weight.sum(axis=1, keepdims=True)).astype(
            np.float32
        )

    result = np.empty((len(query), k), dtype=np.float64)
    for row in range(len(query)):
        delta = train[selected[row], :2] - query[row, :2]
        scale = max(float(np.median(distance[row, :k])), 1e-3)
        design = polynomial_design(delta / scale, config["degree"])
        kernel = np.power(
            np.maximum(distance[row, :k] / scale, 1e-3),
            -config["power"],
        )
        gram = design.T @ (kernel[:, None] * design)
        regularizer = np.eye(gram.shape[0]) * config["ridge"] * kernel.sum()
        regularizer[0, 0] = 0.0
        coefficients = np.linalg.solve(
            gram + regularizer,
            np.eye(gram.shape[0], dtype=np.float64)[0],
        )
        weight = kernel * (design @ coefficients)
        if config["clip"]:
            weight = np.maximum(weight, 0.0)
        total = weight.sum()
        if not np.isfinite(total) or abs(total) < 1e-8:
            weight = kernel
            total = weight.sum()
        result[row] = weight / total
    return result.astype(np.float32)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    tune_seeds = [
        int(value) for value in args.tune_seeds.split(",") if value.strip()
    ]
    datadir = Path(args.datadir)
    train_pos = np.load(datadir / "Round1_Train_Pos.npy").astype(np.float32)
    cache_dir = Path(args.cache_dir)
    spectra = {
        "pas": torch.from_numpy(
            np.array(
                np.load(
                    cache_dir / f"train_pas_{args.pas_layout}.npy",
                    mmap_mode="r",
                ),
                copy=True,
            )
        ).to(device),
        "pdp": torch.from_numpy(
            np.array(
                np.load(cache_dir / "train_pdp.npy", mmap_mode="r"),
                copy=True,
            )
        ).to(device),
    }
    configs = configurations()
    max_k = max(config["k"] for config in configs)
    all_idx = np.arange(len(train_pos), dtype=np.int64)
    splits = {
        str(seed): np.asarray(
            sorted(reproduce_val_indices(len(train_pos), 0.1, seed)),
            dtype=np.int64,
        )
        for seed in tune_seeds + [args.audit_seed]
    }
    splits["testmatched"] = np.load(args.testmatched).astype(np.int64)
    scores = {
        domain: {config["name"]: {} for config in configs}
        for domain in spectra
    }

    for split_name, val_idx in splits.items():
        print(f"[split] {split_name}", flush=True)
        pool_idx = np.setdiff1d(all_idx, val_idx, assume_unique=False)
        tree = cKDTree(train_pos[pool_idx, :2])
        distance, local_idx = tree.query(train_pos[val_idx, :2], k=max_k)
        neighbors = pool_idx[local_idx]
        target_idx = torch.as_tensor(
            val_idx, dtype=torch.long, device=device
        )
        for config in configs:
            weight_np = local_weights(
                train_pos[val_idx],
                train_pos,
                neighbors,
                distance,
                config,
            )
            k = config["k"]
            weight = torch.as_tensor(weight_np, device=device)
            neighbor_idx = torch.as_tensor(
                neighbors[:, :k], dtype=torch.long, device=device
            )
            for domain, values in spectra.items():
                prediction = torch.einsum(
                    "bk,bk...->b...", weight, values[neighbor_idx]
                ).clamp_min(0.0)
                truth = values[target_idx]
                prediction /= prediction.norm(
                    dim=-1, keepdim=True
                ).clamp_min(torch.finfo(torch.float32).tiny)
                truth = truth / truth.norm(
                    dim=-1, keepdim=True
                ).clamp_min(torch.finfo(torch.float32).tiny)
                value = (
                    (prediction * truth)
                    .sum(dim=-1)
                    .clamp(-1.0, 1.0)
                    .mean()
                )
                scores[domain][config["name"]][split_name] = float(value)
                del prediction, truth
            del weight, neighbor_idx

    config_by_name = {config["name"]: config for config in configs}

    def rank(domain: str) -> list[dict]:
        rows = []
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
        "selection_policy": "ranked only on tune seeds",
        "pas_layout": args.pas_layout,
        "tune_seeds": tune_seeds,
        "audit_seed": args.audit_seed,
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
