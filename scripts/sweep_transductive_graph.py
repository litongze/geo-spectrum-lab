#!/usr/bin/env python3
"""Probe transductive graph interpolation as a clean spectrum expert."""
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
from wireless_twin.signal import pas_spectrum_phv, pdp_spectrum


def stable_unit(value: torch.Tensor) -> torch.Tensor:
    maximum = value.amax(dim=-1, keepdim=True)
    scaled = value / maximum.clamp_min(torch.finfo(value.dtype).tiny)
    norm = scaled.norm(dim=-1, keepdim=True)
    return torch.where(norm > 0, scaled / norm.clamp_min(1e-30), 0.0)


def direct_weights(
    query: np.ndarray,
    pool: np.ndarray,
    k: int,
    power: float,
) -> np.ndarray:
    distance, neighbors = cKDTree(pool[:, :2]).query(query[:, :2], k=k)
    distance = np.atleast_2d(distance)
    neighbors = np.atleast_2d(neighbors)
    weight = np.maximum(distance, 1e-3) ** -power
    weight /= weight.sum(axis=1, keepdims=True)
    result = np.zeros((len(query), len(pool)), dtype=np.float64)
    np.put_along_axis(result, neighbors, weight, axis=1)
    return result


def graph_weights(
    query: np.ndarray,
    pool: np.ndarray,
    direct: np.ndarray,
    graph_k: int,
    power: float,
    eta: float,
) -> np.ndarray:
    if eta == 0:
        return direct.astype(np.float32)
    points = np.concatenate([pool[:, :2], query[:, :2]], axis=0)
    distance, neighbors = cKDTree(points).query(
        points[len(pool) :], k=graph_k + 1
    )
    distance, neighbors = distance[:, 1:], neighbors[:, 1:]
    edge = np.maximum(distance, 1e-3) ** -power
    edge /= edge.sum(axis=1, keepdims=True)
    labeled = np.zeros((len(query), len(pool)), dtype=np.float64)
    unlabeled = np.zeros((len(query), len(query)), dtype=np.float64)
    for row in range(len(query)):
        label_mask = neighbors[row] < len(pool)
        labeled[row, neighbors[row, label_mask]] = edge[row, label_mask]
        query_columns = neighbors[row, ~label_mask] - len(pool)
        unlabeled[row, query_columns] = edge[row, ~label_mask]
    system = np.eye(len(query), dtype=np.float64) - eta * unlabeled
    rhs = (1.0 - eta) * direct + eta * labeled
    result = np.linalg.solve(system, rhs)
    result = np.maximum(result, 0.0)
    result /= result.sum(axis=1, keepdims=True).clip(1e-12)
    return result.astype(np.float32)


def sparsify(
    coefficients: np.ndarray, keep: int
) -> tuple[np.ndarray, np.ndarray, float]:
    keep = min(keep, coefficients.shape[1])
    local = np.argpartition(coefficients, -keep, axis=1)[:, -keep:]
    weight = np.take_along_axis(coefficients, local, axis=1)
    retained = float(weight.sum(axis=1).mean())
    order = np.argsort(weight, axis=1)[:, ::-1]
    local = np.take_along_axis(local, order, axis=1)
    weight = np.take_along_axis(weight, order, axis=1)
    weight /= weight.sum(axis=1, keepdims=True).clip(1e-12)
    return local.astype(np.int64), weight.astype(np.float32), retained


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", default="Round1_Map(2)")
    parser.add_argument("--cache-dir", default="cache/teammate_knn_hvp")
    parser.add_argument("--split-seed", type=int, default=1890)
    parser.add_argument("--baseline-val", required=True)
    parser.add_argument("--direct-k", type=int, default=8)
    parser.add_argument("--direct-power", type=float, default=2.0)
    parser.add_argument("--graph-k-grid", default="4,8,12,16")
    parser.add_argument("--graph-power-grid", default="1,2,4")
    parser.add_argument("--eta-grid", default="0.25,0.5,0.75,0.9")
    parser.add_argument("--blend-grid", default="0,0.1,0.2,0.3,0.5")
    parser.add_argument("--sparse-keep", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out", default="docs/transductive_graph_s1890/result.json"
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
    direct = direct_weights(
        positions[val_idx],
        positions[pool_idx],
        args.direct_k,
        args.direct_power,
    )
    graph_ks = [int(value) for value in args.graph_k_grid.split(",")]
    graph_powers = [
        float(value) for value in args.graph_power_grid.split(",")
    ]
    etas = [float(value) for value in args.eta_grid.split(",")]
    configs = [
        {
            "name": "direct",
            "graph_k": 0,
            "graph_power": 0.0,
            "eta": 0.0,
            "coefficients": direct.astype(np.float32),
        }
    ]
    for graph_k in graph_ks:
        for graph_power in graph_powers:
            for eta in etas:
                configs.append(
                    {
                        "name": (
                            f"g{graph_k}_p{graph_power:g}_e{eta:g}"
                        ),
                        "graph_k": graph_k,
                        "graph_power": graph_power,
                        "eta": eta,
                        "coefficients": graph_weights(
                            positions[val_idx],
                            positions[pool_idx],
                            direct,
                            graph_k,
                            graph_power,
                            eta,
                        ),
                    }
                )
    for config in configs:
        local, weight, retained = sparsify(
            config.pop("coefficients"), args.sparse_keep
        )
        config["neighbors"] = pool_idx[local]
        config["weight"] = weight
        config["retained_mass"] = retained

    baseline_channel = torch.as_tensor(
        np.array(np.load(args.baseline_val, mmap_mode="r"), copy=True),
        dtype=torch.complex64,
        device=device,
    )
    baseline = {
        "pas": stable_unit(
            pas_spectrum_phv(baseline_channel, spec).reshape(
                len(val_idx), -1, spec.mh * spec.mv
            )
        ),
        "pdp": stable_unit(
            pdp_spectrum(baseline_channel, spec).reshape(
                len(val_idx), -1, spec.s
            )
        ),
    }
    del baseline_channel
    blend_grid = [
        float(value) for value in args.blend_grid.split(",")
    ]
    payload = {
        "split_seed": args.split_seed,
        "direct_k": args.direct_k,
        "direct_power": args.direct_power,
        "sparse_keep": args.sparse_keep,
        "domains": {},
    }
    cache_dir = Path(args.cache_dir)
    for domain, filename in (
        ("pas", "train_pas_phv.npy"),
        ("pdp", "train_pdp.npy"),
    ):
        print(f"[graph] loading {domain}", flush=True)
        spectra = stable_unit(
            torch.as_tensor(
                np.array(
                    np.load(cache_dir / filename, mmap_mode="r"), copy=True
                ),
                dtype=torch.float32,
                device=device,
            )
        )
        vector_length = spectra.shape[-1]
        spectra = spectra.reshape(len(positions), -1, vector_length)
        truth = spectra[
            torch.as_tensor(val_idx, dtype=torch.long, device=device)
        ]
        rows = []
        for config_index, config in enumerate(configs):
            totals = torch.zeros(len(blend_grid), device=device)
            count = 0
            for start in range(0, len(val_idx), args.batch_size):
                stop = min(start + args.batch_size, len(val_idx))
                neighbors = torch.as_tensor(
                    config["neighbors"][start:stop],
                    dtype=torch.long,
                    device=device,
                )
                weight = torch.as_tensor(
                    config["weight"][start:stop], device=device
                )
                graph = stable_unit(
                    torch.einsum(
                        "bk,bk...->b...", weight, spectra[neighbors]
                    )
                )
                target = truth[start:stop]
                for blend_index, beta in enumerate(blend_grid):
                    prediction = stable_unit(
                        (1.0 - beta) * baseline[domain][start:stop]
                        + beta * graph
                    )
                    totals[blend_index] += (
                        (prediction * target).sum(-1).sum()
                    )
                count += target.shape[0] * target.shape[1]
            scores = (totals / count).detach().cpu().tolist()
            row = {
                key: value
                for key, value in config.items()
                if key not in ("neighbors", "weight")
            }
            row["blend_scores"] = {
                str(beta): float(score)
                for beta, score in zip(blend_grid, scores)
            }
            row["best_blend"] = max(
                (
                    {"beta": beta, "score": float(score)}
                    for beta, score in zip(blend_grid, scores)
                ),
                key=lambda value: value["score"],
            )
            rows.append(row)
            if config_index % 8 == 0:
                print(
                    f"[graph] {domain} {config_index + 1}/{len(configs)}",
                    flush=True,
                )
        payload["domains"][domain] = sorted(
            rows,
            key=lambda row: row["best_blend"]["score"],
            reverse=True,
        )
        del spectra, truth, baseline[domain]
        torch.cuda.empty_cache()
        print(
            f"[graph] {domain} best="
            f"{payload['domains'][domain][0]['best_blend']}",
            flush=True,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"TRANSDUCTIVE_GRAPH_DONE out={out}", flush=True)


if __name__ == "__main__":
    main()
