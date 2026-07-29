#!/usr/bin/env python3
"""Build label-free validation panels matched to the test coordinates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree, distance


def nearest_statistics(
    query: np.ndarray, pool: np.ndarray
) -> dict[str, float | list[float]]:
    nearest = cKDTree(pool[:, :2]).query(query[:, :2], k=16)[0]
    return {
        "nearest_mean": float(nearest[:, 0].mean()),
        "nearest_quantiles": [
            float(value)
            for value in np.quantile(
                nearest[:, 0], [0.0, 0.1, 0.5, 0.9, 1.0]
            )
        ],
        "k4_mean": float(nearest[:, 3].mean()),
        "k16_mean": float(nearest[:, 15].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", default="Round1_Map(2)")
    parser.add_argument("--panel-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--density-weight", type=float, default=1.0)
    parser.add_argument(
        "--outdir", default="cache/test_geometry_panel"
    )
    parser.add_argument(
        "--report", default="docs/test_geometry_panel/result.json"
    )
    args = parser.parse_args()

    datadir = Path(args.datadir)
    train = np.load(datadir / "Round1_Train_Pos.npy").astype(
        np.float64
    )
    test = np.load(datadir / "Round1_Test_Pos.npy").astype(
        np.float64
    )
    coordinate_scale = np.std(
        np.concatenate([train[:, :2], test[:, :2]], axis=0),
        axis=0,
    )
    coordinate_cost = distance.cdist(
        test[:, :2] / coordinate_scale,
        train[:, :2] / coordinate_scale,
        "sqeuclidean",
    )
    tree = cKDTree(train[:, :2])
    test_neighbor_distance = tree.query(
        test[:, :2], k=16
    )[0][:, [0, 3, 15]]
    train_neighbor_distance = tree.query(
        train[:, :2], k=17
    )[0][:, [1, 4, 16]]
    density_scale = np.std(
        np.concatenate(
            [test_neighbor_distance, train_neighbor_distance],
            axis=0,
        ),
        axis=0,
    )
    density_cost = distance.cdist(
        test_neighbor_distance / density_scale,
        train_neighbor_distance / density_scale,
        "sqeuclidean",
    ) / test_neighbor_distance.shape[1]
    cost = coordinate_cost + args.density_weight * density_cost
    test_rows, train_columns = linear_sum_assignment(cost)
    order = np.argsort(test_rows)
    matched_train = train_columns[order].astype(np.int64)
    matched_test = test_rows[order].astype(np.int64)

    # Sort into coarse spatial strata, then distribute each stratum
    # round-robin so every disjoint panel spans the full test region.
    x_edges = np.quantile(test[:, 0], np.linspace(0.0, 1.0, 6))
    y_edges = np.quantile(test[:, 1], np.linspace(0.0, 1.0, 6))
    x_bin = np.clip(
        np.searchsorted(
            x_edges[1:-1], test[matched_test, 0], side="right"
        ),
        0,
        4,
    )
    y_bin = np.clip(
        np.searchsorted(
            y_edges[1:-1], test[matched_test, 1], side="right"
        ),
        0,
        4,
    )
    rng = np.random.default_rng(args.seed)
    panel_members: list[list[int]] = [
        [] for _ in range(args.panel_count)
    ]
    for stratum in range(25):
        members = np.flatnonzero(x_bin * 5 + y_bin == stratum)
        rng.shuffle(members)
        offset = int(rng.integers(args.panel_count))
        for local_rank, member in enumerate(members):
            panel_members[
                (offset + local_rank) % args.panel_count
            ].append(int(matched_train[member]))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "proxy_all.npy", matched_train)
    all_indices = np.arange(len(train), dtype=np.int64)
    panels = {}
    for panel_number, members in enumerate(panel_members):
        indices = np.asarray(sorted(members), dtype=np.int64)
        np.save(outdir / f"proxy_p{panel_number}.npy", indices)
        pool = np.setdiff1d(all_indices, indices)
        panels[f"p{panel_number}"] = {
            "count": int(len(indices)),
            "indices": str(
                outdir / f"proxy_p{panel_number}.npy"
            ),
            "query_pool_neighbors": nearest_statistics(
                train[indices], train[pool]
            ),
            "position_mean": [
                float(value)
                for value in train[indices].mean(axis=0)
            ],
            "position_std": [
                float(value)
                for value in train[indices].std(axis=0)
            ],
        }

    matched_distance = np.linalg.norm(
        test[matched_test, :2] - train[matched_train, :2],
        axis=1,
    )
    payload = {
        "selection_policy": (
            "minimum-cost one-to-one matching uses XY and label-free "
            "d1/d4/d16 neighborhood geometry; matched proxies are "
            "stratified into disjoint panels"
        ),
        "density_weight": args.density_weight,
        "matched_count": int(len(matched_train)),
        "matched_distance": {
            "mean": float(matched_distance.mean()),
            "median": float(np.median(matched_distance)),
            "q90": float(np.quantile(matched_distance, 0.9)),
            "max": float(matched_distance.max()),
        },
        "test_neighbors_in_full_train": nearest_statistics(test, train),
        "test_position_mean": [
            float(value) for value in test.mean(axis=0)
        ],
        "test_position_std": [
            float(value) for value in test.std(axis=0)
        ],
        "proxy_position_mean": [
            float(value) for value in train[matched_train].mean(axis=0)
        ],
        "proxy_position_std": [
            float(value) for value in train[matched_train].std(axis=0)
        ],
        "panels": panels,
    }
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n"
    )
    print(
        "TEST_GEOMETRY_PANEL_DONE "
        f"matched_mean={matched_distance.mean():.4f} "
        f"sizes={[len(items) for items in panel_members]} "
        f"report={report}",
        flush=True,
    )


if __name__ == "__main__":
    main()
