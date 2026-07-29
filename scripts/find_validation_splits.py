#!/usr/bin/env python3
"""Find validation splits whose position distribution resembles the test set.

The search intentionally uses only public, label-free information:
Train_Pos, Test_Pos, and map-derived indoor/outdoor flags. It should be used to
choose validation seeds before scoring channel predictions, not after.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import _bootstrap  # noqa: F401
from score_holdout import reproduce_val_indices
from wireless_twin.data.map_loader import load_point_cloud
from wireless_twin.models.raytrace2 import build_heightmap


@dataclass
class SplitScore:
    seed: int
    total: float
    xy_quantile: float
    xy_mean_std: float
    nn_quantile: float
    grid_tv: float
    grid_miss: float
    indoor: float
    overlap_with_seed0: int


def load_positions(datadir: Path) -> tuple[np.ndarray, np.ndarray, str]:
    setup_files = sorted(datadir.glob("*_Setup.json"))
    if not setup_files:
        raise FileNotFoundError(f"No *_Setup.json found in {datadir}")
    tag = setup_files[0].name.replace("_Setup.json", "")
    train_pos = np.load(datadir / f"{tag}_Train_Pos.npy").astype(np.float32)
    test_pos = np.load(datadir / f"{tag}_Test_Pos.npy").astype(np.float32)
    return train_pos, test_pos, tag


def indoor_flags(datadir: Path, tag: str, pos: np.ndarray) -> np.ndarray:
    pts = load_point_cloud(datadir / f"{tag}_Map.ply")
    hm, x0, y0, res = build_heightmap(pts)
    gx = np.clip(((pos[:, 0] - x0) / res).astype(int), 0, hm.shape[0] - 1)
    gy = np.clip(((pos[:, 1] - y0) / res).astype(int), 0, hm.shape[1] - 1)
    return (hm[gx, gy] > 2.0).astype(np.float32)


def hist2d(xy: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray) -> np.ndarray:
    h, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=(x_edges, y_edges))
    h = h.astype(np.float64)
    return h / max(h.sum(), 1.0)


def quantile_loss(a: np.ndarray, b: np.ndarray, spans: np.ndarray) -> float:
    qs = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    aq = np.quantile(a, qs, axis=0)
    bq = np.quantile(b, qs, axis=0)
    return float(np.mean(np.abs(aq - bq) / spans))


def mean_std_loss(a: np.ndarray, b: np.ndarray, spans: np.ndarray) -> float:
    am = a.mean(0)
    bm = b.mean(0)
    asd = a.std(0)
    bsd = b.std(0)
    return float(0.5 * np.mean(np.abs(am - bm) / spans)
                 + 0.5 * np.mean(np.abs(asd - bsd) / spans))


def first_kept_neighbor_distance(
    val_idx: np.ndarray,
    full_neighbor_idx: np.ndarray,
    full_neighbor_dist: np.ndarray,
    n: int,
) -> np.ndarray:
    is_val = np.zeros(n, dtype=bool)
    is_val[val_idx] = True
    d = np.empty(len(val_idx), dtype=np.float64)
    for out_i, row_i in enumerate(val_idx):
        found = False
        for dist, nb in zip(full_neighbor_dist[row_i], full_neighbor_idx[row_i]):
            if nb != row_i and not is_val[nb]:
                d[out_i] = dist
                found = True
                break
        if not found:
            d[out_i] = full_neighbor_dist[row_i, -1]
    return d


def score_seed(
    seed: int,
    train_pos: np.ndarray,
    test_pos: np.ndarray,
    train_indoor: np.ndarray,
    test_indoor: np.ndarray,
    spans: np.ndarray,
    test_hist: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    train_neighbor_idx: np.ndarray,
    train_neighbor_dist: np.ndarray,
    test_nn_dist: np.ndarray,
    seed0_set: set[int],
    val_fraction: float,
) -> SplitScore:
    n = len(train_pos)
    val_idx = np.array(sorted(reproduce_val_indices(n, val_fraction, seed)))
    val_pos = train_pos[val_idx]
    val_indoor = train_indoor[val_idx]

    xy_q = quantile_loss(val_pos[:, :2], test_pos[:, :2], spans[:2])
    xy_ms = mean_std_loss(val_pos[:, :2], test_pos[:, :2], spans[:2])

    val_nn = first_kept_neighbor_distance(
        val_idx, train_neighbor_idx, train_neighbor_dist, n)
    nn_span = max(float(np.quantile(test_nn_dist, 0.99)
                        - np.quantile(test_nn_dist, 0.01)), 1.0)
    nn_q = quantile_loss(val_nn[:, None], test_nn_dist[:, None],
                         np.array([nn_span], dtype=np.float64))

    val_hist = hist2d(val_pos[:, :2], x_edges, y_edges)
    grid_tv = float(0.5 * np.abs(val_hist - test_hist).sum())
    test_cells = test_hist > 0
    grid_miss = float(((val_hist <= 0) & test_cells).sum()
                      / max(test_cells.sum(), 1))
    indoor = float(abs(val_indoor.mean() - test_indoor.mean()))

    total = (
        0.32 * xy_q
        + 0.18 * xy_ms
        + 0.20 * nn_q
        + 0.18 * grid_tv
        + 0.07 * grid_miss
        + 0.05 * indoor
    )
    overlap = len(set(val_idx.tolist()) & seed0_set)
    return SplitScore(seed, total, xy_q, xy_ms, nn_q, grid_tv, grid_miss,
                      indoor, overlap)


def choose_panel(
    rows: list[SplitScore],
    n_train: int,
    val_fraction: float,
    n_panel: int,
    max_overlap: int,
) -> list[SplitScore]:
    chosen: list[SplitScore] = []
    chosen_sets: list[set[int]] = []
    for row in rows:
        idx = set(reproduce_val_indices(n_train, val_fraction, row.seed))
        if all(len(idx & s) <= max_overlap for s in chosen_sets):
            chosen.append(row)
            chosen_sets.append(idx)
        if len(chosen) == n_panel:
            break
    return chosen


def write_report(
    outdir: Path,
    rows: list[SplitScore],
    panel: list[SplitScore],
    seed0: SplitScore,
    max_seed: int,
    val_fraction: float,
    train_pos: np.ndarray,
    test_pos: np.ndarray,
    train_indoor: np.ndarray,
    test_indoor: np.ndarray,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "validation_split_search.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("rank,seed,total,xy_quantile,xy_mean_std,nn_quantile,"
                "grid_tv,grid_miss,indoor,overlap_with_seed0\n")
        for rank, r in enumerate(rows, 1):
            f.write(
                f"{rank},{r.seed},{r.total:.8f},{r.xy_quantile:.8f},"
                f"{r.xy_mean_std:.8f},{r.nn_quantile:.8f},{r.grid_tv:.8f},"
                f"{r.grid_miss:.8f},{r.indoor:.8f},{r.overlap_with_seed0}\n"
            )

    md_path = outdir / "validation_split_search.md"
    seed0_rank = next(i for i, r in enumerate(rows, 1) if r.seed == 0)
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Validation Split Search\n\n")
        f.write(f"Scanned seeds: 0..{max_seed - 1}, val_fraction={val_fraction}\n\n")
        f.write("Objective uses only Train_Pos, Test_Pos, and map-derived "
                "indoor/outdoor flags. Lower score is better.\n\n")
        f.write("## Distribution Summary\n\n")
        f.write("| set | n | x_mean | y_mean | indoor_ratio |\n")
        f.write("| --- | ---: | ---: | ---: | ---: |\n")
        f.write(f"| train | {len(train_pos)} | {train_pos[:,0].mean():.3f} | "
                f"{train_pos[:,1].mean():.3f} | {train_indoor.mean():.3f} |\n")
        f.write(f"| test | {len(test_pos)} | {test_pos[:,0].mean():.3f} | "
                f"{test_pos[:,1].mean():.3f} | {test_indoor.mean():.3f} |\n\n")
        f.write("## Top Splits\n\n")
        f.write("| rank | seed | total | xy_q | xy_ms | nn_q | grid_tv | "
                "grid_miss | indoor | overlap_seed0 |\n")
        f.write("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for rank, r in enumerate(rows[:20], 1):
            f.write(f"| {rank} | {r.seed} | {r.total:.5f} | {r.xy_quantile:.5f} | "
                    f"{r.xy_mean_std:.5f} | {r.nn_quantile:.5f} | "
                    f"{r.grid_tv:.5f} | {r.grid_miss:.5f} | {r.indoor:.5f} | "
                    f"{r.overlap_with_seed0} |\n")
        f.write("\n")
        f.write("## Seed 0 Bridge\n\n")
        f.write(f"seed0 rank: {seed0_rank}/{len(rows)}, total={seed0.total:.5f}. "
                "Keep it as the historical bridge to GS8/GS9 scores, but do "
                "not use it as the only tuning target.\n\n")
        f.write("## Recommended Panel\n\n")
        f.write("| slot | seed | total | overlap_seed0 |\n")
        f.write("| ---: | ---: | ---: | ---: |\n")
        for i, r in enumerate(panel, 1):
            f.write(f"| {i} | {r.seed} | {r.total:.5f} | {r.overlap_with_seed0} |\n")
        f.write("\nUse the panel by taking median/worst split score for future "
                "alpha/weight choices. Do not pick a split because a channel "
                "model scores high on it.\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datadir", default="Round1_Map(2)")
    ap.add_argument("--max-seed", type=int, default=5000)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--grid", type=int, default=8)
    ap.add_argument("--panel-size", type=int, default=5)
    ap.add_argument("--panel-max-overlap", type=int, default=35)
    ap.add_argument("--outdir", default="docs")
    args = ap.parse_args()

    datadir = Path(args.datadir)
    outdir = Path(args.outdir)
    train_pos, test_pos, tag = load_positions(datadir)
    train_indoor = indoor_flags(datadir, tag, train_pos)
    test_indoor = indoor_flags(datadir, tag, test_pos)

    xy_all = np.concatenate([train_pos[:, :2], test_pos[:, :2]], axis=0)
    spans = np.maximum(xy_all.max(0) - xy_all.min(0), 1e-6)
    pad = 1e-6
    x_edges = np.linspace(xy_all[:, 0].min() - pad, xy_all[:, 0].max() + pad,
                          args.grid + 1)
    y_edges = np.linspace(xy_all[:, 1].min() - pad, xy_all[:, 1].max() + pad,
                          args.grid + 1)
    test_hist = hist2d(test_pos[:, :2], x_edges, y_edges)

    tree_train = cKDTree(train_pos[:, :2])
    # k=64 is far beyond what random 10% validation removal should need.
    train_neighbor_dist, train_neighbor_idx = tree_train.query(train_pos[:, :2],
                                                               k=64)
    test_nn_dist, _ = tree_train.query(test_pos[:, :2], k=1)

    seed0_set = set(reproduce_val_indices(len(train_pos), args.val_fraction, 0))
    rows = [
        score_seed(seed, train_pos, test_pos, train_indoor, test_indoor, spans,
                   test_hist, x_edges, y_edges, train_neighbor_idx,
                   train_neighbor_dist, test_nn_dist, seed0_set,
                   args.val_fraction)
        for seed in range(args.max_seed)
    ]
    rows.sort(key=lambda r: r.total)
    seed0 = next(r for r in rows if r.seed == 0)
    panel = choose_panel(rows, len(train_pos), args.val_fraction,
                         args.panel_size, args.panel_max_overlap)

    outdir.mkdir(parents=True, exist_ok=True)
    best = rows[0]
    np.save(outdir / f"val_indices_representative_seed{best.seed}.npy",
            np.array(sorted(reproduce_val_indices(
                len(train_pos), args.val_fraction, best.seed)), dtype=np.int64))
    np.save(outdir / "val_indices_representative_panel.npy",
            np.array([
                sorted(reproduce_val_indices(
                    len(train_pos), args.val_fraction, row.seed))
                for row in panel
            ], dtype=np.int64))
    write_report(outdir, rows, panel, seed0, args.max_seed, args.val_fraction,
                 train_pos, test_pos, train_indoor, test_indoor)

    print("best seed:", best.seed, "score=%.5f" % best.total)
    print("seed0 rank:", next(i for i, r in enumerate(rows, 1) if r.seed == 0),
          "score=%.5f" % seed0.total)
    print("panel:", ", ".join(f"{r.seed}(score={r.total:.5f})" for r in panel))
    print("wrote:", outdir / "validation_split_search.md")


if __name__ == "__main__":
    main()
