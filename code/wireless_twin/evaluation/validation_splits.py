"""Deterministic validation split builders used by the v0.5 audit plan."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass(frozen=True)
class Split:
    name: str
    train_idx: np.ndarray
    val_idx: np.ndarray
    weight: float


def random_interpolation_split(n: int, val_fraction: float, seed: int) -> Split:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_val = max(1, int(round(n * val_fraction)))
    val = np.sort(order[:n_val])
    train = np.sort(order[n_val:])
    return Split("random_interpolation", train, val, 0.2)


def spatial_block_folds(
    positions: np.ndarray,
    n_folds: int = 3,
    seed: int = 0,
) -> Dict[str, Split]:
    """Create deterministic compact spatial validation folds."""
    pos = np.asarray(positions, dtype=np.float32)
    n = len(pos)
    rng = np.random.default_rng(seed)
    anchors = [int(rng.integers(n))]
    min_dist = np.sum((pos - pos[anchors[0]]) ** 2, axis=1)
    for _ in range(1, n_folds):
        anchors.append(int(np.argmax(min_dist)))
        dist = np.sum((pos - pos[anchors[-1]]) ** 2, axis=1)
        min_dist = np.minimum(min_dist, dist)
    dist_to_anchors = np.stack(
        [np.sum((pos - pos[a]) ** 2, axis=1) for a in anchors], axis=1)
    assignment = np.argmin(dist_to_anchors, axis=1)
    splits: Dict[str, Split] = {}
    for fold in range(n_folds):
        val = np.flatnonzero(assignment == fold)
        if len(val) == 0:
            val = np.asarray([anchors[fold]], dtype=np.int64)
        train = np.setdiff1d(np.arange(n), val, assume_unique=False)
        splits[f"spatial_block_{fold}"] = Split(
            f"spatial_block_{fold}", train.astype(np.int64), val.astype(np.int64),
            0.4 / n_folds)
    return splits


def power_stratified_split(
    channels: np.ndarray,
    val_fraction: float,
    seed: int,
    bins: int = 5,
) -> Split:
    powers = np.empty(len(channels), dtype=np.float64)
    for i in range(len(channels)):
        h = np.asarray(channels[i])
        powers[i] = float(np.vdot(h, h).real)
    rng = np.random.default_rng(seed)
    val_parts = []
    order = np.argsort(powers)
    for group in np.array_split(order, max(1, bins)):
        group = np.asarray(group, dtype=np.int64)
        rng.shuffle(group)
        n_val = max(1, int(round(len(group) * val_fraction)))
        val_parts.append(group[:n_val])
    val = np.sort(np.concatenate(val_parts))
    train = np.setdiff1d(np.arange(len(channels)), val, assume_unique=False)
    return Split("power_stratified", train.astype(np.int64), val.astype(np.int64), 0.1)


def test_matched_split(
    train_positions: np.ndarray,
    test_positions: Optional[np.ndarray],
    val_fraction: float,
) -> Optional[Split]:
    """Hold out train points closest to hidden-test positions in position space."""
    if test_positions is None or len(test_positions) == 0:
        return None
    train_pos = np.asarray(train_positions, dtype=np.float32)
    test_pos = np.asarray(test_positions, dtype=np.float32)
    n_val = max(1, int(round(len(train_pos) * val_fraction)))
    best = np.full(len(train_pos), np.inf, dtype=np.float64)
    for start in range(0, len(test_pos), 128):
        diff = train_pos[:, None, :] - test_pos[None, start:start + 128, :]
        dist2 = np.sum(diff * diff, axis=-1)
        best = np.minimum(best, dist2.min(axis=1))
    val = np.argsort(best)[:n_val]
    train = np.setdiff1d(np.arange(len(train_pos)), val, assume_unique=False)
    return Split("test_matched", train.astype(np.int64), np.sort(val).astype(np.int64), 0.3)


def build_validation_splits(
    train_positions: np.ndarray,
    channels: np.ndarray,
    test_positions: Optional[np.ndarray],
    val_fraction: float = 0.1,
    seed: int = 0,
) -> Dict[str, Split]:
    splits: Dict[str, Split] = {
        "random_interpolation": random_interpolation_split(
            len(train_positions), val_fraction, seed)
    }
    splits.update(spatial_block_folds(train_positions, n_folds=3, seed=seed))
    matched = test_matched_split(train_positions, test_positions, val_fraction)
    if matched is not None:
        splits["test_matched"] = matched
    splits["power_stratified"] = power_stratified_split(
        channels, val_fraction, seed)
    return splits


def split_summary(
    name: str,
    val_idx: np.ndarray,
    positions: np.ndarray,
    channels: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    pos = np.asarray(positions, dtype=np.float32)
    vp = pos[val_idx]
    out: Dict[str, float] = {
        "n_val": float(len(val_idx)),
        "x_min": float(vp[:, 0].min()),
        "x_max": float(vp[:, 0].max()),
        "y_min": float(vp[:, 1].min()),
        "y_max": float(vp[:, 1].max()),
        "z_min": float(vp[:, 2].min()),
        "z_max": float(vp[:, 2].max()),
    }
    if len(val_idx) > 1:
        diff = vp[:, None, :] - vp[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=-1))
        dist[dist == 0] = np.nan
        out["nearest_val_distance_median"] = float(np.nanmedian(dist))
    else:
        out["nearest_val_distance_median"] = 0.0
    if channels is not None:
        powers = []
        for idx in val_idx:
            h = np.asarray(channels[int(idx)])
            powers.append(float(np.vdot(h, h).real))
        p = np.asarray(powers, dtype=np.float64)
        out["power_mean"] = float(p.mean())
        out["power_p10"] = float(np.quantile(p, 0.1))
        out["power_p90"] = float(np.quantile(p, 0.9))
    return out
