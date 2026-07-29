#!/usr/bin/env python3
"""KNN PAS/PDP interpolation with projection-based channel reconstruction."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

import _bootstrap  # noqa: F401
from wireless_twin.data import load_round
from wireless_twin.evaluation.metrics import evaluate_channels


def _pas_np(h: np.ndarray, spec) -> np.ndarray:
    hm = h.reshape(spec.mh, spec.mv, spec.mp, spec.n, spec.s)
    hf = np.fft.fft2(hm, axes=(0, 1))
    power = np.abs(hf).astype(np.float32) ** 2
    power = power.sum(axis=2)
    return power.reshape(spec.mh * spec.mv, spec.n, spec.s).transpose(1, 2, 0)


def _pdp_np(h: np.ndarray) -> np.ndarray:
    hd = np.fft.ifft(h, axis=-1)
    return (np.abs(hd).astype(np.float32) ** 2)


def _enforce_pas(h: np.ndarray, pas: np.ndarray, spec, eps: float) -> np.ndarray:
    hm = h.reshape(spec.mh, spec.mv, spec.mp, spec.n, spec.s)
    hf = np.fft.fft2(hm, axes=(0, 1))
    cur = np.abs(hf).astype(np.float32) ** 2
    cur = cur.sum(axis=2).reshape(spec.mh * spec.mv, spec.n, spec.s)
    target = pas.transpose(2, 0, 1)
    scale = np.sqrt(target / np.maximum(cur, eps)).reshape(
        spec.mh, spec.mv, 1, spec.n, spec.s)
    hf = hf * scale
    h2 = np.fft.ifft2(hf, axes=(0, 1))
    return h2.reshape(spec.m, spec.n, spec.s).astype(np.complex64)


def _enforce_pdp(h: np.ndarray, pdp: np.ndarray, eps: float) -> np.ndarray:
    hd = np.fft.ifft(h, axis=-1)
    cur = np.abs(hd).astype(np.float32) ** 2
    scale = np.sqrt(pdp / np.maximum(cur, eps))
    hd = hd * scale
    return np.fft.fft(hd, axis=-1).astype(np.complex64)


def _reconstruct_from_spectra(
    pas: np.ndarray,
    pdp: np.ndarray,
    phase_source: np.ndarray,
    spec,
    n_iter: int,
    eps: float = 1e-12,
) -> np.ndarray:
    h = _enforce_pas(phase_source.astype(np.complex64), pas, spec, eps)
    for _ in range(n_iter):
        h = _enforce_pdp(h, pdp, eps)
        h = _enforce_pas(h, pas, spec, eps)
    return h.astype(np.complex64)


def _knn_indices(
    train_pos: np.ndarray,
    query_pos: np.ndarray,
    k: int,
    distance_power: float,
    weight_mode: str,
    sigma: Optional[float],
) -> tuple[np.ndarray, np.ndarray]:
    diff = train_pos[None, :, :] - query_pos[:, None, :]
    dist2 = np.sum(diff * diff, axis=-1)
    idx = np.argpartition(dist2, kth=min(k, train_pos.shape[0] - 1), axis=1)[:, :k]
    selected_dist2 = np.take_along_axis(dist2, idx, axis=1)
    order = np.argsort(selected_dist2, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    selected_dist2 = np.take_along_axis(selected_dist2, order, axis=1)
    if weight_mode == "inverse":
        weights = 1.0 / np.power(selected_dist2 + 1e-6, 0.5 * distance_power)
    elif weight_mode == "gaussian":
        if sigma is None or sigma <= 0:
            local_sigma2 = np.maximum(selected_dist2[:, -1:], 1e-6)
        else:
            local_sigma2 = float(sigma) ** 2
        weights = np.exp(-selected_dist2 / (2.0 * local_sigma2))
    else:
        raise ValueError(f"unknown weight_mode: {weight_mode}")
    weights = weights / weights.sum(axis=1, keepdims=True)
    return idx.astype(np.int64), weights.astype(np.float32)


def _predict_one(
    train_channels: np.ndarray,
    neighbor_idx: np.ndarray,
    weights: np.ndarray,
    spec,
    n_iter: int,
) -> np.ndarray:
    pas = None
    pdp = None
    for idx, weight in zip(neighbor_idx, weights):
        h = np.asarray(train_channels[int(idx)], dtype=np.complex64)
        p1 = _pas_np(h, spec)
        p2 = _pdp_np(h)
        pas = p1 * weight if pas is None else pas + p1 * weight
        pdp = p2 * weight if pdp is None else pdp + p2 * weight
    assert pas is not None and pdp is not None
    phase_source = np.asarray(train_channels[int(neighbor_idx[0])], dtype=np.complex64)
    return _reconstruct_from_spectra(
        pas.astype(np.float32), pdp.astype(np.float32), phase_source, spec, n_iter)


def _predict_one_ensemble(
    train_channels: np.ndarray,
    neighbor_idx: np.ndarray,
    weights: np.ndarray,
    spec,
    n_iter: int,
    k_values: list[int],
) -> np.ndarray:
    acc = None
    for k in k_values:
        pred = _predict_one(
            train_channels,
            neighbor_idx[:k],
            weights[:k] / weights[:k].sum(),
            spec,
            n_iter,
        )
        acc = pred if acc is None else acc + pred
    assert acc is not None
    return (acc / float(len(k_values))).astype(np.complex64)


def _predict(
    train_pos: np.ndarray,
    train_channels: np.ndarray,
    query_pos: np.ndarray,
    spec,
    k_values: list[int],
    n_iter: int,
    scale: float,
    distance_power: float,
    weight_mode: str,
    sigma: Optional[float],
    exclude_idx: Optional[np.ndarray] = None,
) -> np.ndarray:
    pool_idx = np.arange(len(train_pos), dtype=np.int64)
    if exclude_idx is not None:
        pool_idx = np.setdiff1d(pool_idx, exclude_idx.astype(np.int64), assume_unique=False)
    pool_pos = train_pos[pool_idx]
    k = max(k_values)
    local_knn, weights = _knn_indices(
        pool_pos, query_pos, k, distance_power, weight_mode, sigma)
    knn = pool_idx[local_knn]
    out = np.empty((len(query_pos), spec.m, spec.n, spec.s), dtype=np.complex64)
    for i in range(len(query_pos)):
        if len(k_values) == 1:
            pred = _predict_one(train_channels, knn[i], weights[i], spec, n_iter)
        else:
            pred = _predict_one_ensemble(
                train_channels, knn[i], weights[i], spec, n_iter, k_values)
        out[i] = pred * scale
        if (i + 1) % 25 == 0 or i + 1 == len(query_pos):
            print(f"[spectrum_knn] predicted {i + 1}/{len(query_pos)}")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--val-indices", default=None)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument(
        "--k-values",
        default=None,
        help="Comma-separated K values to ensemble, e.g. 4,5,6. Overrides --k.",
    )
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument(
        "--distance-power",
        type=float,
        default=2.0,
        help="KNN distance weighting exponent. Default 2.0 preserves old 1/dist^2 behavior.",
    )
    parser.add_argument(
        "--weight-mode",
        choices=("inverse", "gaussian"),
        default="inverse",
        help="KNN weighting mode. Default inverse preserves previous behavior.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="Gaussian distance sigma in raw position units. Defaults to adaptive kth-neighbor distance.",
    )
    parser.add_argument("--chunk", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.k_values:
        k_values = sorted({int(v) for v in args.k_values.split(",") if v.strip()})
    else:
        k_values = [int(args.k)]
    if not k_values or min(k_values) <= 0:
        raise ValueError("K values must be positive")
    rd = load_round(args.datadir, load_test=True, mmap_channels=True)
    raw_train_pos = rd.train.positions * rd.pos_std + rd.pos_mean
    train_channels = rd.train.channels

    if args.val_indices:
        val_idx = np.load(args.val_indices).astype(np.int64)
        pred = _predict(
            raw_train_pos,
            train_channels,
            raw_train_pos[val_idx],
            rd.spec,
            k_values,
            args.iters,
            args.scale,
            args.distance_power,
            args.weight_mode,
            args.sigma,
            exclude_idx=val_idx,
        )
        gt = np.asarray(train_channels[val_idx])
        scores = evaluate_channels(pred, gt, rd.spec)
        for key, value in scores.items():
            print(f"{key}: {value:.9f}")
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            np.save(args.out, pred)
            print(f"[spectrum_knn] wrote {args.out}")
        return

    if rd.test_positions is None:
        raise ValueError("test positions are unavailable")
    pred = _predict(
        raw_train_pos,
        train_channels,
        rd.test_positions,
        rd.spec,
        k_values,
        args.iters,
        args.scale,
        args.distance_power,
        args.weight_mode,
        args.sigma,
    )
    out = Path(args.out) if args.out else Path(args.datadir) / f"{rd.round_tag}_Test_Channel.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, pred.astype(np.complex64))
    print(f"[spectrum_knn] wrote {out}; shape={pred.shape}, dtype={pred.dtype}")


if __name__ == "__main__":
    main()
