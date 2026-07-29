#!/usr/bin/env python3
"""Fit geometry-driven carrier and subcarrier phase ramps cleanly."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from _bootstrap import ROOT  # noqa: F401
from score_holdout import reproduce_val_indices
from wireless_twin.data.setup_config import load_setup


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", default="Round1_Map(2)")
    parser.add_argument("--tune-seeds", default="1890,3716,962,1022")
    parser.add_argument("--audit-seed", type=int, default=2262)
    parser.add_argument("--k0-min", type=float, default=-160.0)
    parser.add_argument("--k0-max", type=float, default=160.0)
    parser.add_argument("--k0-count", type=int, default=1281)
    parser.add_argument("--k1-min", type=float, default=-0.01)
    parser.add_argument("--k1-max", type=float, default=0.01)
    parser.add_argument("--k1-count", type=int, default=161)
    parser.add_argument(
        "--out", default="docs/geometric_phase/result.json"
    )
    args = parser.parse_args()

    datadir = Path(args.datadir)
    spec = load_setup(datadir / "Round1_Setup.json")
    positions = np.load(
        datadir / "Round1_Train_Pos.npy"
    ).astype(np.float64)
    channels = np.load(
        datadir / "Round1_Train_Channel.npy", mmap_mode="r"
    )
    tune_seeds = [
        int(value) for value in args.tune_seeds.split(",") if value
    ]
    seeds = tune_seeds + [args.audit_seed]
    all_idx = np.arange(len(positions), dtype=np.int64)
    bs = np.asarray(spec.bs_position, dtype=np.float64)
    radius = np.linalg.norm(positions - bs[None], axis=1)
    split_data = {}

    for seed in seeds:
        val_idx = np.asarray(
            sorted(reproduce_val_indices(len(positions), 0.1, seed)),
            dtype=np.int64,
        )
        pool_idx = np.setdiff1d(all_idx, val_idx)
        distance, local = cKDTree(positions[pool_idx, :2]).query(
            positions[val_idx, :2], k=1
        )
        neighbor_idx = pool_idx[np.asarray(local)]
        source = np.array(channels[neighbor_idx], copy=True)
        truth = np.array(channels[val_idx], copy=True)
        cross = np.einsum(
            "bmns,bmns->bs", source.conj(), truth, optimize=True
        ).astype(np.complex128)
        source_energy = np.square(np.abs(source), dtype=np.float64).sum(
            axis=(1, 2, 3)
        )
        truth_energy = np.square(np.abs(truth), dtype=np.float64).sum(
            axis=(1, 2, 3)
        )
        split_data[seed] = {
            "cross": cross,
            "delta_radius": radius[val_idx] - radius[neighbor_idx],
            "distance": np.asarray(distance, dtype=np.float64),
            "source_energy": source_energy,
            "truth_energy": truth_energy,
        }
        print(
            f"[phase] split={seed} nearest_mean={distance.mean():.4f}",
            flush=True,
        )

    tune_cross = np.concatenate(
        [split_data[seed]["cross"] for seed in tune_seeds]
    )
    tune_delta = np.concatenate(
        [split_data[seed]["delta_radius"] for seed in tune_seeds]
    )
    subcarrier = np.arange(spec.s, dtype=np.float64)
    subcarrier -= subcarrier.mean()
    k0_grid = np.linspace(
        args.k0_min, args.k0_max, args.k0_count
    )
    k1_grid = np.linspace(
        args.k1_min, args.k1_max, args.k1_count
    )
    best = None
    for k1 in k1_grid:
        slope_phase = np.exp(
            -1j
            * k1
            * tune_delta[:, None]
            * subcarrier[None]
        )
        collapsed = (tune_cross * slope_phase).sum(axis=1)
        carrier_phase = np.exp(
            -1j * tune_delta[:, None] * k0_grid[None]
        )
        objective = np.abs(
            (collapsed[:, None] * carrier_phase).sum(axis=0)
        )
        index = int(objective.argmax())
        row = (
            float(objective[index]),
            float(k0_grid[index]),
            float(k1),
        )
        if best is None or row[0] > best[0]:
            best = row
    assert best is not None
    _, selected_k0, selected_k1 = best

    def corrected_cross(data: dict) -> tuple[np.ndarray, complex]:
        phase = np.exp(
            -1j
            * (
                selected_k0
                + selected_k1 * subcarrier[None]
            )
            * data["delta_radius"][:, None]
        )
        per_point = (data["cross"] * phase).sum(axis=1)
        total = per_point.sum()
        return per_point, total

    tune_total = sum(
        corrected_cross(split_data[seed])[1] for seed in tune_seeds
    )
    tune_source_energy = sum(
        split_data[seed]["source_energy"].sum()
        for seed in tune_seeds
    )
    global_phase = float(np.angle(tune_total))
    complex_scale = (
        np.abs(tune_total) / max(tune_source_energy, 1e-30)
    )

    thresholds = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 1e9]
    threshold_scales = {}
    for threshold in thresholds:
        selected_cross = 0.0 + 0.0j
        selected_energy = 0.0
        for seed in tune_seeds:
            data = split_data[seed]
            per_point, _ = corrected_cross(data)
            keep = data["distance"] <= threshold
            selected_cross += per_point[keep].sum()
            selected_energy += data["source_energy"][keep].sum()
        phase_aligned = selected_cross * np.exp(-1j * global_phase)
        threshold_scales[str(threshold)] = max(
            float(phase_aligned.real) / max(selected_energy, 1e-30),
            0.0,
        )

    split_results = {}
    for seed in seeds:
        data = split_data[seed]
        per_point, total = corrected_cross(data)
        source_energy = data["source_energy"].sum()
        truth_energy = data["truth_energy"].sum()
        raw_total = data["cross"].sum()
        split_row = {
            "nearest_distance_mean": float(data["distance"].mean()),
            "raw_coherence": float(
                np.abs(raw_total)
                / np.sqrt(max(source_energy * truth_energy, 1e-30))
            ),
            "corrected_coherence": float(
                np.abs(total)
                / np.sqrt(max(source_energy * truth_energy, 1e-30))
            ),
            "fixed_phase_real_coherence": float(
                (
                    total * np.exp(-1j * global_phase)
                ).real
                / np.sqrt(max(source_energy * truth_energy, 1e-30))
            ),
            "fixed_scale_nmse": float(
                (
                    truth_energy
                    + complex_scale**2 * source_energy
                    - 2.0
                    * complex_scale
                    * (total * np.exp(-1j * global_phase)).real
                )
                / max(truth_energy, 1e-30)
            ),
            "thresholds": {},
        }
        for threshold in thresholds:
            keep = data["distance"] <= threshold
            selected_source = data["source_energy"][keep].sum()
            selected_cross = per_point[keep].sum()
            scale = threshold_scales[str(threshold)]
            nmse = (
                truth_energy
                + scale**2 * selected_source
                - 2.0
                * scale
                * (
                    selected_cross * np.exp(-1j * global_phase)
                ).real
            ) / max(truth_energy, 1e-30)
            split_row["thresholds"][str(threshold)] = {
                "fraction": float(keep.mean()),
                "scale": scale,
                "nmse": float(nmse),
            }
        split_results[str(seed)] = split_row

    payload = {
        "selection_policy": "phase ramp fitted on four tune splits",
        "tune_seeds": tune_seeds,
        "audit_seed": args.audit_seed,
        "selected": {
            "k0_rad_per_meter": selected_k0,
            "k1_rad_per_meter_per_subcarrier": selected_k1,
            "global_phase": global_phase,
            "complex_scale": float(complex_scale),
        },
        "threshold_scales": threshold_scales,
        "splits": split_results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        "[selected]",
        json.dumps(payload["selected"], ensure_ascii=False),
        flush=True,
    )
    print(
        "[splits]",
        json.dumps(split_results, ensure_ascii=False),
        flush=True,
    )
    print(f"GEOMETRIC_PHASE_DONE out={out}", flush=True)


if __name__ == "__main__":
    main()
