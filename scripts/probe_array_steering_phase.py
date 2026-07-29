#!/usr/bin/env python3
"""Fit a low-dimensional BS-array steering correction after range phase."""
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


def array_coordinates(spec, layout: str) -> tuple[np.ndarray, np.ndarray]:
    if layout == "phv":
        _, horizontal, vertical = np.indices(
            (spec.mp, spec.mh, spec.mv)
        )
    elif layout == "hvp":
        horizontal, vertical, _ = np.indices(
            (spec.mh, spec.mv, spec.mp)
        )
    elif layout == "pvh":
        _, vertical, horizontal = np.indices(
            (spec.mp, spec.mv, spec.mh)
        )
    else:
        raise ValueError(layout)
    horizontal = horizontal.reshape(-1).astype(np.float64)
    vertical = vertical.reshape(-1).astype(np.float64)
    horizontal -= horizontal.mean()
    vertical -= vertical.mean()
    return horizontal, vertical


def fit_steering(
    cross: np.ndarray,
    delta_unit: np.ndarray,
    horizontal: np.ndarray,
    vertical: np.ndarray,
    device: torch.device,
    steps: int,
    bound: float,
) -> dict:
    theta_init = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    coefficient_init = np.asarray(
        [-np.pi, 0.0, np.pi], dtype=np.float32
    )
    theta_grid, horizontal_grid, vertical_grid = np.meshgrid(
        theta_init,
        coefficient_init,
        coefficient_init,
        indexing="ij",
    )
    theta = torch.nn.Parameter(
        torch.as_tensor(
            theta_grid.reshape(-1),
            dtype=torch.float32,
            device=device,
        )
    )
    def inverse_bound(value: np.ndarray) -> torch.Tensor:
        normalized = np.clip(value / bound, -0.999, 0.999)
        return torch.as_tensor(
            np.arctanh(normalized),
            dtype=torch.float32,
            device=device,
        )

    horizontal_raw = torch.nn.Parameter(
        inverse_bound(horizontal_grid.reshape(-1))
    )
    vertical_raw = torch.nn.Parameter(
        inverse_bound(vertical_grid.reshape(-1))
    )
    optimizer = torch.optim.Adam(
        [theta, horizontal_raw, vertical_raw], lr=0.04
    )
    cross_t = torch.as_tensor(
        cross, dtype=torch.complex64, device=device
    )
    delta_t = torch.as_tensor(
        delta_unit, dtype=torch.float32, device=device
    )
    horizontal_t = torch.as_tensor(
        horizontal, dtype=torch.float32, device=device
    )
    vertical_t = torch.as_tensor(
        vertical, dtype=torch.float32, device=device
    )
    normalizer = cross_t.abs().sum().clamp_min(1e-30)

    for _ in range(steps):
        axis = torch.stack([theta.cos(), theta.sin()], dim=1)
        delta_horizontal = torch.einsum(
            "bd,jd->jb", delta_t[:, :2], axis
        )
        horizontal_coefficient = bound * horizontal_raw.tanh()
        vertical_coefficient = bound * vertical_raw.tanh()
        phase = (
            horizontal_coefficient[:, None, None]
            * delta_horizontal[:, :, None]
            * horizontal_t[None, None, :]
            + vertical_coefficient[:, None, None]
            * delta_t[None, :, 2:3]
            * vertical_t[None, None, :]
        )
        total = (
            cross_t[None] * torch.exp(-1j * phase)
        ).sum(dim=(1, 2))
        objective = total.abs() / normalizer
        loss = -objective.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.inference_mode():
        axis = torch.stack([theta.cos(), theta.sin()], dim=1)
        delta_horizontal = torch.einsum(
            "bd,jd->jb", delta_t[:, :2], axis
        )
        horizontal_coefficient = bound * horizontal_raw.tanh()
        vertical_coefficient = bound * vertical_raw.tanh()
        phase = (
            horizontal_coefficient[:, None, None]
            * delta_horizontal[:, :, None]
            * horizontal_t[None, None, :]
            + vertical_coefficient[:, None, None]
            * delta_t[None, :, 2:3]
            * vertical_t[None, None, :]
        )
        total = (
            cross_t[None] * torch.exp(-1j * phase)
        ).sum(dim=(1, 2))
        objective = total.abs()
        best = int(objective.argmax())
    return {
        "theta": float(theta[best].remainder(2.0 * np.pi)),
        "horizontal_coefficient": float(horizontal_coefficient[best]),
        "vertical_coefficient": float(vertical_coefficient[best]),
        "objective": float(objective[best]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", default="Round1_Map(2)")
    parser.add_argument(
        "--config", default="configs/gs37_geophase_moment.json"
    )
    parser.add_argument("--tune-seeds", default="1890,3716,962,1022")
    parser.add_argument("--audit-seed", type=int, default=2262)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--bound", type=float, default=8.0)
    parser.add_argument("--layouts", default="phv,hvp,pvh")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out", default="docs/array_steering_phase/result.json"
    )
    args = parser.parse_args()

    tune_seeds = [
        int(value) for value in args.tune_seeds.split(",") if value
    ]
    seeds = tune_seeds + [args.audit_seed]
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    k0 = float(config["phase"]["k0_rad_per_meter"])
    k1 = float(
        config["phase"]["k1_rad_per_meter_per_subcarrier"]
    )
    datadir = Path(args.datadir)
    spec = load_setup(datadir / "Round1_Setup.json")
    positions = np.load(
        datadir / "Round1_Train_Pos.npy"
    ).astype(np.float64)
    channels = np.load(
        datadir / "Round1_Train_Channel.npy", mmap_mode="r"
    )
    all_idx = np.arange(len(positions), dtype=np.int64)
    bs = np.asarray(spec.bs_position, dtype=np.float64)
    radius = np.linalg.norm(positions - bs[None], axis=1)
    unit = (positions - bs[None]) / np.maximum(
        radius[:, None], 1e-12
    )
    subcarrier = np.arange(spec.s, dtype=np.float64)
    subcarrier -= subcarrier.mean()
    device = torch.device(args.device)
    split_data = {}

    for seed in seeds:
        val_idx = np.asarray(
            sorted(reproduce_val_indices(len(positions), 0.1, seed)),
            dtype=np.int64,
        )
        pool_idx = np.setdiff1d(all_idx, val_idx)
        _, local = cKDTree(positions[pool_idx, :2]).query(
            positions[val_idx, :2], k=1
        )
        neighbor_idx = pool_idx[np.asarray(local)]
        delta_radius = radius[val_idx] - radius[neighbor_idx]
        cross = np.empty(
            (len(val_idx), spec.m), dtype=np.complex128
        )
        source_energy = np.empty(len(val_idx), dtype=np.float64)
        truth_energy = np.empty(len(val_idx), dtype=np.float64)
        with torch.inference_mode():
            for start in range(0, len(val_idx), args.batch_size):
                stop = min(start + args.batch_size, len(val_idx))
                source = torch.as_tensor(
                    np.array(
                        channels[neighbor_idx[start:stop]], copy=True
                    ),
                    dtype=torch.complex64,
                    device=device,
                )
                truth = torch.as_tensor(
                    np.array(channels[val_idx[start:stop]], copy=True),
                    dtype=torch.complex64,
                    device=device,
                )
                phase = (
                    (
                        k0 + k1 * subcarrier[None]
                    )
                    * delta_radius[start:stop, None]
                )
                source *= torch.as_tensor(
                    np.exp(1j * phase),
                    dtype=torch.complex64,
                    device=device,
                )[:, None, None, :]
                cross[start:stop] = (
                    (source.conj() * truth)
                    .sum(dim=(2, 3))
                    .cpu()
                    .numpy()
                )
                source_energy[start:stop] = (
                    source.abs()
                    .square()
                    .sum(dim=(1, 2, 3))
                    .cpu()
                    .numpy()
                )
                truth_energy[start:stop] = (
                    truth.abs()
                    .square()
                    .sum(dim=(1, 2, 3))
                    .cpu()
                    .numpy()
                )
        split_data[seed] = {
            "cross": cross,
            "delta_unit": unit[val_idx] - unit[neighbor_idx],
            "source_energy": source_energy,
            "truth_energy": truth_energy,
        }
        print(f"[steering] split={seed} prepared", flush=True)

    tune_cross = np.concatenate(
        [split_data[seed]["cross"] for seed in tune_seeds], axis=0
    )
    tune_delta = np.concatenate(
        [split_data[seed]["delta_unit"] for seed in tune_seeds],
        axis=0,
    )
    ranked = []
    for layout in [
        value for value in args.layouts.split(",") if value
    ]:
        horizontal, vertical = array_coordinates(spec, layout)
        fitted = fit_steering(
            tune_cross,
            tune_delta,
            horizontal,
            vertical,
            device,
            args.steps,
            args.bound,
        )
        axis = np.asarray(
            [np.cos(fitted["theta"]), np.sin(fitted["theta"])],
            dtype=np.float64,
        )
        corrected = {}
        for seed in seeds:
            data = split_data[seed]
            delta_horizontal = data["delta_unit"][:, :2] @ axis
            phase = (
                fitted["horizontal_coefficient"]
                * delta_horizontal[:, None]
                * horizontal[None]
                + fitted["vertical_coefficient"]
                * data["delta_unit"][:, 2:3]
                * vertical[None]
            )
            corrected[seed] = (
                data["cross"] * np.exp(-1j * phase)
            ).sum()
        total_cross = sum(corrected[seed] for seed in tune_seeds)
        total_source_energy = sum(
            split_data[seed]["source_energy"].sum()
            for seed in tune_seeds
        )
        global_phase = float(np.angle(total_cross))
        phase_factor = np.exp(-1j * global_phase)
        scale = max(
            float(np.real(total_cross * phase_factor))
            / max(total_source_energy, 1e-30),
            0.0,
        )
        scores = {}
        for seed in seeds:
            data = split_data[seed]
            source_energy = data["source_energy"].sum()
            truth_energy = data["truth_energy"].sum()
            cross_real = float(
                np.real(corrected[seed] * phase_factor)
            )
            nmse = (
                truth_energy
                + scale**2 * source_energy
                - 2.0 * scale * cross_real
            ) / max(truth_energy, 1e-30)
            coherence = (
                np.abs(corrected[seed])
                / np.sqrt(
                    max(source_energy * truth_energy, 1e-30)
                )
            )
            scores[str(seed)] = {
                "NMSE": float(nmse),
                "coherence": float(coherence),
            }
        tune_nmse = [
            scores[str(seed)]["NMSE"] for seed in tune_seeds
        ]
        ranked.append(
            {
                "layout": layout,
                **fitted,
                "global_phase": global_phase,
                "scale": scale,
                "tune_median_nmse": float(np.median(tune_nmse)),
                "tune_mean_nmse": float(np.mean(tune_nmse)),
                "tune_worst_nmse": float(np.max(tune_nmse)),
                "audit_nmse": scores[str(args.audit_seed)]["NMSE"],
                "scores": scores,
            }
        )
    ranked.sort(
        key=lambda row: (
            row["tune_median_nmse"],
            row["tune_mean_nmse"],
        )
    )
    payload = {
        "selection_policy": (
            "steering parameters selected on four tune splits; "
            "audit external"
        ),
        "tune_seeds": tune_seeds,
        "audit_seed": args.audit_seed,
        "phase": {"k0": k0, "k1": k1},
        "ranked": ranked,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("[RESULT]", json.dumps(ranked, ensure_ascii=False), flush=True)
    print(f"ARRAY_STEERING_PHASE_DONE out={out}", flush=True)


if __name__ == "__main__":
    main()
