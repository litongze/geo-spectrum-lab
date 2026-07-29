#!/usr/bin/env python3
"""Fit general leakage-free BS/UE array steering phase corrections."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from _bootstrap import ROOT  # noqa: F401
from probe_array_steering_phase import array_coordinates
from score_holdout import reproduce_val_indices
from wireless_twin.data.setup_config import load_setup


def receiver_vertical(spec, layout: str) -> np.ndarray:
    if layout == "phv":
        _, _, vertical = np.indices((spec.np, spec.nh, spec.nv))
    elif layout == "vph":
        vertical, _, _ = np.indices((spec.nv, spec.np, spec.nh))
    else:
        raise ValueError(layout)
    result = vertical.reshape(-1).astype(np.float64)
    result -= result.mean()
    return result


def array_steering_phase(
    delta_unit: np.ndarray,
    spec,
    steering: dict,
) -> np.ndarray:
    bs_layout = str(
        steering.get("bs_layout", steering.get("layout", "phv"))
    )
    horizontal, vertical = array_coordinates(spec, bs_layout)
    if "coefficients" in steering:
        coefficients = np.asarray(
            steering["coefficients"], dtype=np.float64
        )
        projection = np.einsum(
            "...d,td->...t", delta_unit, coefficients
        )
        phase = (
            projection[..., 0, None, None]
            * horizontal[None, None, :, None]
            + projection[..., 1, None, None]
            * vertical[None, None, :, None]
        )
        if len(coefficients) == 3:
            receiver = receiver_vertical(
                spec, str(steering["receiver_layout"])
            )
            phase = phase + (
                projection[..., 2, None, None]
                * receiver[None, None, None, :]
            )
        return phase
    axis = np.asarray(
        [
            np.cos(float(steering["theta"])),
            np.sin(float(steering["theta"])),
        ],
        dtype=np.float64,
    )
    delta_horizontal = np.einsum(
        "...d,d->...", delta_unit[..., :2], axis
    )
    return (
        float(steering["horizontal_coefficient"])
        * delta_horizontal[..., None, None]
        * horizontal[None, None, :, None]
        + float(steering["vertical_coefficient"])
        * delta_unit[..., 2, None, None]
        * vertical[None, None, :, None]
    )


def inverse_bound(value: np.ndarray, bound: float) -> np.ndarray:
    normalized = np.clip(value / bound, -0.999, 0.999)
    return np.arctanh(normalized).astype(np.float32)


def fit_general_steering(
    cross: np.ndarray,
    delta_unit: np.ndarray,
    horizontal: np.ndarray,
    vertical: np.ndarray,
    receiver: np.ndarray,
    current: dict,
    include_receiver: bool,
    device: torch.device,
    steps: int,
    starts: int,
    bound: float,
) -> dict:
    rng = np.random.default_rng(20260730)
    term_count = 3 if include_receiver else 2
    initial = np.zeros((starts, term_count, 3), dtype=np.float32)
    axis = np.asarray(
        [np.cos(current["theta"]), np.sin(current["theta"])],
        dtype=np.float32,
    )
    initial[:, 0, :2] = (
        float(current["horizontal_coefficient"]) * axis
    )
    initial[:, 1, 2] = float(current["vertical_coefficient"])
    if starts > 1:
        initial[1:] += rng.normal(
            0.0, np.pi, size=initial[1:].shape
        ).astype(np.float32)
    if starts > 3:
        initial[3::4] = rng.uniform(
            -np.pi, np.pi, size=initial[3::4].shape
        ).astype(np.float32)

    raw = torch.nn.Parameter(
        torch.as_tensor(
            inverse_bound(initial, bound),
            dtype=torch.float32,
            device=device,
        )
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
    receiver_t = torch.as_tensor(
        receiver, dtype=torch.float32, device=device
    )
    optimizer = torch.optim.Adam([raw], lr=0.04)
    normalizer = cross_t.abs().sum().clamp_min(1e-30)

    def objective_and_coefficients() -> tuple[torch.Tensor, torch.Tensor]:
        coefficients = bound * raw.tanh()
        projection = torch.einsum(
            "bd,jtd->jbt", delta_t, coefficients
        )
        phase = (
            projection[:, :, 0, None, None]
            * horizontal_t[None, None, :, None]
            + projection[:, :, 1, None, None]
            * vertical_t[None, None, :, None]
        )
        if include_receiver:
            phase = phase + (
                projection[:, :, 2, None, None]
                * receiver_t[None, None, None, :]
            )
        total = (
            cross_t[None] * torch.exp(-1j * phase)
        ).sum(dim=(1, 2, 3))
        return total.abs() / normalizer, coefficients

    for _ in range(steps):
        objective, _ = objective_and_coefficients()
        optimizer.zero_grad()
        (-objective.sum()).backward()
        optimizer.step()

    with torch.inference_mode():
        objective, coefficients = objective_and_coefficients()
        best = int(objective.argmax())
    return {
        "objective": float(objective[best]),
        "coefficients": coefficients[best].cpu().numpy(),
    }


def corrected_cross(
    cross: np.ndarray,
    delta_unit: np.ndarray,
    horizontal: np.ndarray,
    vertical: np.ndarray,
    receiver: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    projection = np.einsum("bd,td->bt", delta_unit, coefficients)
    phase = (
        projection[:, 0, None, None] * horizontal[None, :, None]
        + projection[:, 1, None, None] * vertical[None, :, None]
    )
    if len(coefficients) == 3:
        phase = phase + (
            projection[:, 2, None, None]
            * receiver[None, None, :]
        )
    return (cross * np.exp(-1j * phase)).sum(axis=(1, 2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", default="Round1_Map(2)")
    parser.add_argument(
        "--config", default="configs/gs37_geophase_moment.json"
    )
    parser.add_argument(
        "--steering-result",
        default="docs/array_steering_phase_bound64/result.json",
    )
    parser.add_argument("--tune-seeds", default="1890,3716,962,1022")
    parser.add_argument("--audit-seed", type=int, default=2262)
    parser.add_argument("--receiver-layouts", default="phv,vph")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--starts", type=int, default=12)
    parser.add_argument("--bound", type=float, default=64.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out", default="docs/full_array_steering_phase/result.json"
    )
    args = parser.parse_args()

    tune_seeds = [
        int(value) for value in args.tune_seeds.split(",") if value
    ]
    seeds = tune_seeds + [args.audit_seed]
    tune_union = set().union(
        *[
            set(reproduce_val_indices(2000, 0.1, seed))
            for seed in tune_seeds
        ]
    )
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    k0 = float(config["phase"]["k0_rad_per_meter"])
    k1 = float(
        config["phase"]["k1_rad_per_meter_per_subcarrier"]
    )
    current = json.loads(
        Path(args.steering_result).read_text(encoding="utf-8")
    )["ranked"][0]

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
            (len(val_idx), spec.m, spec.n), dtype=np.complex64
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
                    (source.conj() * truth).sum(dim=3).cpu().numpy()
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
            "strict_keep": np.asarray(
                [int(index) not in tune_union for index in val_idx],
                dtype=bool,
            ),
        }
        print(f"[full-steering] split={seed} prepared", flush=True)

    tune_cross = np.concatenate(
        [split_data[seed]["cross"] for seed in tune_seeds], axis=0
    )
    tune_delta = np.concatenate(
        [split_data[seed]["delta_unit"] for seed in tune_seeds],
        axis=0,
    )
    horizontal, vertical = array_coordinates(spec, "phv")
    ranked = []
    receiver_layouts = [
        value for value in args.receiver_layouts.split(",") if value
    ]
    for layout_index, receiver_layout in enumerate(receiver_layouts):
        receiver = receiver_vertical(spec, receiver_layout)
        for include_receiver in (False, True):
            if not include_receiver and layout_index > 0:
                continue
            fitted = fit_general_steering(
                tune_cross,
                tune_delta,
                horizontal,
                vertical,
                receiver,
                current,
                include_receiver,
                device,
                args.steps,
                args.starts,
                args.bound,
            )
            coefficients = fitted["coefficients"]
            per_point_cross = {
                seed: corrected_cross(
                    split_data[seed]["cross"],
                    split_data[seed]["delta_unit"],
                    horizontal,
                    vertical,
                    receiver,
                    coefficients,
                )
                for seed in seeds
            }
            total_cross = sum(
                per_point_cross[seed].sum() for seed in tune_seeds
            )
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
                source_energy = float(data["source_energy"].sum())
                truth_energy = float(data["truth_energy"].sum())
                cross_sum = per_point_cross[seed].sum()
                nmse = (
                    truth_energy
                    + scale**2 * source_energy
                    - 2.0
                    * scale
                    * float(np.real(cross_sum * phase_factor))
                ) / truth_energy
                keep = data["strict_keep"]
                strict_nmse = None
                if keep.any():
                    strict_source_energy = float(
                        data["source_energy"][keep].sum()
                    )
                    strict_truth_energy = float(
                        data["truth_energy"][keep].sum()
                    )
                    strict_cross = per_point_cross[seed][keep].sum()
                    strict_nmse = (
                        strict_truth_energy
                        + scale**2 * strict_source_energy
                        - 2.0
                        * scale
                        * float(np.real(strict_cross * phase_factor))
                    ) / strict_truth_energy
                scores[str(seed)] = {
                    "NMSE": float(nmse),
                    "strict_NMSE": (
                        float(strict_nmse)
                        if strict_nmse is not None
                        else None
                    ),
                }
            tune_nmse = [
                scores[str(seed)]["NMSE"] for seed in tune_seeds
            ]
            ranked.append(
                {
                    "name": (
                        f"bs_general_{receiver_layout}"
                        + ("_ue" if include_receiver else "")
                    ),
                    "bs_layout": "phv",
                    "receiver_layout": receiver_layout,
                    "include_receiver": include_receiver,
                    "objective": fitted["objective"],
                    "coefficients": coefficients.tolist(),
                    "global_phase": global_phase,
                    "scale": scale,
                    "tune_median_nmse": float(np.median(tune_nmse)),
                    "tune_mean_nmse": float(np.mean(tune_nmse)),
                    "tune_worst_nmse": float(np.max(tune_nmse)),
                    "audit_nmse": scores[str(args.audit_seed)]["NMSE"],
                    "strict_audit_nmse": scores[str(args.audit_seed)][
                        "strict_NMSE"
                    ],
                    "scores": scores,
                }
            )
            print(
                f"[full-steering] {ranked[-1]['name']} "
                f"tune={ranked[-1]['tune_median_nmse']:.6f} "
                f"audit={ranked[-1]['audit_nmse']:.6f} "
                f"strict={ranked[-1]['strict_audit_nmse']:.6f}",
                flush=True,
            )
    ranked.sort(
        key=lambda row: (
            row["tune_median_nmse"],
            row["tune_mean_nmse"],
        )
    )
    payload = {
        "selection_policy": (
            "coefficients selected on four tune splits; strict audit "
            "excludes all tune indices"
        ),
        "phase": {"k0": k0, "k1": k1},
        "bound": args.bound,
        "steps": args.steps,
        "starts": args.starts,
        "tune_seeds": tune_seeds,
        "audit_seed": args.audit_seed,
        "strict_audit_count": int(
            split_data[args.audit_seed]["strict_keep"].sum()
        ),
        "ranked": ranked,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("[RESULT]", json.dumps(ranked, ensure_ascii=False), flush=True)
    print(f"FULL_ARRAY_STEERING_DONE out={out}", flush=True)


if __name__ == "__main__":
    main()
