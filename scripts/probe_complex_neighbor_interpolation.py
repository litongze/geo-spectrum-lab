#!/usr/bin/env python3
"""Probe leakage-free complex interpolation of phase-aligned neighbors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from _bootstrap import ROOT  # noqa: F401
from probe_full_array_steering_phase import array_steering_phase
from score_holdout import reproduce_val_indices
from wireless_twin.data.setup_config import load_setup


def normalize_real(weight: np.ndarray) -> np.ndarray:
    denominator = weight.sum(axis=1, keepdims=True)
    return weight / np.where(
        np.abs(denominator) > 1e-12, denominator, 1.0
    )


def equalize_amplitude(
    coefficient: np.ndarray,
    gram: np.ndarray,
) -> np.ndarray:
    energy = np.maximum(
        np.real(np.diagonal(gram, axis1=1, axis2=2)),
        1e-30,
    )
    amplitude = np.sqrt(energy)
    magnitude = np.abs(coefficient)
    target = (magnitude * amplitude).sum(axis=1, keepdims=True)
    target /= np.maximum(magnitude.sum(axis=1, keepdims=True), 1e-30)
    return coefficient * target / amplitude


def phase_corrections(
    gram: np.ndarray,
    base: np.ndarray,
) -> dict[str, np.ndarray]:
    energy = np.maximum(
        np.real(np.diagonal(gram, axis1=1, axis2=2)),
        1e-30,
    )
    normalized = gram / np.sqrt(
        energy[:, :, None] * energy[:, None, :]
    )
    nearest_cross = normalized[:, 0, :]
    nearest_angle = -np.angle(nearest_cross)
    nearest_confidence = np.clip(np.abs(nearest_cross), 0.0, 1.0)

    weighted = (
        np.sqrt(np.maximum(base, 0.0))[:, :, None]
        * normalized
        * np.sqrt(np.maximum(base, 0.0))[:, None, :]
    )
    _, eigenvectors = np.linalg.eigh(weighted)
    principal = eigenvectors[:, :, -1]
    principal *= np.exp(
        -1j * np.angle(principal[:, :1])
    )
    eigen_angle = np.angle(principal)

    result = {"none": np.ones_like(normalized[:, :, 0])}
    for strength in (0.25, 0.5, 0.75, 1.0):
        result[f"near_a{strength:g}"] = np.exp(
            1j * strength * nearest_angle
        )
        result[f"nearconf_a{strength:g}"] = np.exp(
            1j * strength * nearest_confidence * nearest_angle
        )
        result[f"eigen_a{strength:g}"] = np.exp(
            1j * strength * eigen_angle
        )
    return result


def moment_corrected_weights(
    base: np.ndarray,
    radial_delta: np.ndarray,
    tangent_delta: np.ndarray,
) -> dict[str, np.ndarray]:
    scale = np.maximum(
        np.median(
            np.sqrt(radial_delta**2 + tangent_delta**2),
            axis=1,
            keepdims=True,
        ),
        0.3,
    )
    delta = np.stack(
        [radial_delta / scale, tangent_delta / scale], axis=1
    )
    moment = np.einsum("bdk,bk->bd", delta, base)
    covariance = np.einsum(
        "bdk,bk,bek->bde", delta, base, delta
    )
    eye = np.eye(2, dtype=np.float64)[None]
    result = {}
    for ridge in (0.03, 0.1, 0.3):
        solved = np.linalg.solve(
            covariance + ridge * eye,
            moment[..., None],
        )[..., 0]
        correction = base * np.einsum(
            "bdk,bd->bk", delta, solved
        )
        for strength in (0.25, 0.5, 0.75, 1.0):
            candidate = base - strength * correction
            result[
                f"moment_r{ridge:g}_a{strength:g}"
            ] = normalize_real(candidate)
    return result


def geometry_weights(
    distance: np.ndarray,
    radial_delta: np.ndarray,
    tangent_delta: np.ndarray,
) -> dict[str, np.ndarray]:
    result = {}
    k_max = distance.shape[1]
    for k in (8, 12, 16, 24, 32):
        if k > k_max:
            continue
        mask = np.zeros_like(distance)
        mask[:, :k] = 1.0
        for ratio in (4.0, 5.0, 6.0, 8.0):
            effective = np.sqrt(
                (ratio * radial_delta) ** 2 + tangent_delta**2
            )
            for power in (1.5, 2.0, 2.5):
                base = mask / np.maximum(effective, 0.05) ** power
                name = f"k{k}_r{ratio:g}_p{power:g}"
                base = normalize_real(base)
                result[name] = base
                for suffix, candidate in moment_corrected_weights(
                    base, radial_delta, tangent_delta
                ).items():
                    result[f"{name}_{suffix}"] = candidate
    return result


def parse_split_indices(
    size: int,
    seed: int | str,
    external_name: str,
    external_indices: str | None,
) -> np.ndarray:
    if seed == external_name:
        if not external_indices:
            raise ValueError("external split requested without indices")
        return np.asarray(
            sorted(
                np.load(external_indices).astype(np.int64).tolist()
            ),
            dtype=np.int64,
        )
    return np.asarray(
        sorted(reproduce_val_indices(size, 0.1, int(seed))),
        dtype=np.int64,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", default="Round1_Map(2)")
    parser.add_argument(
        "--config", default="configs/gs39_steer_k16_r5.json"
    )
    parser.add_argument("--tune-seeds", default="1890,3716,962,1022")
    parser.add_argument("--audit-seed", type=int, default=2262)
    parser.add_argument("--external-indices")
    parser.add_argument("--external-name", default="testmatched")
    parser.add_argument("--k-max", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out",
        default="docs/complex_neighbor_interpolation/result.json",
    )
    args = parser.parse_args()

    tune_seeds = [
        int(value) for value in args.tune_seeds.split(",") if value
    ]
    split_keys: list[int | str] = tune_seeds + [args.audit_seed]
    if args.external_indices:
        split_keys.append(args.external_name)

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    k0 = float(config["phase"]["k0_rad_per_meter"])
    k1 = float(config["phase"]["k1_rad_per_meter_per_subcarrier"])
    steering = config["steering"]

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
    for split in split_keys:
        val_idx = parse_split_indices(
            len(positions),
            split,
            args.external_name,
            args.external_indices,
        )
        pool_idx = np.setdiff1d(all_idx, val_idx)
        distance, local = cKDTree(positions[pool_idx, :2]).query(
            positions[val_idx, :2], k=args.k_max
        )
        distance = np.asarray(distance, dtype=np.float64)
        neighbor_idx = pool_idx[np.asarray(local)]
        delta_xy = (
            positions[neighbor_idx, :2]
            - positions[val_idx, None, :2]
        )
        radial = positions[val_idx, :2] - bs[None, :2]
        radial /= np.maximum(
            np.linalg.norm(radial, axis=1, keepdims=True), 1e-12
        )
        tangent = np.column_stack([-radial[:, 1], radial[:, 0]])
        radial_delta = np.einsum("bkd,bd->bk", delta_xy, radial)
        tangent_delta = np.einsum("bkd,bd->bk", delta_xy, tangent)
        delta_radius = radius[val_idx, None] - radius[neighbor_idx]
        delta_unit = unit[val_idx, None] - unit[neighbor_idx]

        cross = np.empty(
            (len(val_idx), args.k_max), dtype=np.complex128
        )
        gram = np.empty(
            (len(val_idx), args.k_max, args.k_max),
            dtype=np.complex128,
        )
        truth_energy = np.empty(len(val_idx), dtype=np.float64)
        with torch.inference_mode():
            for start in range(0, len(val_idx), args.batch_size):
                stop = min(start + args.batch_size, len(val_idx))
                batch = stop - start
                source = torch.as_tensor(
                    np.array(
                        channels[neighbor_idx[start:stop].reshape(-1)],
                        copy=True,
                    ).reshape(
                        batch,
                        args.k_max,
                        spec.m,
                        spec.n,
                        spec.s,
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
                    k0 + k1 * subcarrier[None, None]
                ) * delta_radius[start:stop, :, None]
                source *= torch.as_tensor(
                    np.exp(1j * phase),
                    dtype=torch.complex64,
                    device=device,
                )[:, :, None, None, :]
                steering_phase = array_steering_phase(
                    delta_unit[start:stop], spec, steering
                )
                source *= torch.as_tensor(
                    np.exp(1j * steering_phase),
                    dtype=torch.complex64,
                    device=device,
                )[..., None]

                flat = source.reshape(batch, args.k_max, -1)
                truth_flat = truth.reshape(batch, -1)
                cross[start:stop] = (
                    torch.einsum(
                        "bkl,bl->bk", flat.conj(), truth_flat
                    )
                    .cpu()
                    .numpy()
                )
                gram[start:stop] = (
                    torch.einsum(
                        "bkl,bjl->bkj", flat.conj(), flat
                    )
                    .cpu()
                    .numpy()
                )
                truth_energy[start:stop] = (
                    truth_flat.abs()
                    .square()
                    .sum(dim=1)
                    .cpu()
                    .numpy()
                )
                del source, truth, flat, truth_flat

        split_data[split] = {
            "distance": distance,
            "radial_delta": radial_delta,
            "tangent_delta": tangent_delta,
            "cross": cross,
            "gram": gram,
            "truth_energy": truth_energy,
        }
        print(
            f"[complex-neighbor] split={split} "
            f"nearest_mean={distance[:, 0].mean():.4f}",
            flush=True,
        )
        torch.cuda.empty_cache()

    sufficient: dict[str, dict[int | str, dict]] = {}
    for split, data in split_data.items():
        bases = geometry_weights(
            data["distance"],
            data["radial_delta"],
            data["tangent_delta"],
        )
        for base_name, base in bases.items():
            corrections = (
                {
                    "none": np.ones(
                        base.shape, dtype=np.complex128
                    )
                }
                if "_moment_" in base_name
                else phase_corrections(data["gram"], base)
            )
            for phase_name, correction in corrections.items():
                coefficient = base.astype(np.complex128) * correction
                for amplitude_name in ("raw", "equalized"):
                    current = coefficient
                    if amplitude_name == "equalized":
                        current = equalize_amplitude(
                            current, data["gram"]
                        )
                    name = (
                        f"{base_name}_{phase_name}_{amplitude_name}"
                    )
                    cross_per_point = np.einsum(
                        "bk,bk->b",
                        current.conj(),
                        data["cross"],
                    )
                    energy_per_point = np.real(
                        np.einsum(
                            "bk,bkj,bj->b",
                            current.conj(),
                            data["gram"],
                            current,
                        )
                    )
                    sufficient.setdefault(name, {})[split] = {
                        "cross": complex(cross_per_point.sum()),
                        "prediction_energy": float(
                            energy_per_point.sum()
                        ),
                        "truth_energy": float(
                            data["truth_energy"].sum()
                        ),
                    }

    ranked = []
    for name, rows in sufficient.items():
        if any(split not in rows for split in split_keys):
            continue
        tune_cross = sum(rows[split]["cross"] for split in tune_seeds)
        tune_prediction_energy = sum(
            rows[split]["prediction_energy"] for split in tune_seeds
        )
        global_phase = float(np.angle(tune_cross))
        cross_rotation = np.exp(-1j * global_phase)
        scale = max(
            float(np.real(tune_cross * cross_rotation))
            / max(tune_prediction_energy, 1e-30),
            0.0,
        )
        scores = {}
        for split in split_keys:
            row = rows[split]
            cross_real = float(
                np.real(row["cross"] * cross_rotation)
            )
            nmse = (
                row["truth_energy"]
                + scale**2 * row["prediction_energy"]
                - 2.0 * scale * cross_real
            ) / max(row["truth_energy"], 1e-30)
            coherence = np.abs(row["cross"]) / np.sqrt(
                max(
                    row["prediction_energy"]
                    * row["truth_energy"],
                    1e-30,
                )
            )
            scores[str(split)] = {
                "NMSE": float(nmse),
                "coherence": float(coherence),
            }
        tune_nmse = [
            scores[str(split)]["NMSE"] for split in tune_seeds
        ]
        ranked.append(
            {
                "name": name,
                "global_phase": global_phase,
                "scale": scale,
                "tune_median_nmse": float(np.median(tune_nmse)),
                "tune_mean_nmse": float(np.mean(tune_nmse)),
                "tune_worst_nmse": float(np.max(tune_nmse)),
                "audit_nmse": scores[str(args.audit_seed)]["NMSE"],
                "external_nmse": (
                    scores[args.external_name]["NMSE"]
                    if args.external_indices
                    else None
                ),
                "scores": scores,
            }
        )
    ranked.sort(
        key=lambda row: (
            row["tune_median_nmse"],
            row["tune_mean_nmse"],
            row["tune_worst_nmse"],
        )
    )
    payload = {
        "selection_policy": (
            "all interpolation coefficients use positions and source "
            "channels only; phase/scale selected on four tune splits"
        ),
        "tune_seeds": tune_seeds,
        "audit_seed": args.audit_seed,
        "external_name": (
            args.external_name if args.external_indices else None
        ),
        "k_max": args.k_max,
        "config_count": len(ranked),
        "ranked": ranked,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        "[TOP]",
        json.dumps(ranked[:20], ensure_ascii=False),
        flush=True,
    )
    print(f"COMPLEX_NEIGHBOR_DONE out={out}", flush=True)


if __name__ == "__main__":
    main()
