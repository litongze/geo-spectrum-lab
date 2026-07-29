#!/usr/bin/env python3
"""Probe source-only phase synchronization in angle-delay blocks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

from _bootstrap import ROOT  # noqa: F401
from complex_neighbor_gate import (
    FEATURE_NAMES,
    ComplexNeighborGate,
    baseline_coefficients,
    build_complex_features,
)
from neighbor_source_weights import anisotropic_weights
from probe_full_array_steering_phase import array_steering_phase
from score_holdout import reproduce_val_indices
from wireless_twin.data.setup_config import load_setup


def parse_grid(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item]


def parse_kernels(value: str) -> list[tuple[int, int, int]]:
    return [
        tuple(int(part) for part in item.split("x"))
        for item in value.split(",")
        if item
    ]


def transform_channel(
    value: torch.Tensor, spec
) -> torch.Tensor:
    leading = value.shape[:-3]
    shaped = value.reshape(
        *leading,
        spec.mp,
        spec.mh,
        spec.mv,
        spec.n,
        spec.s,
    )
    angular = torch.fft.fft2(
        shaped, dim=(-4, -3), norm="ortho"
    )
    return torch.fft.ifft(angular, dim=-1, norm="ortho")


def smooth_block(
    value: torch.Tensor, kernel: tuple[int, int, int]
) -> torch.Tensor:
    if kernel == (1, 1, 1):
        return value
    batch, neighbors, mh, mv, delay = value.shape
    flat = value.reshape(batch * neighbors, 1, mh, mv, delay)
    padding = tuple(size // 2 for size in kernel)
    if value.is_complex():
        real = F.avg_pool3d(
            flat.real,
            kernel_size=kernel,
            stride=1,
            padding=padding,
            count_include_pad=False,
        )
        imag = F.avg_pool3d(
            flat.imag,
            kernel_size=kernel,
            stride=1,
            padding=padding,
            count_include_pad=False,
        )
        result = torch.complex(real, imag)
    else:
        result = F.avg_pool3d(
            flat,
            kernel_size=kernel,
            stride=1,
            padding=padding,
            count_include_pad=False,
        )
    return result.reshape(batch, neighbors, mh, mv, delay)


def load_component(
    checkpoint: Path,
    role: str,
    alpha: float,
    device: torch.device,
) -> dict:
    payload = torch.load(
        checkpoint,
        map_location=device,
        weights_only=False,
    )
    names = tuple(payload["feature_names"])
    model = ComplexNeighborGate(
        len(names), payload.get("architecture", "mlp16")
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return {
        "role": role,
        "alpha": alpha,
        "model": model,
        "indices": [FEATURE_NAMES.index(name) for name in names],
        "mean": torch.as_tensor(
            payload["feature_mean"],
            dtype=torch.float32,
            device=device,
        ),
        "std": torch.as_tensor(
            payload["feature_std"],
            dtype=torch.float32,
            device=device,
        ),
    }


def add_sufficient(
    storage: dict,
    key: str,
    split: str,
    prediction: torch.Tensor,
    truth: torch.Tensor,
    keep: torch.Tensor | None = None,
) -> None:
    cross = (
        prediction.conj() * truth
    ).reshape(prediction.shape[0], -1).sum(dim=1)
    prediction_energy = (
        prediction.abs()
        .square()
        .reshape(prediction.shape[0], -1)
        .sum(dim=1)
    )
    truth_energy = (
        truth.abs()
        .square()
        .reshape(truth.shape[0], -1)
        .sum(dim=1)
    )
    if keep is not None:
        cross = cross[keep]
        prediction_energy = prediction_energy[keep]
        truth_energy = truth_energy[keep]
    row = storage.setdefault(key, {}).setdefault(
        split,
        {
            "cross": 0.0j,
            "prediction_energy": 0.0,
            "truth_energy": 0.0,
        },
    )
    row["cross"] += complex(cross.sum().cpu())
    row["prediction_energy"] += float(
        prediction_energy.sum().cpu()
    )
    row["truth_energy"] += float(truth_energy.sum().cpu())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", default="Round1_Map(2)")
    parser.add_argument(
        "--config", default="configs/gs46_dual_gate50_10.json"
    )
    parser.add_argument(
        "--tune-seeds", default="1890,3716,962,1022"
    )
    parser.add_argument("--audit-seed", default="2262")
    parser.add_argument(
        "--kernels", default="1x1x1,1x1x3,1x1x7,3x3x3"
    )
    parser.add_argument("--confidence-powers", default="0.5,1,2")
    parser.add_argument("--strength-grid", default="0.1,0.25,0.5,0.75,1")
    parser.add_argument("--blend-grid", default="0.25,0.5,0.75,1")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out",
        default="docs/angle_delay_phase_sync/result.json",
    )
    args = parser.parse_args()

    tune = [item for item in args.tune_seeds.split(",") if item]
    names = [*tune, args.audit_seed]
    kernels = parse_kernels(args.kernels)
    confidence_powers = parse_grid(args.confidence_powers)
    strengths = parse_grid(args.strength_grid)
    blends = parse_grid(args.blend_grid)
    configs = [
        (kernel, confidence_power, strength, blend)
        for kernel in kernels
        for confidence_power in confidence_powers
        for strength in strengths
        for blend in blends
    ]
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    source_config = config["source"]
    k = int(source_config["k"])
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
    tune_union = {
        int(index)
        for name in tune
        for index in reproduce_val_indices(
            len(positions), 0.1, int(name)
        )
    }
    device = torch.device(args.device)
    sufficient = {}

    with torch.inference_mode():
        for name in names:
            val_idx = np.asarray(
                sorted(
                    reproduce_val_indices(
                        len(positions), 0.1, int(name)
                    )
                ),
                dtype=np.int64,
            )
            pool_idx = np.setdiff1d(all_idx, val_idx)
            distance, local = cKDTree(
                positions[pool_idx, :2]
            ).query(positions[val_idx, :2], k=k)
            distance = np.asarray(distance, dtype=np.float64)
            neighbor_idx = pool_idx[np.asarray(local)]
            delta_xy = (
                positions[neighbor_idx, :2]
                - positions[val_idx, None, :2]
            )
            radial = positions[val_idx, :2] - bs[None, :2]
            radial /= np.maximum(
                np.linalg.norm(
                    radial, axis=1, keepdims=True
                ),
                1e-12,
            )
            tangent = np.column_stack(
                [-radial[:, 1], radial[:, 0]]
            )
            radial_delta = np.einsum(
                "bkd,bd->bk", delta_xy, radial
            )
            tangent_delta = np.einsum(
                "bkd,bd->bk", delta_xy, tangent
            )
            base_weight, effective = anisotropic_weights(
                radial_delta,
                tangent_delta,
                float(source_config["radial_ratio"]),
                float(source_config["distance_power"]),
            )
            delta_radius = (
                radius[val_idx, None] - radius[neighbor_idx]
            )
            delta_unit = (
                unit[val_idx, None] - unit[neighbor_idx]
            )
            gate_config = config["complex_gate"]
            components = [
                load_component(
                    Path(gate_config["checkpoint"]).parent
                    / f"s{name}.pt",
                    "primary",
                    float(gate_config["alpha"]),
                    device,
                )
            ]
            if gate_config.get("secondary_checkpoint"):
                components.append(
                    load_component(
                        Path(
                            gate_config["secondary_checkpoint"]
                        ).parent
                        / f"s{name}.pt",
                        "secondary",
                        float(gate_config["secondary_alpha"]),
                        device,
                    )
                )
            strict_keep = torch.as_tensor(
                [
                    int(index) not in tune_union
                    for index in val_idx
                ],
                dtype=torch.bool,
                device=device,
            )

            for start in range(0, len(val_idx), args.batch_size):
                stop = min(start + args.batch_size, len(val_idx))
                batch = stop - start
                values = torch.as_tensor(
                    np.array(
                        channels[
                            neighbor_idx[start:stop].reshape(-1)
                        ],
                        copy=True,
                    ).reshape(
                        batch,
                        k,
                        spec.m,
                        spec.n,
                        spec.s,
                    ),
                    dtype=torch.complex64,
                    device=device,
                )
                truth = torch.as_tensor(
                    np.array(
                        channels[val_idx[start:stop]], copy=True
                    ),
                    dtype=torch.complex64,
                    device=device,
                )
                phase = (
                    float(config["phase"]["k0_rad_per_meter"])
                    + float(
                        config["phase"][
                            "k1_rad_per_meter_per_subcarrier"
                        ]
                    )
                    * subcarrier[None, None]
                ) * delta_radius[start:stop, :, None]
                values *= torch.as_tensor(
                    np.exp(1j * phase),
                    dtype=torch.complex64,
                    device=device,
                )[:, :, None, None, :]
                steering_phase = array_steering_phase(
                    delta_unit[start:stop],
                    spec,
                    config["steering"],
                )
                values *= torch.as_tensor(
                    np.exp(1j * steering_phase),
                    dtype=torch.complex64,
                    device=device,
                )[..., None]

                flat = values.reshape(batch, k, -1)
                gram = torch.einsum(
                    "bil,bjl->bij", flat.conj(), flat
                )
                base_weight_t = torch.as_tensor(
                    base_weight[start:stop],
                    dtype=torch.float32,
                    device=device,
                )
                features, amplitude = build_complex_features(
                    gram,
                    base_weight_t,
                    torch.as_tensor(
                        distance[start:stop],
                        dtype=torch.float32,
                        device=device,
                    ),
                    torch.as_tensor(
                        effective[start:stop],
                        dtype=torch.float32,
                        device=device,
                    ),
                    torch.as_tensor(
                        radial_delta[start:stop],
                        dtype=torch.float32,
                        device=device,
                    ),
                    torch.as_tensor(
                        tangent_delta[start:stop],
                        dtype=torch.float32,
                        device=device,
                    ),
                    torch.as_tensor(
                        positions[val_idx[start:stop]],
                        dtype=torch.float32,
                        device=device,
                    ),
                    torch.as_tensor(
                        bs, dtype=torch.float32, device=device
                    ),
                )
                baseline_coefficient = baseline_coefficients(
                    base_weight_t, amplitude
                )
                coefficient = baseline_coefficient
                for component in components:
                    normalized = (
                        (
                            features[..., component["indices"]]
                            - component["mean"]
                        )
                        / component["std"]
                    ).clamp(-8.0, 8.0)
                    learned, _ = component[
                        "model"
                    ].coefficients(
                        normalized, base_weight_t, amplitude
                    )
                    if component["role"] == "primary":
                        coefficient = (
                            (1.0 - component["alpha"])
                            * baseline_coefficient
                            + component["alpha"] * learned
                        )
                    else:
                        coefficient = coefficient + component[
                            "alpha"
                        ] * (learned - baseline_coefficient)

                transformed = transform_channel(values, spec)
                transformed_truth = transform_channel(truth, spec)
                baseline = (
                    coefficient[
                        :, :, None, None, None, None, None
                    ]
                    * transformed
                ).sum(dim=1)
                keep = strict_keep[start:stop]
                add_sufficient(
                    sufficient,
                    "baseline",
                    name,
                    baseline,
                    transformed_truth,
                )
                if name == args.audit_seed:
                    add_sufficient(
                        sufficient,
                        "baseline",
                        "audit_strict",
                        baseline,
                        transformed_truth,
                        keep,
                    )

                nearest = transformed[:, :1]
                block_cross = (
                    nearest.conj() * transformed
                ).sum(dim=(2, 5))
                nearest_energy = (
                    nearest.abs().square().sum(dim=(2, 5))
                ).expand(-1, k, -1, -1, -1)
                block_energy = transformed.abs().square().sum(
                    dim=(2, 5)
                )
                block_banks = {}
                for kernel in kernels:
                    cross = smooth_block(block_cross, kernel)
                    left_energy = smooth_block(
                        nearest_energy, kernel
                    )
                    right_energy = smooth_block(
                        block_energy, kernel
                    )
                    normalized_cross = cross / (
                        left_energy * right_energy
                    ).clamp_min(1e-30).sqrt()
                    block_banks[kernel] = (
                        -torch.angle(normalized_cross),
                        normalized_cross.abs().clamp(0.0, 1.0),
                    )
                for config_key in configs:
                    (
                        kernel,
                        confidence_power,
                        strength,
                        blend,
                    ) = config_key
                    angle, confidence = block_banks[kernel]
                    correction = torch.exp(
                        1j
                        * strength
                        * confidence.pow(confidence_power)
                        * angle
                    )
                    local_coefficient = (
                        coefficient[:, :, None, None, None]
                        * correction
                    )
                    synchronized = (
                        local_coefficient[
                            :, :, None, :, :, None, :
                        ]
                        * transformed
                    ).sum(dim=1)
                    prediction = (
                        (1.0 - blend) * baseline
                        + blend * synchronized
                    )
                    key = (
                        f"k{kernel[0]}x{kernel[1]}x{kernel[2]}"
                        f"_cp{confidence_power:g}"
                        f"_s{strength:g}_b{blend:g}"
                    )
                    add_sufficient(
                        sufficient,
                        key,
                        name,
                        prediction,
                        transformed_truth,
                    )
                    if name == args.audit_seed:
                        add_sufficient(
                            sufficient,
                            key,
                            "audit_strict",
                            prediction,
                            transformed_truth,
                            keep,
                        )
                del values, truth, transformed, transformed_truth
            print(
                f"[angle-delay-phase] split={name} done",
                flush=True,
            )
            del components
            torch.cuda.empty_cache()

    rows = []
    for key, split_rows in sufficient.items():
        tune_cross = sum(
            split_rows[name]["cross"] for name in tune
        )
        tune_energy = sum(
            split_rows[name]["prediction_energy"] for name in tune
        )
        phase = float(np.angle(tune_cross))
        rotation = np.exp(-1j * phase)
        scale = max(
            float(np.real(tune_cross * rotation))
            / max(tune_energy, 1e-30),
            0.0,
        )
        scores = {}
        for name, row in split_rows.items():
            scores[name] = float(
                (
                    row["truth_energy"]
                    + scale**2 * row["prediction_energy"]
                    - 2.0
                    * scale
                    * float(
                        np.real(row["cross"] * rotation)
                    )
                )
                / max(row["truth_energy"], 1e-30)
            )
        rows.append(
            {
                "name": key,
                "phase": phase,
                "scale": scale,
                "scores": scores,
            }
        )
    baseline = next(row for row in rows if row["name"] == "baseline")
    for row in rows:
        tune_delta = {
            name: row["scores"][name] - baseline["scores"][name]
            for name in tune
        }
        row["tune_delta_nmse"] = tune_delta
        row["tune_delta_nmse_mean"] = float(
            np.mean(list(tune_delta.values()))
        )
        row["tune_delta_nmse_worst"] = float(
            np.max(list(tune_delta.values()))
        )
        row["audit_delta_nmse"] = float(
            row["scores"][args.audit_seed]
            - baseline["scores"][args.audit_seed]
        )
        row["strict_audit_delta_nmse"] = float(
            row["scores"]["audit_strict"]
            - baseline["scores"]["audit_strict"]
        )
    rows.sort(
        key=lambda row: (
            row["tune_delta_nmse_mean"],
            row["tune_delta_nmse_worst"],
        )
    )
    robust = [
        row
        for row in rows
        if row["tune_delta_nmse_worst"] <= 0.0
        and row["audit_delta_nmse"] <= 0.0
        and row["strict_audit_delta_nmse"] <= 0.0
    ]
    payload = {
        "selection_policy": (
            "angle-delay phase synchronization uses source channels "
            "only; parameters selected on four tune folds"
        ),
        "config": args.config,
        "baseline": baseline,
        "best": rows[0],
        "best_robust": robust[0] if robust else None,
        "ranked": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    selected = robust[0] if robust else rows[0]
    print(
        "ANGLE_DELAY_PHASE_SYNC_DONE "
        f"name={selected['name']} "
        f"tune={selected['tune_delta_nmse_mean']:+.6f} "
        f"worst={selected['tune_delta_nmse_worst']:+.6f} "
        f"audit={selected['audit_delta_nmse']:+.6f} "
        f"strict={selected['strict_audit_delta_nmse']:+.6f} "
        f"out={out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
