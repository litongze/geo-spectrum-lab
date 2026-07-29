#!/usr/bin/env python3
"""Probe grouped amplitude consensus in a unitary angle-delay domain."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from _bootstrap import ROOT  # noqa: F401
from complex_neighbor_gate import (
    FEATURE_NAMES,
    ComplexNeighborGate,
    baseline_coefficients,
    build_complex_features,
)
from probe_full_array_steering_phase import array_steering_phase
from score_holdout import reproduce_val_indices
from wireless_twin.data.setup_config import load_setup


def transform_channel(
    value: torch.Tensor,
    mh: int,
    mv: int,
    mp: int,
    layout: str,
) -> torch.Tensor:
    leading = value.shape[:-3]
    if layout == "phv":
        shaped = value.reshape(
            *leading,
            mp,
            mh,
            mv,
            value.shape[-2],
            value.shape[-1],
        )
        fft_dims = (-4, -3)
    elif layout == "pvh":
        shaped = value.reshape(
            *leading,
            mp,
            mv,
            mh,
            value.shape[-2],
            value.shape[-1],
        )
        fft_dims = (-4, -3)
    elif layout == "hvp":
        shaped = value.reshape(
            *leading,
            mh,
            mv,
            mp,
            value.shape[-2],
            value.shape[-1],
        )
        fft_dims = (-5, -4)
    else:
        raise ValueError(f"unsupported layout {layout}")
    angular = torch.fft.fft2(
        shaped, dim=fft_dims, norm="ortho"
    )
    return torch.fft.ifft(angular, dim=-1, norm="ortho")


def add_sufficient(
    storage: dict[str, dict[str, dict[str, complex | float]]],
    name: str,
    split: str,
    prediction: torch.Tensor,
    truth: torch.Tensor,
    keep: torch.Tensor | None = None,
) -> None:
    flat_prediction = prediction.reshape(prediction.shape[0], -1)
    flat_truth = truth.reshape(truth.shape[0], -1)
    cross = (flat_prediction.conj() * flat_truth).sum(dim=1)
    prediction_energy = flat_prediction.abs().square().sum(dim=1)
    truth_energy = flat_truth.abs().square().sum(dim=1)
    if keep is not None:
        cross = cross[keep]
        prediction_energy = prediction_energy[keep]
        truth_energy = truth_energy[keep]
    row = storage.setdefault(name, {}).setdefault(
        split,
        {
            "cross": 0j,
            "prediction_energy": 0.0,
            "truth_energy": 0.0,
        },
    )
    row["cross"] += complex(cross.sum().cpu())
    row["prediction_energy"] += float(prediction_energy.sum())
    row["truth_energy"] += float(truth_energy.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", default="Round1_Map(2)")
    parser.add_argument(
        "--config", default="configs/gs43_dual_mix10.json"
    )
    parser.add_argument(
        "--tune-seeds", default="1890,3716,962,1022"
    )
    parser.add_argument("--audit-seed", type=int, default=2262)
    parser.add_argument("--external-indices")
    parser.add_argument("--external-name", default="testmatched")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--gamma-grid", default="0.25,0.5,0.75,1")
    parser.add_argument("--blend-grid", default="0.25,0.5,0.75,1")
    parser.add_argument("--groups", default="pol,path")
    parser.add_argument(
        "--layout", choices=("phv", "pvh", "hvp"), default="phv"
    )
    parser.add_argument("--complex-gate-dir")
    parser.add_argument(
        "--complex-gate-alpha", type=float, default=0.6
    )
    parser.add_argument("--ratio-min", type=float, default=0.25)
    parser.add_argument("--ratio-max", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out",
        default="docs/angle_delay_neighbor_consensus/result.json",
    )
    args = parser.parse_args()

    tune_seeds = [
        int(value) for value in args.tune_seeds.split(",") if value
    ]
    split_names = [str(seed) for seed in tune_seeds]
    split_names.append(str(args.audit_seed))
    if args.external_indices:
        split_names.append(args.external_name)
    tune_union = set().union(
        *[
            set(reproduce_val_indices(2000, 0.1, seed))
            for seed in tune_seeds
        ]
    )
    gammas = [
        float(value)
        for value in args.gamma_grid.split(",")
        if value
    ]
    blends = [
        float(value)
        for value in args.blend_grid.split(",")
        if value
    ]
    groups = [value for value in args.groups.split(",") if value]
    if any(group not in {"pol", "path"} for group in groups):
        raise ValueError("groups must contain only pol or path")

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    source_config = config["source"]
    phase_config = config["phase"]
    steering = config["steering"]
    k0 = float(phase_config["k0_rad_per_meter"])
    k1 = float(
        phase_config["k1_rad_per_meter_per_subcarrier"]
    )
    radial_ratio = float(source_config["radial_ratio"])
    distance_power = float(source_config["distance_power"])

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
    sufficient: dict[
        str, dict[str, dict[str, complex | float]]
    ] = {}

    for split_name in split_names:
        if split_name == args.external_name:
            val_idx = np.load(args.external_indices).astype(np.int64)
        else:
            val_idx = np.asarray(
                sorted(
                    reproduce_val_indices(
                        len(positions), 0.1, int(split_name)
                    )
                ),
                dtype=np.int64,
            )
        pool_idx = np.setdiff1d(all_idx, val_idx)
        distance, local = cKDTree(positions[pool_idx, :2]).query(
            positions[val_idx, :2], k=args.k
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
        radial_delta = np.einsum(
            "bkd,bd->bk", delta_xy, radial
        )
        tangent_delta = np.einsum(
            "bkd,bd->bk", delta_xy, tangent
        )
        effective = np.sqrt(
            np.square(radial_ratio * radial_delta)
            + np.square(tangent_delta)
        )
        base_weight = (
            1.0
            / np.maximum(effective, 0.05) ** distance_power
        )
        base_weight /= np.maximum(
            base_weight.sum(axis=1, keepdims=True), 1e-30
        )
        delta_radius = (
            radius[val_idx, None] - radius[neighbor_idx]
        )
        delta_unit = unit[val_idx, None] - unit[neighbor_idx]
        strict_keep = torch.as_tensor(
            [
                int(index) not in tune_union
                for index in val_idx
            ],
            dtype=torch.bool,
            device=device,
        )
        gate_model = None
        gate_feature_indices = None
        gate_feature_mean = None
        gate_feature_std = None
        if args.complex_gate_dir:
            gate_checkpoint = torch.load(
                Path(args.complex_gate_dir)
                / f"s{split_name}.pt",
                map_location=device,
                weights_only=False,
            )
            checkpoint_names = tuple(
                gate_checkpoint["feature_names"]
            )
            gate_feature_indices = [
                FEATURE_NAMES.index(name)
                for name in checkpoint_names
            ]
            gate_model = ComplexNeighborGate(
                len(checkpoint_names),
                gate_checkpoint.get("architecture", "mlp16"),
            ).to(device)
            gate_model.load_state_dict(
                gate_checkpoint["model_state"]
            )
            gate_model.eval()
            gate_feature_mean = torch.as_tensor(
                gate_checkpoint["feature_mean"],
                dtype=torch.float32,
                device=device,
            )
            gate_feature_std = torch.as_tensor(
                gate_checkpoint["feature_std"],
                dtype=torch.float32,
                device=device,
            )

        with torch.inference_mode():
            for start in range(0, len(val_idx), args.batch_size):
                stop = min(start + args.batch_size, len(val_idx))
                batch = stop - start
                neighbors = torch.as_tensor(
                    np.array(
                        channels[
                            neighbor_idx[start:stop].reshape(-1)
                        ],
                        copy=True,
                    ).reshape(
                        batch,
                        args.k,
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
                neighbors *= torch.as_tensor(
                    np.exp(1j * phase),
                    dtype=torch.complex64,
                    device=device,
                )[:, :, None, None, :]
                steering_phase = array_steering_phase(
                    delta_unit[start:stop], spec, steering
                )
                neighbors *= torch.as_tensor(
                    np.exp(1j * steering_phase),
                    dtype=torch.complex64,
                    device=device,
                )[..., None]

                weight = torch.as_tensor(
                    base_weight[start:stop],
                    dtype=torch.float32,
                    device=device,
                )
                energy = neighbors.abs().square().sum(
                    dim=(2, 3, 4)
                )
                amplitude = energy.clamp_min(1e-30).sqrt()
                target_amplitude = (
                    weight * amplitude
                ).sum(dim=1, keepdim=True)
                baseline_coefficient = (
                    weight
                    * target_amplitude
                    / amplitude.clamp_min(1e-30)
                ).to(torch.complex64)
                source_coefficient = baseline_coefficient
                consensus_coefficient = weight.to(torch.complex64)
                group_weight = weight
                if gate_model is not None:
                    flat = neighbors.reshape(batch, args.k, -1)
                    gram = torch.einsum(
                        "bil,bjl->bij", flat.conj(), flat
                    )
                    gate_features, gate_amplitude = (
                        build_complex_features(
                            gram,
                            weight,
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
                                bs,
                                dtype=torch.float32,
                                device=device,
                            ),
                        )
                    )
                    normalized_features = (
                        (
                            gate_features[
                                ..., gate_feature_indices
                            ]
                            - gate_feature_mean
                        )
                        / gate_feature_std
                    ).clamp(-8.0, 8.0)
                    learned_coefficient, _ = (
                        gate_model.coefficients(
                            normalized_features,
                            weight,
                            gate_amplitude,
                        )
                    )
                    source_coefficient = (
                        (1.0 - args.complex_gate_alpha)
                        * baseline_coefficients(
                            weight, gate_amplitude
                        )
                        + args.complex_gate_alpha
                        * learned_coefficient
                    )
                transformed = transform_channel(
                    neighbors,
                    spec.mh,
                    spec.mv,
                    spec.mp,
                    args.layout,
                )
                transformed_truth = transform_channel(
                    truth,
                    spec.mh,
                    spec.mv,
                    spec.mp,
                    args.layout,
                )
                broadcast_coefficient = source_coefficient[
                    :, :, None, None, None, None, None
                ]
                baseline = (
                    broadcast_coefficient * transformed
                ).sum(dim=1)
                keep_batch = strict_keep[start:stop]
                add_sufficient(
                    sufficient,
                    "baseline",
                    split_name,
                    baseline,
                    transformed_truth,
                )
                if split_name == str(args.audit_seed):
                    add_sufficient(
                        sufficient,
                        "baseline",
                        "audit_strict",
                        baseline,
                        transformed_truth,
                        keep_batch,
                    )

                for group in groups:
                    polarization_dim = (
                        4 if args.layout == "hvp" else 2
                    )
                    reduce_dims = (
                        (polarization_dim,)
                        if group == "pol"
                        else (polarization_dim, 5)
                    )
                    group_amplitude = (
                        transformed.abs()
                        .square()
                        .sum(dim=reduce_dims, keepdim=True)
                        .clamp_min(1e-30)
                        .sqrt()
                    )
                    broadcast_weight = group_weight[
                        :, :, None, None, None, None, None
                    ]
                    target_group_amplitude = (
                        broadcast_weight * group_amplitude
                    ).sum(dim=1, keepdim=True)
                    ratio = (
                        target_group_amplitude
                        / group_amplitude.clamp_min(1e-30)
                    ).clamp(args.ratio_min, args.ratio_max)
                    for gamma in gammas:
                        adjusted = (
                            transformed * ratio.pow(gamma)
                        )
                        consensus = (
                            consensus_coefficient[
                                :,
                                :,
                                None,
                                None,
                                None,
                                None,
                                None,
                            ]
                            * adjusted
                        ).sum(dim=1)
                        for blend in blends:
                            prediction = (
                                (1.0 - blend) * baseline
                                + blend * consensus
                            )
                            name = (
                                f"{args.layout}_{group}_g{gamma:g}"
                                f"_b{blend:g}"
                            )
                            add_sufficient(
                                sufficient,
                                name,
                                split_name,
                                prediction,
                                transformed_truth,
                            )
                            if split_name == str(args.audit_seed):
                                add_sufficient(
                                    sufficient,
                                    name,
                                    "audit_strict",
                                    prediction,
                                    transformed_truth,
                                    keep_batch,
                                )
                del neighbors, truth, transformed, transformed_truth
        print(
            f"[angle-delay-consensus] split={split_name} done",
            flush=True,
        )
        torch.cuda.empty_cache()

    required = [*map(str, tune_seeds), str(args.audit_seed)]
    if args.external_indices:
        required.append(args.external_name)
    ranked = []
    baseline_rows = sufficient["baseline"]
    for name, rows in sufficient.items():
        if any(split not in rows for split in required):
            continue
        tune_cross = sum(
            complex(rows[str(seed)]["cross"])
            for seed in tune_seeds
        )
        tune_prediction_energy = sum(
            float(rows[str(seed)]["prediction_energy"])
            for seed in tune_seeds
        )
        phase = float(np.angle(tune_cross))
        rotation = np.exp(-1j * phase)
        scale = max(
            float(np.real(tune_cross * rotation))
            / max(tune_prediction_energy, 1e-30),
            0.0,
        )
        scores = {}
        for split in [*required, "audit_strict"]:
            row = rows[split]
            nmse = (
                float(row["truth_energy"])
                + scale**2 * float(row["prediction_energy"])
                - 2.0
                * scale
                * float(np.real(complex(row["cross"]) * rotation))
            ) / max(float(row["truth_energy"]), 1e-30)
            scores[split] = float(nmse)
        baseline_scores = {}
        baseline_cross = sum(
            complex(baseline_rows[str(seed)]["cross"])
            for seed in tune_seeds
        )
        baseline_phase = float(np.angle(baseline_cross))
        baseline_rotation = np.exp(-1j * baseline_phase)
        baseline_scale = max(
            float(np.real(baseline_cross * baseline_rotation))
            / max(
                sum(
                    float(
                        baseline_rows[str(seed)][
                            "prediction_energy"
                        ]
                    )
                    for seed in tune_seeds
                ),
                1e-30,
            ),
            0.0,
        )
        for split in [*required, "audit_strict"]:
            row = baseline_rows[split]
            baseline_scores[split] = float(
                (
                    float(row["truth_energy"])
                    + baseline_scale**2
                    * float(row["prediction_energy"])
                    - 2.0
                    * baseline_scale
                    * float(
                        np.real(
                            complex(row["cross"])
                            * baseline_rotation
                        )
                    )
                )
                / max(float(row["truth_energy"]), 1e-30)
            )
        tune_delta = [
            scores[str(seed)] - baseline_scores[str(seed)]
            for seed in tune_seeds
        ]
        ranked.append(
            {
                "name": name,
                "phase": phase,
                "scale": scale,
                "tune_delta_nmse_median": float(
                    np.median(tune_delta)
                ),
                "tune_delta_nmse_mean": float(np.mean(tune_delta)),
                "tune_delta_nmse_worst": float(np.max(tune_delta)),
                "audit_delta_nmse": (
                    scores[str(args.audit_seed)]
                    - baseline_scores[str(args.audit_seed)]
                ),
                "strict_audit_delta_nmse": (
                    scores["audit_strict"]
                    - baseline_scores["audit_strict"]
                ),
                "external_delta_nmse": (
                    scores[args.external_name]
                    - baseline_scores[args.external_name]
                    if args.external_indices
                    else None
                ),
                "scores": scores,
            }
        )
    ranked.sort(
        key=lambda row: (
            row["tune_delta_nmse_mean"],
            row["tune_delta_nmse_worst"],
        )
    )
    payload = {
        "selection_policy": (
            "ranked only by four tune split aggregate NMSE; audit, "
            "strict audit, and external split are diagnostics"
        ),
        "transform": f"{args.layout}_angle_delay_ortho",
        "groups": groups,
        "gamma_grid": gammas,
        "blend_grid": blends,
        "ratio_clip": [args.ratio_min, args.ratio_max],
        "complex_gate": {
            "directory": args.complex_gate_dir,
            "alpha": args.complex_gate_alpha,
            "composition": (
                "complex blend with an independently geometry-weighted "
                "angle-delay consensus"
                if args.complex_gate_dir
                else None
            ),
        },
        "tune_seeds": tune_seeds,
        "audit_seed": args.audit_seed,
        "ranked": ranked,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n"
    )
    print(
        "[TOP]",
        json.dumps(ranked[:12], ensure_ascii=True),
        flush=True,
    )
    print(f"ANGLE_DELAY_CONSENSUS_DONE out={out}", flush=True)


if __name__ == "__main__":
    main()
