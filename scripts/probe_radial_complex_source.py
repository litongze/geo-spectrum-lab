#!/usr/bin/env python3
"""Probe radial-map neighbor selection for the gated complex source."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

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
from sweep_rayprofile_spectrum_knn import local_radial_profiles
from wireless_twin.data.map_loader import load_point_cloud
from wireless_twin.data.setup_config import load_setup
from wireless_twin.models.raytrace2 import build_heightmap


def parse_grid(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item]


def select_neighbors(
    query_idx: np.ndarray,
    pool_idx: np.ndarray,
    positions: np.ndarray,
    profiles: np.ndarray,
    profile_lambda: float,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    query = positions[query_idx]
    pool = positions[pool_idx]
    xy_delta = query[:, None, :2] - pool[None, :, :2]
    spatial_distance2 = np.square(xy_delta).sum(axis=-1)
    if profile_lambda > 0.0:
        scale = profiles[pool_idx].std(axis=0).clip(0.05)
        profile_delta = (
            profiles[query_idx, None] - profiles[pool_idx][None]
        ) / scale
        effective_distance2 = (
            spatial_distance2
            + profile_lambda**2
            * np.square(profile_delta).mean(axis=-1)
        )
    else:
        effective_distance2 = spatial_distance2
    local = np.argpartition(
        effective_distance2, kth=k - 1, axis=1
    )[:, :k]
    selected_effective = np.take_along_axis(
        effective_distance2, local, axis=1
    )
    order = np.argsort(selected_effective, axis=1)
    local = np.take_along_axis(local, order, axis=1)
    distance = np.sqrt(
        np.maximum(
            np.take_along_axis(spatial_distance2, local, axis=1),
            1e-6,
        )
    )
    return pool_idx[local], distance


def load_gate_component(
    checkpoint: Path,
    alpha: float,
    role: str,
    device: torch.device,
) -> dict:
    payload = torch.load(
        checkpoint,
        map_location=device,
        weights_only=False,
    )
    names = tuple(payload["feature_names"])
    unknown = set(names).difference(FEATURE_NAMES)
    if unknown:
        raise ValueError(
            f"unknown features in {checkpoint}: {sorted(unknown)}"
        )
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


def gated_source(
    channels: torch.Tensor,
    query_position: np.ndarray,
    neighbor_idx: np.ndarray,
    distance: np.ndarray,
    positions: np.ndarray,
    radius: np.ndarray,
    unit: np.ndarray,
    bs: np.ndarray,
    subcarrier: np.ndarray,
    spec,
    phase_config: dict,
    steering_config: dict,
    source_config: dict,
    components: list[dict],
    device: torch.device,
) -> torch.Tensor:
    batch, k = neighbor_idx.shape
    neighbor_t = torch.as_tensor(
        neighbor_idx, dtype=torch.long, device=device
    )
    values = channels[neighbor_t].clone()
    query_radius = np.linalg.norm(
        query_position - bs[None], axis=1
    )
    query_unit = (
        query_position - bs[None]
    ) / np.maximum(query_radius[:, None], 1e-12)
    delta_radius = query_radius[:, None] - radius[neighbor_idx]
    phase = (
        float(phase_config["k0_rad_per_meter"])
        + float(
            phase_config[
                "k1_rad_per_meter_per_subcarrier"
            ]
        )
        * subcarrier[None, None]
    ) * delta_radius[:, :, None]
    values *= torch.as_tensor(
        np.exp(1j * phase),
        dtype=torch.complex64,
        device=device,
    )[:, :, None, None, :]
    delta_unit = query_unit[:, None] - unit[neighbor_idx]
    steering_phase = array_steering_phase(
        delta_unit, spec, steering_config
    )
    values *= torch.as_tensor(
        np.exp(1j * steering_phase),
        dtype=torch.complex64,
        device=device,
    )[..., None]

    delta_xy = (
        positions[neighbor_idx, :2]
        - query_position[:, None, :2]
    )
    radial = query_position[:, :2] - bs[None, :2]
    radial /= np.maximum(
        np.linalg.norm(radial, axis=1, keepdims=True), 1e-12
    )
    tangent = np.column_stack([-radial[:, 1], radial[:, 0]])
    radial_delta = np.einsum("bkd,bd->bk", delta_xy, radial)
    tangent_delta = np.einsum(
        "bkd,bd->bk", delta_xy, tangent
    )
    weight, effective = anisotropic_weights(
        radial_delta,
        tangent_delta,
        float(source_config["radial_ratio"]),
        float(source_config["distance_power"]),
    )
    flat = values.reshape(batch, k, -1)
    gram = torch.einsum("bil,bjl->bij", flat.conj(), flat)
    base_weight = torch.as_tensor(
        weight, dtype=torch.float32, device=device
    )
    features, amplitude = build_complex_features(
        gram,
        base_weight,
        torch.as_tensor(
            distance, dtype=torch.float32, device=device
        ),
        torch.as_tensor(
            effective, dtype=torch.float32, device=device
        ),
        torch.as_tensor(
            radial_delta, dtype=torch.float32, device=device
        ),
        torch.as_tensor(
            tangent_delta, dtype=torch.float32, device=device
        ),
        torch.as_tensor(
            query_position, dtype=torch.float32, device=device
        ),
        torch.as_tensor(bs, dtype=torch.float32, device=device),
    )
    baseline = baseline_coefficients(base_weight, amplitude)
    coefficient = baseline
    for component in components:
        normalized = (
            (
                features[..., component["indices"]]
                - component["mean"]
            )
            / component["std"]
        ).clamp(-8.0, 8.0)
        learned, _ = component["model"].coefficients(
            normalized, base_weight, amplitude
        )
        if component["role"] == "primary":
            coefficient = (
                (1.0 - component["alpha"]) * baseline
                + component["alpha"] * learned
            )
        else:
            coefficient = coefficient + component["alpha"] * (
                learned - baseline
            )
    output = torch.einsum(
        "bk,bkmns->bmns", coefficient, values
    )
    return output * torch.exp(
        torch.tensor(
            1j * float(phase_config["global_phase"]),
            dtype=torch.complex64,
            device=device,
        )
    )


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
    parser.add_argument("--profile", default="radial_height")
    parser.add_argument(
        "--lambda-grid", default="0,4,8,12,16,24,32"
    )
    parser.add_argument("--mix-grid", default="0.25,0.5,0.75,1")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out",
        default="docs/radial_complex_source/result.json",
    )
    args = parser.parse_args()

    tune = [item for item in args.tune_seeds.split(",") if item]
    names = [*tune, args.audit_seed]
    lambdas = parse_grid(args.lambda_grid)
    mixes = parse_grid(args.mix_grid)
    if 0.0 not in lambdas:
        raise ValueError("lambda grid must include 0")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    source_config = config["source"]
    k = int(source_config["k"])

    datadir = Path(args.datadir)
    spec = load_setup(datadir / "Round1_Setup.json")
    positions = np.load(
        datadir / "Round1_Train_Pos.npy"
    ).astype(np.float64)
    points = load_point_cloud(datadir / "Round1_Map.ply")
    heightmap, x0, y0, resolution = build_heightmap(points)
    profiles = local_radial_profiles(
        positions,
        np.asarray(spec.bs_position, dtype=np.float32),
        heightmap,
        x0,
        y0,
        resolution,
    )[args.profile]
    del points, heightmap
    device = torch.device(args.device)
    channels_np = np.load(
        datadir / "Round1_Train_Channel.npy", mmap_mode="r"
    )
    channels = torch.as_tensor(
        np.array(channels_np, copy=True),
        dtype=torch.complex64,
        device=device,
    )
    del channels_np

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
    sufficient: dict = {}

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
            neighbor_bank = {}
            distance_bank = {}
            for profile_lambda in lambdas:
                neighbor_bank[profile_lambda], distance_bank[
                    profile_lambda
                ] = select_neighbors(
                    val_idx,
                    pool_idx,
                    positions,
                    profiles,
                    profile_lambda,
                    k,
                )
            gate_config = config["complex_gate"]
            components = [
                load_gate_component(
                    Path(gate_config["checkpoint"]).parent
                    / f"s{name}.pt",
                    float(gate_config["alpha"]),
                    "primary",
                    device,
                )
            ]
            if gate_config.get("secondary_checkpoint"):
                components.append(
                    load_gate_component(
                        Path(
                            gate_config["secondary_checkpoint"]
                        ).parent
                        / f"s{name}.pt",
                        float(gate_config["secondary_alpha"]),
                        "secondary",
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
                truth = channels[
                    torch.as_tensor(
                        val_idx[start:stop],
                        dtype=torch.long,
                        device=device,
                    )
                ]
                predictions = {}
                for profile_lambda in lambdas:
                    predictions[profile_lambda] = gated_source(
                        channels,
                        positions[val_idx[start:stop]],
                        neighbor_bank[profile_lambda][start:stop],
                        distance_bank[profile_lambda][start:stop],
                        positions,
                        radius,
                        unit,
                        bs,
                        subcarrier,
                        spec,
                        config["phase"],
                        config["steering"],
                        source_config,
                        components,
                        device,
                    )
                baseline = predictions[0.0]
                add_sufficient(
                    sufficient,
                    "lambda0_mix0",
                    name,
                    baseline,
                    truth,
                )
                if name == args.audit_seed:
                    add_sufficient(
                        sufficient,
                        "lambda0_mix0",
                        "audit_strict",
                        baseline,
                        truth,
                        strict_keep[start:stop],
                    )
                for profile_lambda in lambdas:
                    if profile_lambda == 0.0:
                        continue
                    for mix in mixes:
                        output = (
                            (1.0 - mix) * baseline
                            + mix * predictions[profile_lambda]
                        )
                        key = (
                            f"lambda{profile_lambda:g}_mix{mix:g}"
                        )
                        add_sufficient(
                            sufficient, key, name, output, truth
                        )
                        if name == args.audit_seed:
                            add_sufficient(
                                sufficient,
                                key,
                                "audit_strict",
                                output,
                                truth,
                                strict_keep[start:stop],
                            )
                del truth, predictions
            print(
                f"[radial-complex] split={name} done",
                flush=True,
            )
            del components
            torch.cuda.empty_cache()

    baseline_key = "lambda0_mix0"
    rows = []
    for key, split_rows in sufficient.items():
        tune_cross = sum(
            split_rows[name]["cross"] for name in tune
        )
        tune_prediction_energy = sum(
            split_rows[name]["prediction_energy"] for name in tune
        )
        phase = float(np.angle(tune_cross))
        rotation = np.exp(-1j * phase)
        scale = max(
            float(np.real(tune_cross * rotation))
            / max(tune_prediction_energy, 1e-30),
            0.0,
        )
        scores = {}
        for name, row in split_rows.items():
            nmse = (
                row["truth_energy"]
                + scale**2 * row["prediction_energy"]
                - 2.0
                * scale
                * float(np.real(row["cross"] * rotation))
            ) / max(row["truth_energy"], 1e-30)
            scores[name] = float(nmse)
        if key == baseline_key:
            profile_lambda, mix = 0.0, 0.0
        else:
            prefix, suffix = key.split("_")
            profile_lambda = float(prefix.removeprefix("lambda"))
            mix = float(suffix.removeprefix("mix"))
        rows.append(
            {
                "profile_lambda": profile_lambda,
                "mix": mix,
                "phase": phase,
                "scale": scale,
                "scores": scores,
            }
        )

    baseline = next(
        row for row in rows if row["profile_lambda"] == 0.0
    )
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
            "radial-height neighbor parameters selected on four tune "
            "folds; audit and strict audit are diagnostics"
        ),
        "config": args.config,
        "profile": args.profile,
        "k": k,
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
        "RADIAL_COMPLEX_SOURCE_DONE "
        f"lambda={selected['profile_lambda']:g} "
        f"mix={selected['mix']:g} "
        f"tune={selected['tune_delta_nmse_mean']:+.6f} "
        f"worst={selected['tune_delta_nmse_worst']:+.6f} "
        f"audit={selected['audit_delta_nmse']:+.6f} "
        f"strict={selected['strict_audit_delta_nmse']:+.6f} "
        f"out={out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
