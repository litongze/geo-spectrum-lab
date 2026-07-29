#!/usr/bin/env python3
"""Audit joint source-only radial-neighbor PAS and PDP corrections."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from _bootstrap import ROOT  # noqa: F401
from score_holdout import reproduce_val_indices
from sweep_rayprofile_spectrum_knn import local_radial_profiles
from validate_moment_projection import (
    enforce_pas,
    enforce_pas_layout,
    enforce_pdp,
    stable_unit,
)
from wireless_twin.data.map_loader import load_point_cloud
from wireless_twin.data.setup_config import load_setup
from wireless_twin.models.raytrace2 import build_heightmap
from wireless_twin.signal import (
    pas_spectrum,
    pas_spectrum_phv,
    pas_spectrum_pvh,
    pdp_spectrum,
)


def parse_grid(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item]


def radial_neighbors(
    query_idx: np.ndarray,
    pool_idx: np.ndarray,
    positions: np.ndarray,
    profiles: np.ndarray,
    profile_lambda: float,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    xy_delta = (
        positions[query_idx, None, :2]
        - positions[pool_idx][None, :, :2]
    )
    spatial_distance2 = np.square(xy_delta).sum(axis=-1)
    profile_scale = profiles[pool_idx].std(axis=0).clip(0.05)
    profile_delta = (
        profiles[query_idx, None] - profiles[pool_idx][None]
    ) / profile_scale
    effective_distance2 = (
        spatial_distance2
        + profile_lambda**2
        * np.square(profile_delta).mean(axis=-1)
    )
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
    ).astype(np.float32)
    return pool_idx[local], distance


def bilateral_target(
    spectra: torch.Tensor,
    neighbors: np.ndarray,
    distance: np.ndarray,
    distance_power: float,
    anchor_strength: float,
    device: torch.device,
) -> torch.Tensor:
    neighbor_t = torch.as_tensor(
        neighbors, dtype=torch.long, device=device
    )
    values = spectra[neighbor_t]
    agreement = (values * values[:, :1]).sum(dim=-1)
    while agreement.ndim > 2:
        agreement = agreement.mean(dim=-1)
    distance_t = torch.as_tensor(
        distance, dtype=torch.float32, device=device
    ).clamp_min(0.3)
    weight = torch.softmax(
        -distance_power * distance_t.log()
        + anchor_strength * agreement,
        dim=1,
    )
    return stable_unit(
        torch.einsum("bk,bk...s->b...s", weight, values)
    )


def candidate_configs(
    pas_betas: list[float],
    pdp_betas: list[float],
) -> list[tuple[str, float, float]]:
    configs: list[tuple[str, float, float]] = [("none", 0.0, 0.0)]
    configs.extend(
        ("pas", beta, 0.0)
        for beta in pas_betas
        if beta > 0.0
    )
    configs.extend(
        ("pdp", 0.0, beta)
        for beta in pdp_betas
        if beta > 0.0
    )
    for pas_beta in pas_betas:
        if pas_beta <= 0.0:
            continue
        for pdp_beta in pdp_betas:
            if pdp_beta <= 0.0:
                continue
            configs.append(("pas_pdp", pas_beta, pdp_beta))
            configs.append(("pdp_pas", pas_beta, pdp_beta))
    return configs


def config_key(config: tuple[str, float, float]) -> str:
    order, pas_beta, pdp_beta = config
    return f"{order}_{pas_beta:g}_{pdp_beta:g}"


def correct_channel(
    predicted: torch.Tensor,
    selected_pas: torch.Tensor,
    selected_pdp: torch.Tensor,
    config: tuple[str, float, float],
    spec,
    pas_channel_mix: float,
    pdp_channel_mix: float,
    pas_layout: str,
) -> torch.Tensor:
    order, pas_beta, pdp_beta = config
    output = predicted
    for step in order.split("_"):
        if step == "none":
            continue
        if step == "pas":
            pas_transform = {
                "phv": pas_spectrum_phv,
                "hvp": pas_spectrum,
                "pvh": pas_spectrum_pvh,
            }[pas_layout]
            current = stable_unit(pas_transform(output, spec))
            target = stable_unit(
                (1.0 - pas_beta) * current
                + pas_beta * selected_pas
            )
            corrected = (
                enforce_pas(output, target, spec)
                if pas_layout == "phv"
                else enforce_pas_layout(
                    output, target, spec, pas_layout
                )
            )
            output = (
                (1.0 - pas_channel_mix) * output
                + pas_channel_mix * corrected
            )
        elif step == "pdp":
            current = stable_unit(pdp_spectrum(output, spec))
            target = stable_unit(
                (1.0 - pdp_beta) * current
                + pdp_beta * selected_pdp
            )
            corrected = enforce_pdp(output, target)
            output = (
                (1.0 - pdp_channel_mix) * output
                + pdp_channel_mix * corrected
            )
        else:
            raise ValueError(f"unknown correction step: {step}")
    return output


def empty_accumulator() -> dict[str, float | complex | int]:
    return {
        "pas": 0.0,
        "pdp": 0.0,
        "error": 0.0,
        "cross": 0.0j,
        "prediction_energy": 0.0,
        "truth_energy": 0.0,
        "pas_count": 0,
        "pdp_count": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", default="Round1_Map(2)")
    parser.add_argument(
        "--cache-dir", default="cache/teammate_knn_hvp"
    )
    parser.add_argument(
        "--prediction-template",
        default=(
            "cache/dual_gate50_10_projection/"
            "split_s{seed}_prediction.npy"
        ),
    )
    parser.add_argument(
        "--tune-seeds", default="1890,3716,962,1022"
    )
    parser.add_argument("--audit-seed", default="2262")
    parser.add_argument("--strict-audit", action="store_true")
    parser.add_argument("--profile-lambda", type=float, default=24.0)
    parser.add_argument("--pas-k", type=int, default=8)
    parser.add_argument("--pas-distance-power", type=float, default=1.0)
    parser.add_argument("--pas-anchor-strength", type=float, default=4.0)
    parser.add_argument("--pas-beta-grid", default="0.1,0.15,0.2,0.25")
    parser.add_argument("--pas-channel-mix", type=float, default=1.0)
    parser.add_argument("--pdp-k", type=int, default=32)
    parser.add_argument("--pdp-distance-power", type=float, default=0.5)
    parser.add_argument("--pdp-anchor-strength", type=float, default=4.0)
    parser.add_argument("--pdp-beta-grid", default="0.25,0.3,0.35,0.4")
    parser.add_argument("--pdp-channel-mix", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--pas-layout",
        choices=("phv", "hvp", "pvh"),
        default="phv",
        help="PAS layout used for source spectra, correction, and scoring",
    )
    parser.add_argument(
        "--out",
        default="docs/postprojection_radial_joint/result.json",
    )
    args = parser.parse_args()

    tune = [item for item in args.tune_seeds.split(",") if item]
    names = [*tune, args.audit_seed]
    configs = candidate_configs(
        parse_grid(args.pas_beta_grid),
        parse_grid(args.pdp_beta_grid),
    )
    datadir = Path(args.datadir)
    spec = load_setup(datadir / "Round1_Setup.json")
    positions = np.load(
        datadir / "Round1_Train_Pos.npy"
    ).astype(np.float32)
    points = load_point_cloud(datadir / "Round1_Map.ply")
    heightmap, x0, y0, resolution = build_heightmap(points)
    profiles = local_radial_profiles(
        positions,
        np.asarray(spec.bs_position, dtype=np.float32),
        heightmap,
        x0,
        y0,
        resolution,
    )["radial_height"]
    del points, heightmap

    channels = np.load(
        datadir / "Round1_Train_Channel.npy", mmap_mode="r"
    )
    device = torch.device(args.device)
    pas_cache_name = {
        "phv": "train_pas_phv.npy",
        "hvp": "train_pas_hvp.npy",
        "pvh": "train_pas_pvh.npy",
    }[args.pas_layout]
    cached_pas = np.load(
        Path(args.cache_dir) / pas_cache_name,
        mmap_mode="r",
    )
    cached_pdp = np.load(
        Path(args.cache_dir) / "train_pdp.npy",
        mmap_mode="r",
    )
    source_pas = stable_unit(
        torch.as_tensor(
            np.array(cached_pas, copy=True),
            dtype=torch.float32,
            device=device,
        )
    )
    source_pdp = stable_unit(
        torch.as_tensor(
            np.array(cached_pdp, copy=True),
            dtype=torch.float32,
            device=device,
        )
    )
    del cached_pas, cached_pdp
    pas_transform = {
        "phv": pas_spectrum_phv,
        "hvp": pas_spectrum,
        "pvh": pas_spectrum_pvh,
    }[args.pas_layout]

    all_idx = np.arange(len(positions), dtype=np.int64)
    tune_union = {
        int(index)
        for name in tune
        for index in reproduce_val_indices(
            len(positions), 0.1, int(name)
        )
    }
    split_scores: dict[str, dict[str, dict[str, float]]] = {}
    split_sufficient: dict[
        str, dict[str, dict[str, float | complex]]
    ] = {}
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
            max_k = max(args.pas_k, args.pdp_k)
            neighbors, distance = radial_neighbors(
                val_idx,
                pool_idx,
                positions,
                profiles,
                args.profile_lambda,
                max_k,
            )
            prediction = np.load(
                args.prediction_template.format(seed=name),
                mmap_mode="r",
            )
            prediction_rows = np.arange(len(val_idx), dtype=np.int64)
            if name == args.audit_seed and args.strict_audit:
                keep = np.asarray(
                    [int(index) not in tune_union for index in val_idx],
                    dtype=bool,
                )
                val_idx = val_idx[keep]
                prediction_rows = prediction_rows[keep]
                neighbors = neighbors[keep]
                distance = distance[keep]

            accumulators = {
                config_key(config): empty_accumulator()
                for config in configs
            }
            for start in range(0, len(val_idx), args.batch_size):
                stop = min(start + args.batch_size, len(val_idx))
                predicted = torch.as_tensor(
                    np.array(
                        prediction[prediction_rows[start:stop]],
                        copy=True,
                    ),
                    dtype=torch.complex64,
                    device=device,
                )
                truth = torch.as_tensor(
                    np.array(channels[val_idx[start:stop]], copy=True),
                    dtype=torch.complex64,
                    device=device,
                )
                selected_pas = bilateral_target(
                    source_pas,
                    neighbors[start:stop, : args.pas_k],
                    distance[start:stop, : args.pas_k],
                    args.pas_distance_power,
                    args.pas_anchor_strength,
                    device,
                )
                selected_pdp = bilateral_target(
                    source_pdp,
                    neighbors[start:stop, : args.pdp_k],
                    distance[start:stop, : args.pdp_k],
                    args.pdp_distance_power,
                    args.pdp_anchor_strength,
                    device,
                )
                truth_pas = stable_unit(
                    pas_transform(truth, spec)
                )
                truth_pdp = stable_unit(pdp_spectrum(truth, spec))
                truth_energy = float(truth.abs().square().sum())
                for config in configs:
                    output = correct_channel(
                        predicted,
                        selected_pas,
                        selected_pdp,
                        config,
                        spec,
                        args.pas_channel_mix,
                        args.pdp_channel_mix,
                        args.pas_layout,
                    )
                    output_pas = stable_unit(
                        pas_transform(output, spec)
                    )
                    output_pdp = stable_unit(
                        pdp_spectrum(output, spec)
                    )
                    accumulator = accumulators[config_key(config)]
                    accumulator["pas"] += float(
                        (output_pas * truth_pas).sum()
                    )
                    accumulator["pdp"] += float(
                        (output_pdp * truth_pdp).sum()
                    )
                    accumulator["error"] += float(
                        (output - truth).abs().square().sum()
                    )
                    accumulator["cross"] += complex(
                        (output.conj() * truth).sum().cpu().item()
                    )
                    accumulator["prediction_energy"] += float(
                        output.abs().square().sum()
                    )
                    accumulator["truth_energy"] += truth_energy
                    accumulator["pas_count"] += (
                        output_pas.numel() // output_pas.shape[-1]
                    )
                    accumulator["pdp_count"] += (
                        output_pdp.numel() // output_pdp.shape[-1]
                    )

            scores = {}
            sufficient = {}
            for key, accumulator in accumulators.items():
                pas = float(accumulator["pas"]) / int(
                    accumulator["pas_count"]
                )
                pdp = float(accumulator["pdp"]) / int(
                    accumulator["pdp_count"]
                )
                nmse = float(accumulator["error"]) / float(
                    accumulator["truth_energy"]
                )
                scores[key] = {
                    "PAS": pas,
                    "PDP": pdp,
                    "NMSE": nmse,
                    "C": float(
                        spec.metric_weights[0] * pas
                        + spec.metric_weights[1] * pdp
                        + spec.metric_weights[2] / (1.0 + nmse)
                    ),
                }
                sufficient[key] = {
                    "cross": accumulator["cross"],
                    "prediction_energy": accumulator[
                        "prediction_energy"
                    ],
                    "truth_energy": accumulator["truth_energy"],
                }
            split_scores[name] = scores
            split_sufficient[name] = sufficient
            print(f"[radial-joint] split={name} done", flush=True)

    calibrations = {}
    for config in configs:
        key = config_key(config)
        tune_cross = sum(
            complex(split_sufficient[name][key]["cross"])
            for name in tune
        )
        tune_prediction_energy = sum(
            float(
                split_sufficient[name][key]["prediction_energy"]
            )
            for name in tune
        )
        phase = float(np.angle(tune_cross))
        phase_factor = np.exp(-1j * phase)
        scale = max(
            float(np.real(tune_cross * phase_factor))
            / max(tune_prediction_energy, 1e-30),
            0.0,
        )
        calibrations[key] = {
            "residual_global_phase": phase,
            "scale": scale,
        }
        for name in names:
            current = split_sufficient[name][key]
            cross_real = float(
                np.real(complex(current["cross"]) * phase_factor)
            )
            truth_energy = float(current["truth_energy"])
            prediction_energy = float(
                current["prediction_energy"]
            )
            nmse = (
                truth_energy
                + scale**2 * prediction_energy
                - 2.0 * scale * cross_real
            ) / max(truth_energy, 1e-30)
            score = split_scores[name][key]
            score["NMSE"] = float(nmse)
            score["C"] = float(
                spec.metric_weights[0] * score["PAS"]
                + spec.metric_weights[1] * score["PDP"]
                + spec.metric_weights[2] / (1.0 + nmse)
            )

    baseline_key = config_key(("none", 0.0, 0.0))
    rows = []
    for config in configs:
        key = config_key(config)
        tune_delta = {
            name: (
                split_scores[name][key]["C"]
                - split_scores[name][baseline_key]["C"]
            )
            for name in tune
        }
        rows.append(
            {
                "order": config[0],
                "pas_beta": config[1],
                "pdp_beta": config[2],
                **calibrations[key],
                "tune_delta": tune_delta,
                "tune_delta_mean": float(
                    np.mean(list(tune_delta.values()))
                ),
                "tune_delta_worst": float(
                    np.min(list(tune_delta.values()))
                ),
                "audit_delta": float(
                    split_scores[args.audit_seed][key]["C"]
                    - split_scores[args.audit_seed][baseline_key]["C"]
                ),
                "scores": {
                    name: split_scores[name][key] for name in names
                },
            }
        )
    rows.sort(
        key=lambda row: (
            row["tune_delta_mean"],
            row["tune_delta_worst"],
        ),
        reverse=True,
    )
    robust = [
        row
        for row in rows
        if row["tune_delta_worst"] >= 0.0
        and row["audit_delta"] >= 0.0
    ]
    payload = {
        "selection_policy": (
            "source-only radial-height neighbor targets; correction "
            "order and strengths selected on four tune folds"
        ),
        "profile_lambda": args.profile_lambda,
        "pas_layout": args.pas_layout,
        "pas": {
            "k": args.pas_k,
            "distance_power": args.pas_distance_power,
            "anchor_strength": args.pas_anchor_strength,
            "channel_mix": args.pas_channel_mix,
        },
        "pdp": {
            "k": args.pdp_k,
            "distance_power": args.pdp_distance_power,
            "anchor_strength": args.pdp_anchor_strength,
            "channel_mix": args.pdp_channel_mix,
        },
        "strict_audit": args.strict_audit,
        "prediction_template": args.prediction_template,
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
        "POSTPROJECTION_RADIAL_JOINT_DONE "
        f"order={selected['order']} "
        f"pas_beta={selected['pas_beta']:g} "
        f"pdp_beta={selected['pdp_beta']:g} "
        f"tune={selected['tune_delta_mean']:+.6f} "
        f"worst={selected['tune_delta_worst']:+.6f} "
        f"audit={selected['audit_delta']:+.6f} "
        f"out={out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
