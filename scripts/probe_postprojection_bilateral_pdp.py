#!/usr/bin/env python3
"""Probe a source-only bilateral PDP correction after channel projection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from _bootstrap import ROOT  # noqa: F401
from score_holdout import reproduce_val_indices
from sweep_rayprofile_spectrum_knn import (
    local_multiscale_radial_profiles,
    local_radial_profiles,
)
from validate_moment_projection import enforce_pdp, stable_unit
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", default="Round1_Map(2)")
    parser.add_argument(
        "--cache-dir", default="cache/teammate_knn_hvp"
    )
    parser.add_argument(
        "--prediction-template",
        default=(
            "cache/geometric_phase_neighbors_projection_transport/"
            "split_s{seed}_prediction.npy"
        ),
    )
    parser.add_argument(
        "--tune-seeds", default="1890,3716,962,1022"
    )
    parser.add_argument("--audit-seed", default="2262")
    parser.add_argument("--strict-audit", action="store_true")
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--distance-power", type=float, default=0.5)
    parser.add_argument("--anchor-strength", type=float, default=4.0)
    parser.add_argument(
        "--neighbor-profile",
        choices=(
            "none",
            "radial_height",
            "radial_combined",
            "radial_multiscale_height",
            "radial_multiscale_mean",
            "radial_multiscale_occupancy",
            "radial_multiscale_height_occupancy",
            "radial_multiscale_all",
        ),
        default="none",
    )
    parser.add_argument("--profile-lambda", type=float, default=0.0)
    parser.add_argument(
        "--target-beta-grid", default="0,0.025,0.05,0.075,0.1,0.15"
    )
    parser.add_argument(
        "--channel-mix-grid", default="0.25,0.5,0.75,1"
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--pas-layout",
        choices=("phv", "hvp", "pvh"),
        default="phv",
        help="PAS layout used for reporting the combined score",
    )
    parser.add_argument(
        "--out",
        default="docs/postprojection_bilateral_pdp/result.json",
    )
    args = parser.parse_args()

    tune = [item for item in args.tune_seeds.split(",") if item]
    names = [*tune, args.audit_seed]
    beta_grid = parse_grid(args.target_beta_grid)
    channel_mix_grid = parse_grid(args.channel_mix_grid)
    configs = [
        (beta, channel_mix)
        for beta in beta_grid
        for channel_mix in channel_mix_grid
    ]
    datadir = Path(args.datadir)
    spec = load_setup(datadir / "Round1_Setup.json")
    positions = np.load(
        datadir / "Round1_Train_Pos.npy"
    ).astype(np.float32)
    profiles = None
    if args.neighbor_profile != "none":
        points = load_point_cloud(datadir / "Round1_Map.ply")
        heightmap, x0, y0, resolution = build_heightmap(points)
        profile_builder = (
            local_multiscale_radial_profiles
            if args.neighbor_profile.startswith("radial_multiscale")
            else local_radial_profiles
        )
        profiles = profile_builder(
            positions,
            np.asarray(spec.bs_position, dtype=np.float32),
            heightmap,
            x0,
            y0,
            resolution,
        )[args.neighbor_profile]
        del points, heightmap
    tune_union = {
        int(index)
        for name in tune
        for index in reproduce_val_indices(
            len(positions), 0.1, int(name)
        )
    }
    channels = np.load(
        datadir / "Round1_Train_Channel.npy", mmap_mode="r"
    )
    device = torch.device(args.device)
    pas_transform = {
        "phv": pas_spectrum_phv,
        "hvp": pas_spectrum,
        "pvh": pas_spectrum_pvh,
    }[args.pas_layout]
    cached = np.load(
        Path(args.cache_dir) / "train_pdp.npy", mmap_mode="r"
    )
    spectra = stable_unit(
        torch.as_tensor(
            np.array(cached, copy=True),
            dtype=torch.float32,
            device=device,
        )
    )
    del cached

    all_idx = np.arange(len(positions), dtype=np.int64)
    split_scores = {}
    split_sufficient = {}
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
            if profiles is None:
                distance, local = cKDTree(
                    positions[pool_idx, :2]
                ).query(positions[val_idx, :2], k=args.k)
                distance = np.asarray(distance, dtype=np.float32)
                neighbor_idx = pool_idx[np.asarray(local)]
            else:
                xy_delta = (
                    positions[val_idx, None, :2]
                    - positions[pool_idx][None, :, :2]
                )
                spatial_distance2 = np.square(xy_delta).sum(axis=-1)
                profile_scale = profiles[pool_idx].std(
                    axis=0
                ).clip(0.05)
                profile_delta = (
                    profiles[val_idx, None]
                    - profiles[pool_idx][None]
                ) / profile_scale
                profile_distance2 = np.square(profile_delta).mean(
                    axis=-1
                )
                effective_distance2 = (
                    spatial_distance2
                    + args.profile_lambda**2 * profile_distance2
                )
                local = np.argpartition(
                    effective_distance2,
                    kth=args.k - 1,
                    axis=1,
                )[:, : args.k]
                selected_effective = np.take_along_axis(
                    effective_distance2, local, axis=1
                )
                order = np.argsort(selected_effective, axis=1)
                local = np.take_along_axis(local, order, axis=1)
                neighbor_idx = pool_idx[local]
                distance = np.sqrt(
                    np.maximum(
                        np.take_along_axis(
                            spatial_distance2, local, axis=1
                        ),
                        1e-6,
                    )
                ).astype(np.float32)
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
                distance = distance[keep]
                neighbor_idx = neighbor_idx[keep]
                prediction_rows = prediction_rows[keep]
            accumulators = {
                config: {
                    "pas": 0.0,
                    "pdp": 0.0,
                    "error": 0.0,
                    "cross": 0.0j,
                    "prediction_energy": 0.0,
                    "truth_energy": 0.0,
                    "pas_count": 0,
                    "pdp_count": 0,
                }
                for config in configs
            }
            for start in range(0, len(val_idx), args.batch_size):
                stop = min(start + args.batch_size, len(val_idx))
                predicted = torch.as_tensor(
                    np.array(
                        prediction[
                            prediction_rows[start:stop]
                        ],
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
                neighbor_t = torch.as_tensor(
                    neighbor_idx[start:stop],
                    dtype=torch.long,
                    device=device,
                )
                values = spectra[neighbor_t]
                nearest_agreement = (
                    values * values[:, :1]
                ).sum(dim=-1).mean(dim=(2, 3))
                distance_t = torch.as_tensor(
                    distance[start:stop],
                    dtype=torch.float32,
                    device=device,
                ).clamp_min(0.3)
                weight = torch.softmax(
                    -args.distance_power * distance_t.log()
                    + args.anchor_strength * nearest_agreement,
                    dim=1,
                )
                selected = stable_unit(
                    torch.einsum(
                        "bk,bkmns->bmns", weight, values
                    )
                )
                baseline_pdp = stable_unit(
                    pdp_spectrum(predicted, spec)
                )
                truth_pas = stable_unit(
                    pas_transform(truth, spec)
                )
                truth_pdp = stable_unit(pdp_spectrum(truth, spec))
                truth_energy = float(
                    truth.abs().square().sum()
                )
                for beta in beta_grid:
                    target = stable_unit(
                        (1.0 - beta) * baseline_pdp
                        + beta * selected
                    )
                    corrected = enforce_pdp(predicted, target)
                    for channel_mix in channel_mix_grid:
                        output = (
                            (1.0 - channel_mix) * predicted
                            + channel_mix * corrected
                        )
                        output_pas = stable_unit(
                            pas_transform(output, spec)
                        )
                        output_pdp = stable_unit(
                            pdp_spectrum(output, spec)
                        )
                        accumulator = accumulators[
                            (beta, channel_mix)
                        ]
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
                            output_pas.numel()
                            // output_pas.shape[-1]
                        )
                        accumulator["pdp_count"] += (
                            output_pdp.numel()
                            // output_pdp.shape[-1]
                        )
            scores = {}
            sufficient = {}
            for config, accumulator in accumulators.items():
                pas = (
                    accumulator["pas"]
                    / accumulator["pas_count"]
                )
                pdp = (
                    accumulator["pdp"]
                    / accumulator["pdp_count"]
                )
                nmse = (
                    accumulator["error"]
                    / accumulator["truth_energy"]
                )
                scores[f"{config[0]:g}_{config[1]:g}"] = {
                    "PAS": pas,
                    "PDP": pdp,
                    "NMSE": nmse,
                    "C": (
                        spec.metric_weights[0] * pas
                        + spec.metric_weights[1] * pdp
                        + spec.metric_weights[2] / (1.0 + nmse)
                    ),
                }
                sufficient[f"{config[0]:g}_{config[1]:g}"] = {
                    "cross": accumulator["cross"],
                    "prediction_energy": accumulator[
                        "prediction_energy"
                    ],
                    "truth_energy": accumulator["truth_energy"],
                }
            split_scores[name] = scores
            split_sufficient[name] = sufficient
            print(
                f"[bilateral-pdp] split={name} done",
                flush=True,
            )

    calibrations = {}
    for beta, channel_mix in configs:
        key = f"{beta:g}_{channel_mix:g}"
        tune_cross = sum(
            split_sufficient[name][key]["cross"] for name in tune
        )
        tune_prediction_energy = sum(
            split_sufficient[name][key]["prediction_energy"]
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
                np.real(current["cross"] * phase_factor)
            )
            nmse = (
                current["truth_energy"]
                + scale**2 * current["prediction_energy"]
                - 2.0 * scale * cross_real
            ) / max(current["truth_energy"], 1e-30)
            score = split_scores[name][key]
            score["NMSE"] = float(nmse)
            score["C"] = float(
                spec.metric_weights[0] * score["PAS"]
                + spec.metric_weights[1] * score["PDP"]
                + spec.metric_weights[2] / (1.0 + nmse)
            )

    rows = []
    for beta, channel_mix in configs:
        key = f"{beta:g}_{channel_mix:g}"
        baseline_key = f"0_{channel_mix:g}"
        tune_delta = {
            name: (
                split_scores[name][key]["C"]
                - split_scores[name][baseline_key]["C"]
            )
            for name in tune
        }
        rows.append(
            {
                "target_beta": beta,
                "channel_mix": channel_mix,
                **calibrations[key],
                "tune_delta": tune_delta,
                "tune_delta_mean": float(
                    np.mean(list(tune_delta.values()))
                ),
                "tune_delta_worst": float(
                    np.min(list(tune_delta.values()))
                ),
                "audit_delta": (
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
            "bilateral target uses source PDPs only; beta and channel "
            "mix selected on four tune folds"
        ),
        "distance_power": args.distance_power,
        "anchor_strength": args.anchor_strength,
        "neighbor_profile": args.neighbor_profile,
        "profile_lambda": args.profile_lambda,
        "prediction_template": args.prediction_template,
        "pas_layout": args.pas_layout,
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
        "POSTPROJECTION_BILATERAL_PDP_DONE "
        f"beta={selected['target_beta']:g} "
        f"mix={selected['channel_mix']:g} "
        f"tune={selected['tune_delta_mean']:+.6f} "
        f"worst={selected['tune_delta_worst']:+.6f} "
        f"audit={selected['audit_delta']:+.6f} "
        f"out={out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
