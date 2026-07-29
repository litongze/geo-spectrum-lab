#!/usr/bin/env python3
"""Validate multi-neighbor geometric phase with PAS/PDP projections."""
from __future__ import annotations

import argparse
import json
import re
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
from neighbor_source_weights import (
    anisotropic_weights,
    equalize_amplitude,
    moment_correct_weights,
)
from probe_full_array_steering_phase import array_steering_phase
from score_holdout import reproduce_val_indices
from train_covariance_kriging import CorrelationPredictor
from validate_moment_projection import (
    enforce_pas,
    enforce_pas_layout,
    enforce_pdp,
    stable_unit,
)
from wireless_twin.data.setup_config import load_setup
from wireless_twin.signal import (
    pas_spectrum,
    pas_spectrum_phv,
    pas_spectrum_pvh,
    pdp_spectrum,
)


ANISOTROPIC = re.compile(
    r"^anis_k(?P<k>\d+)_r(?P<ratio>[0-9.]+)"
    r"_p(?P<power>[0-9.]+)"
    r"(?:_moment_r(?P<ridge>[0-9.]+)"
    r"_a(?P<strength>[0-9.]+))?"
    r"_(?P<amplitude>raw|equalized)$"
)


def parse_source(name: str) -> dict:
    if name == "rank1_raw":
        return {
            "name": name,
            "k": 1,
            "ratio": 1.0,
            "power": 0.0,
            "amplitude": "raw",
        }
    match = ANISOTROPIC.match(name)
    if match is None:
        raise ValueError(f"unsupported source configuration: {name}")
    source = {
        "name": name,
        "k": int(match.group("k")),
        "ratio": float(match.group("ratio")),
        "power": float(match.group("power")),
        "amplitude": match.group("amplitude"),
    }
    if match.group("ridge") is not None:
        source["moment"] = {
            "ridge": float(match.group("ridge")),
            "strength": float(match.group("strength")),
        }
        source["phase_lookup_name"] = (
            f"k{source['k']}_r{source['ratio']:g}"
            f"_p{source['power']:g}"
            f"_moment_r{source['moment']['ridge']:g}"
            f"_a{source['moment']['strength']:g}"
            f"_none_{source['amplitude']}"
        )
    return source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", default="Round1_Map(2)")
    parser.add_argument(
        "--config", default="configs/gs37_geophase_moment.json"
    )
    parser.add_argument(
        "--baseline-dir",
        default="docs/clean_noeps_panel_phv_geom_k16",
    )
    parser.add_argument(
        "--expert-dir", default="cache/moment_attention_selected"
    )
    parser.add_argument(
        "--neighbor-result",
        default="docs/geometric_phase_neighbors_k16/result.json",
    )
    parser.add_argument("--steering-result")
    parser.add_argument(
        "--sources",
        default=(
            "rank1_raw,"
            "anis_k16_r4_p2.5_equalized,"
            "anis_k16_r3_p2.5_equalized,"
            "anis_k12_r3_p2.5_equalized,"
            "anis_k16_r4_p3_equalized"
        ),
    )
    parser.add_argument(
        "--complex-gate-dir",
        help=(
            "optional directory containing clean s{seed}.pt complex "
            "neighbor gates"
        ),
    )
    parser.add_argument(
        "--complex-gate-source",
        default="anis_k16_r5_p2_equalized",
        help="source configuration used as the complex-gate baseline",
    )
    parser.add_argument(
        "--complex-gate-alpha", type=float, default=0.7
    )
    parser.add_argument(
        "--complex-gate-secondary-dir",
        help=(
            "optional second clean gate family, added as another "
            "coefficient residual from the same baseline"
        ),
    )
    parser.add_argument(
        "--complex-gate-secondary-alpha",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--covariance-kriging-dir",
        help=(
            "optional directory containing clean s{seed}.pt complex "
            "correlation predictors"
        ),
    )
    parser.add_argument(
        "--covariance-kriging-source",
        default="anis_k16_r5_p2_equalized",
    )
    parser.add_argument(
        "--covariance-kriging-loading", type=float, default=1.0
    )
    parser.add_argument(
        "--covariance-kriging-blend-grid", default="0.5"
    )
    parser.add_argument(
        "--covariance-kriging-hybrid-grid",
        default="",
        help=(
            "optional residual fractions added to the complex-gate source: "
            "gate + alpha * (kriging - baseline)"
        ),
    )
    parser.add_argument(
        "--angle-delay-source",
        help=(
            "optional source configuration replaced by grouped PHV "
            "angle-delay amplitude consensus"
        ),
    )
    parser.add_argument(
        "--angle-delay-group",
        choices=("pol", "path"),
        default="path",
    )
    parser.add_argument(
        "--angle-delay-gamma", type=float, default=0.75
    )
    parser.add_argument(
        "--angle-delay-blend", type=float, default=1.0
    )
    parser.add_argument(
        "--angle-delay-ratio-min", type=float, default=0.25
    )
    parser.add_argument(
        "--angle-delay-ratio-max", type=float, default=4.0
    )
    parser.add_argument("--tune-seeds", default="1890,3716,962,1022")
    parser.add_argument("--audit-seed", type=int, default=2262)
    parser.add_argument(
        "--strict-audit",
        action="store_true",
        help="score audit only on indices absent from every tune split",
    )
    parser.add_argument("--pas-blend", type=float, default=0.8)
    parser.add_argument("--pdp-blend", type=float, default=0.3)
    parser.add_argument("--pas-transport-blend", type=float, default=0.0)
    parser.add_argument("--pas-transport-k", type=int, default=16)
    parser.add_argument("--pas-transport-power", type=float, default=2.0)
    parser.add_argument(
        "--pas-transport-gate-dir",
        help=(
            "optional directory containing clean beta_{seed}.npy gates; "
            "replaces the constant transport blend"
        ),
    )
    parser.add_argument(
        "--hvp-expert-dir",
        help="optional directory containing pas_{seed}_c0.5.npy HVP experts",
    )
    parser.add_argument(
        "--hvp-blend-grid", default="0.2,0.4,0.6,0.8,1"
    )
    parser.add_argument(
        "--dual-orders",
        default=(
            "phv_hvp_pdp,hvp_phv_pdp,"
            "phv_pdp_hvp,hvp_pdp_phv"
        ),
    )
    parser.add_argument(
        "--dual-blend-grid",
        default="",
        help=(
            "optional complex blends from the pas_pdp max-iteration "
            "reference to each dual candidate"
        ),
    )
    parser.add_argument(
        "--source-blend-grid",
        default="",
        help=(
            "optional fractions of each projected channel mixed into "
            "its unprojected complex source"
        ),
    )
    parser.add_argument("--phase-seed-epsilon", type=float, default=0.001)
    parser.add_argument("--iteration-grid", default="1,2,3,5,8")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--report-pas-layouts",
        action="store_true",
        help="also report HVP and PVH PAS scores without changing selection",
    )
    parser.add_argument(
        "--score-pas-layout",
        choices=("phv", "hvp", "pvh"),
        default="phv",
        help="PAS layout used for candidate ranking and the reported C score",
    )
    parser.add_argument(
        "--save-prediction",
        help=(
            "exact candidate name whose raw projected channels should be "
            "saved for downstream diagnostics"
        ),
    )
    parser.add_argument(
        "--prediction-dir",
        default="cache/geometric_phase_neighbors_projection",
    )
    parser.add_argument(
        "--out",
        default=(
            "docs/geometric_phase_neighbors_projection/result.json"
        ),
    )
    args = parser.parse_args()

    source_specs = [
        parse_source(value)
        for value in args.sources.split(",")
        if value.strip()
    ]
    if args.complex_gate_secondary_dir and not args.complex_gate_dir:
        raise ValueError(
            "secondary complex gate requires --complex-gate-dir"
        )
    for label, alpha in (
        ("primary", args.complex_gate_alpha),
        ("secondary", args.complex_gate_secondary_alpha),
    ):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(
                f"{label} complex gate alpha must be in [0, 1]"
            )
    if args.complex_gate_dir:
        matching_sources = [
            source
            for source in source_specs
            if source["name"] == args.complex_gate_source
        ]
        if len(matching_sources) != 1:
            raise ValueError(
                "--complex-gate-source must occur exactly once in "
                "--sources"
            )
        gate_source = dict(matching_sources[0])
        gate_source["gate_base_name"] = gate_source["name"]
        gate_source["name"] = (
            f"{gate_source['name']}_cgate"
            f"{args.complex_gate_alpha:g}"
            + (
                f"_sgate{args.complex_gate_secondary_alpha:g}"
                if args.complex_gate_secondary_dir
                else ""
            )
        )
        source_specs.append(gate_source)
    kriging_sources = []
    if args.covariance_kriging_dir:
        matching_sources = [
            source
            for source in source_specs
            if source["name"] == args.covariance_kriging_source
        ]
        if len(matching_sources) != 1:
            raise ValueError(
                "--covariance-kriging-source must occur exactly once "
                "in --sources"
            )
        for blend_text in args.covariance_kriging_blend_grid.split(","):
            if not blend_text:
                continue
            blend = float(blend_text)
            if not 0.0 <= blend <= 1.0:
                raise ValueError(
                    "covariance kriging blend must be in [0, 1]"
                )
            kriging_source = dict(matching_sources[0])
            kriging_source["kriging_base_name"] = (
                kriging_source["name"]
            )
            kriging_source["kriging_blend"] = blend
            kriging_source["name"] = (
                f"{kriging_source['name']}_ckrig_l"
                f"{args.covariance_kriging_loading:g}_b{blend:g}"
            )
            source_specs.append(kriging_source)
            kriging_sources.append(kriging_source)
    if args.covariance_kriging_hybrid_grid:
        if not args.complex_gate_dir or not kriging_sources:
            raise ValueError(
                "covariance kriging hybrids require both gate and "
                "kriging sources"
            )
        for alpha_text in args.covariance_kriging_hybrid_grid.split(","):
            if not alpha_text:
                continue
            alpha = float(alpha_text)
            if not 0.0 <= alpha <= 1.0:
                raise ValueError(
                    "covariance kriging hybrid alpha must be in [0, 1]"
                )
            for kriging_source in kriging_sources:
                hybrid_source = dict(matching_sources[0])
                hybrid_source["phase_lookup_name"] = (
                    matching_sources[0]["name"]
                )
                hybrid_source["hybrid_components"] = {
                    "gate": gate_source["name"],
                    "kriging": kriging_source["name"],
                    "baseline": matching_sources[0]["name"],
                    "alpha": alpha,
                }
                hybrid_source["name"] = (
                    f"{gate_source['name']}_ckres"
                    f"{kriging_source['kriging_blend']:g}"
                    f"_a{alpha:g}"
                )
                source_specs.append(hybrid_source)
    if args.angle_delay_source:
        matching_sources = [
            source
            for source in source_specs
            if source["name"] == args.angle_delay_source
        ]
        if len(matching_sources) != 1:
            raise ValueError(
                "--angle-delay-source must occur exactly once in "
                "--sources before derived sources are added"
            )
        angle_delay_source = dict(matching_sources[0])
        angle_delay_source["angle_delay_base_name"] = (
            angle_delay_source["name"]
        )
        angle_delay_source["name"] = (
            f"{angle_delay_source['name']}_ad"
            f"{args.angle_delay_group}_g"
            f"{args.angle_delay_gamma:g}_b"
            f"{args.angle_delay_blend:g}"
        )
        source_specs.append(angle_delay_source)
    k_max = max(
        max(source["k"] for source in source_specs),
        args.pas_transport_k if args.pas_transport_blend > 0 else 1,
    )
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
    iterations = [
        int(value) for value in args.iteration_grid.split(",") if value
    ]
    hvp_blends = [
        float(value)
        for value in args.hvp_blend_grid.split(",")
        if value
    ]
    dual_orders = [
        value for value in args.dual_orders.split(",") if value
    ]
    dual_blends = [
        float(value)
        for value in args.dual_blend_grid.split(",")
        if value
    ]
    source_blends = [
        float(value)
        for value in args.source_blend_grid.split(",")
        if value
    ]
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    k0 = float(config["phase"]["k0_rad_per_meter"])
    k1 = float(
        config["phase"]["k1_rad_per_meter_per_subcarrier"]
    )
    neighbor_payload = json.loads(
        Path(args.neighbor_result).read_text(encoding="utf-8")
    )
    neighbor_rows = {
        row["name"]: row for row in neighbor_payload["ranked"]
    }
    for source in source_specs:
        neighbor_name = source.get(
            "phase_lookup_name",
            source.get(
                "kriging_base_name",
                source.get(
                    "gate_base_name",
                    source.get(
                        "angle_delay_base_name", source["name"]
                    ),
                ),
            ),
        )
        source["initial_global_phase"] = float(
            neighbor_rows[neighbor_name]["global_phase"]
        )
    steering = None
    if args.steering_result:
        steering_payload = json.loads(
            Path(args.steering_result).read_text(encoding="utf-8")
        )
        steering = steering_payload["ranked"][0]

    device = torch.device(args.device)
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
    sufficient = {}
    score_pas_transform = {
        "phv": pas_spectrum_phv,
        "hvp": pas_spectrum,
        "pvh": pas_spectrum_pvh,
    }[args.score_pas_layout]

    def record(
        name: str,
        seed: int,
        prediction: torch.Tensor,
        truth: torch.Tensor,
        truth_pas: torch.Tensor,
        truth_pdp: torch.Tensor,
        keep: torch.Tensor | None,
    ) -> None:
        if name == args.save_prediction:
            prediction_dir = Path(args.prediction_dir)
            prediction_dir.mkdir(parents=True, exist_ok=True)
            np.save(
                prediction_dir / f"split_s{seed}_prediction.npy",
                prediction.detach()
                .cpu()
                .numpy()
                .astype(np.complex64),
            )
        if keep is not None:
            prediction = prediction[keep]
            truth = truth[keep]
            truth_pas = truth_pas[keep]
            truth_pdp = truth_pdp[keep]
        prediction_pas = stable_unit(
            score_pas_transform(prediction, spec)
        )
        prediction_pdp = stable_unit(pdp_spectrum(prediction, spec))
        sufficient.setdefault(name, {})[seed] = {
            "PAS": float(
                (prediction_pas * truth_pas).sum(dim=-1).mean()
            ),
            "PDP": float(
                (prediction_pdp * truth_pdp).sum(dim=-1).mean()
            ),
            "cross": complex(
                (prediction.conj() * truth).sum().cpu().item()
            ),
            "prediction_energy": float(
                prediction.abs().square().sum()
            ),
            "truth_energy": float(truth.abs().square().sum()),
        }
        if args.report_pas_layouts:
            row = sufficient[name][seed]
            for layout, transform in (
                ("hvp", pas_spectrum),
                ("pvh", pas_spectrum_pvh),
            ):
                pred_layout = stable_unit(transform(prediction, spec))
                truth_layout = stable_unit(transform(truth, spec))
                row[f"PAS_{layout}"] = float(
                    (pred_layout * truth_layout).sum(dim=-1).mean()
                )

    for seed in seeds:
        val_idx = np.asarray(
            sorted(reproduce_val_indices(len(positions), 0.1, seed)),
            dtype=np.int64,
        )
        keep = None
        if seed == args.audit_seed and args.strict_audit:
            keep = torch.as_tensor(
                [int(index) not in tune_union for index in val_idx],
                dtype=torch.bool,
                device=device,
            )
        pool_idx = np.setdiff1d(all_idx, val_idx)
        distance, local = cKDTree(positions[pool_idx, :2]).query(
            positions[val_idx, :2], k=k_max
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
        delta_radius = (
            radius[val_idx, None] - radius[neighbor_idx]
        )
        delta_unit = unit[val_idx, None] - unit[neighbor_idx]
        gate_components = []
        if args.complex_gate_dir:
            for role, directory, alpha in (
                (
                    "primary",
                    args.complex_gate_dir,
                    args.complex_gate_alpha,
                ),
                (
                    "secondary",
                    args.complex_gate_secondary_dir,
                    args.complex_gate_secondary_alpha,
                ),
            ):
                if not directory or alpha == 0.0:
                    continue
                gate_checkpoint = torch.load(
                    Path(directory) / f"s{seed}.pt",
                    map_location=device,
                    weights_only=False,
                )
                checkpoint_names = tuple(
                    gate_checkpoint["feature_names"]
                )
                model = ComplexNeighborGate(
                    len(checkpoint_names),
                    gate_checkpoint.get(
                        "architecture", "mlp16"
                    ),
                ).to(device)
                model.load_state_dict(
                    gate_checkpoint["model_state"]
                )
                model.eval()
                gate_components.append(
                    {
                        "role": role,
                        "alpha": alpha,
                        "indices": [
                            FEATURE_NAMES.index(item)
                            for item in checkpoint_names
                        ],
                        "model": model,
                        "mean": torch.as_tensor(
                            gate_checkpoint["feature_mean"],
                            dtype=torch.float32,
                            device=device,
                        ),
                        "std": torch.as_tensor(
                            gate_checkpoint["feature_std"],
                            dtype=torch.float32,
                            device=device,
                        ),
                    }
                )
        kriging_component = None
        if args.covariance_kriging_dir:
            kriging_checkpoint = torch.load(
                Path(args.covariance_kriging_dir) / f"s{seed}.pt",
                map_location=device,
                weights_only=False,
            )
            kriging_names = tuple(
                kriging_checkpoint["feature_names"]
            )
            kriging_model = CorrelationPredictor(
                int(kriging_checkpoint["feature_dim"]),
                int(kriging_checkpoint["hidden_dim"]),
            ).to(device)
            kriging_model.load_state_dict(
                kriging_checkpoint["model_state"]
            )
            kriging_model.eval()
            kriging_component = {
                "model": kriging_model,
                "indices": [
                    FEATURE_NAMES.index(item)
                    for item in kriging_names
                ],
                "mean": torch.as_tensor(
                    kriging_checkpoint["feature_mean"],
                    dtype=torch.float32,
                    device=device,
                ),
                "std": torch.as_tensor(
                    kriging_checkpoint["feature_std"],
                    dtype=torch.float32,
                    device=device,
                ),
            }

        truth = torch.as_tensor(
            np.array(channels[val_idx], copy=True),
            dtype=torch.complex64,
            device=device,
        )
        sources = {
            source["name"]: torch.zeros_like(truth)
            for source in source_specs
        }
        transported_pas = (
            torch.zeros(
                (
                    len(val_idx),
                    spec.n,
                    spec.s,
                    spec.mh * spec.mv,
                ),
                dtype=torch.float32,
                device=device,
            )
            if args.pas_transport_blend > 0
            else None
        )
        with torch.inference_mode():
            for start in range(0, len(val_idx), args.batch_size):
                stop = min(start + args.batch_size, len(val_idx))
                batch = stop - start
                neighbors = torch.as_tensor(
                    np.array(
                        channels[neighbor_idx[start:stop].reshape(-1)],
                        copy=True,
                    ).reshape(
                        batch,
                        k_max,
                        spec.m,
                        spec.n,
                        spec.s,
                    ),
                    dtype=torch.complex64,
                    device=device,
                )
                phase = (
                    (
                        k0 + k1 * subcarrier[None, None]
                    )
                    * delta_radius[start:stop, :, None]
                )
                neighbors *= torch.as_tensor(
                    np.exp(1j * phase),
                    dtype=torch.complex64,
                    device=device,
                )[:, :, None, None, :]
                if steering is not None:
                    steering_phase = array_steering_phase(
                        delta_unit[start:stop], spec, steering
                    )
                    neighbors *= torch.as_tensor(
                        np.exp(1j * steering_phase),
                        dtype=torch.complex64,
                        device=device,
                    )[..., None]
                if transported_pas is not None:
                    transport_k = args.pas_transport_k
                    neighbor_pas = stable_unit(
                        pas_spectrum_phv(
                            neighbors[:, :transport_k].reshape(
                                batch * transport_k,
                                spec.m,
                                spec.n,
                                spec.s,
                            ),
                            spec,
                        )
                    ).reshape(
                        batch,
                        transport_k,
                        spec.n,
                        spec.s,
                        spec.mh * spec.mv,
                    )
                    transport_weight = (
                        1.0
                        / np.maximum(
                            distance[start:stop, :transport_k],
                            0.05,
                        )
                        ** args.pas_transport_power
                    )
                    transport_weight /= np.maximum(
                        transport_weight.sum(
                            axis=1, keepdims=True
                        ),
                        1e-30,
                    )
                    transported_pas[start:stop] = stable_unit(
                        torch.einsum(
                            "bk,bknsa->bnsa",
                            torch.as_tensor(
                                transport_weight,
                                dtype=torch.float32,
                                device=device,
                            ),
                            neighbor_pas,
                        )
                    )
                energy = (
                    neighbors.abs()
                    .square()
                    .sum(dim=(2, 3, 4))
                    .cpu()
                    .numpy()
                )
                for source in source_specs:
                    if "hybrid_components" in source:
                        continue
                    k = source["k"]
                    if k == 1:
                        weight = np.ones((batch, 1), dtype=np.float64)
                    else:
                        weight, effective = anisotropic_weights(
                            radial_delta[start:stop, :k],
                            tangent_delta[start:stop, :k],
                            source["ratio"],
                            source["power"],
                        )
                        if "moment" in source:
                            weight = moment_correct_weights(
                                weight,
                                radial_delta[start:stop, :k],
                                tangent_delta[start:stop, :k],
                                **source["moment"],
                            )
                    if "angle_delay_base_name" in source:
                        base_weight_t = torch.as_tensor(
                            weight,
                            dtype=torch.float32,
                            device=device,
                        )
                        amplitude = torch.as_tensor(
                            np.sqrt(
                                np.maximum(energy[:, :k], 1e-30)
                            ),
                            dtype=torch.float32,
                            device=device,
                        )
                        target_amplitude = (
                            base_weight_t * amplitude
                        ).sum(dim=1, keepdim=True)
                        baseline_weight = (
                            base_weight_t
                            * target_amplitude
                            / amplitude.clamp_min(1e-30)
                        ).to(torch.complex64)
                        shaped = neighbors[:, :k].reshape(
                            batch,
                            k,
                            spec.mp,
                            spec.mh,
                            spec.mv,
                            spec.n,
                            spec.s,
                        )
                        transformed = torch.fft.ifft(
                            torch.fft.fft2(
                                shaped,
                                dim=(3, 4),
                                norm="ortho",
                            ),
                            dim=-1,
                            norm="ortho",
                        )
                        reduce_dims = (
                            (2,)
                            if args.angle_delay_group == "pol"
                            else (2, 5)
                        )
                        group_amplitude = (
                            transformed.abs()
                            .square()
                            .sum(
                                dim=reduce_dims,
                                keepdim=True,
                            )
                            .clamp_min(1e-30)
                            .sqrt()
                        )
                        broadcast_weight = base_weight_t[
                            :, :, None, None, None, None, None
                        ]
                        target_group_amplitude = (
                            broadcast_weight * group_amplitude
                        ).sum(dim=1, keepdim=True)
                        ratio = (
                            target_group_amplitude
                            / group_amplitude.clamp_min(1e-30)
                        ).clamp(
                            args.angle_delay_ratio_min,
                            args.angle_delay_ratio_max,
                        )
                        consensus = (
                            broadcast_weight
                            * transformed
                            * ratio.pow(args.angle_delay_gamma)
                        ).sum(dim=1)
                        consensus_channel = torch.fft.ifft2(
                            torch.fft.fft(
                                consensus,
                                dim=-1,
                                norm="ortho",
                            ),
                            dim=(2, 3),
                            norm="ortho",
                        ).reshape(
                            batch, spec.m, spec.n, spec.s
                        )
                        baseline_channel = torch.einsum(
                            "bk,bkmns->bmns",
                            baseline_weight,
                            neighbors[:, :k],
                        )
                        sources[source["name"]][start:stop] = (
                            (1.0 - args.angle_delay_blend)
                            * baseline_channel
                            + args.angle_delay_blend
                            * consensus_channel
                        )
                        continue
                    if "kriging_base_name" in source:
                        if kriging_component is None:
                            raise RuntimeError(
                                "covariance kriging model was not loaded"
                            )
                        source_flat = neighbors[:, :k].reshape(
                            batch, k, -1
                        )
                        gram = torch.einsum(
                            "bil,bjl->bij",
                            source_flat.conj(),
                            source_flat,
                        )
                        base_weight_t = torch.as_tensor(
                            weight,
                            dtype=torch.float32,
                            device=device,
                        )
                        gate_features, gate_amplitude = (
                            build_complex_features(
                                gram,
                                base_weight_t,
                                torch.as_tensor(
                                    distance[start:stop, :k],
                                    dtype=torch.float32,
                                    device=device,
                                ),
                                torch.as_tensor(
                                    effective,
                                    dtype=torch.float32,
                                    device=device,
                                ),
                                torch.as_tensor(
                                    radial_delta[start:stop, :k],
                                    dtype=torch.float32,
                                    device=device,
                                ),
                                torch.as_tensor(
                                    tangent_delta[start:stop, :k],
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
                        correlation = gram / (
                            gate_amplitude[:, :, None]
                            * gate_amplitude[:, None, :]
                        ).clamp_min(1e-30)
                        prior = torch.einsum(
                            "bij,bj->bi",
                            correlation,
                            base_weight_t.to(torch.complex64),
                        )
                        normalized_features = (
                            (
                                gate_features[
                                    ..., kriging_component["indices"]
                                ]
                                - kriging_component["mean"]
                            )
                            / kriging_component["std"]
                        ).clamp(-8.0, 8.0)
                        predicted_correlation = kriging_component[
                            "model"
                        ](normalized_features, prior)
                        loading = args.covariance_kriging_loading
                        identity = torch.eye(
                            k,
                            dtype=torch.complex64,
                            device=device,
                        )[None]
                        solved = torch.linalg.solve(
                            correlation + loading * identity,
                            (
                                predicted_correlation
                                + loading
                                * base_weight_t.to(torch.complex64)
                            )[..., None],
                        )[..., 0]
                        blend = float(source["kriging_blend"])
                        normalized_weight = (
                            base_weight_t.to(torch.complex64)
                            + blend
                            * (
                                solved
                                - base_weight_t.to(torch.complex64)
                            )
                        )
                        target_amplitude = (
                            base_weight_t * gate_amplitude
                        ).sum(dim=1, keepdim=True)
                        weight_t = (
                            normalized_weight
                            * target_amplitude
                            / gate_amplitude.clamp_min(1e-30)
                        )
                    elif "gate_base_name" in source:
                        if not gate_components:
                            raise RuntimeError(
                                "complex gate was not loaded"
                            )
                        source_flat = neighbors[:, :k].reshape(
                            batch, k, -1
                        )
                        gram = torch.einsum(
                            "bil,bjl->bij",
                            source_flat.conj(),
                            source_flat,
                        )
                        base_weight_t = torch.as_tensor(
                            weight,
                            dtype=torch.float32,
                            device=device,
                        )
                        gate_features, gate_amplitude = (
                            build_complex_features(
                                gram,
                                base_weight_t,
                                torch.as_tensor(
                                    distance[start:stop, :k],
                                    dtype=torch.float32,
                                    device=device,
                                ),
                                torch.as_tensor(
                                    effective,
                                    dtype=torch.float32,
                                    device=device,
                                ),
                                torch.as_tensor(
                                    radial_delta[start:stop, :k],
                                    dtype=torch.float32,
                                    device=device,
                                ),
                                torch.as_tensor(
                                    tangent_delta[start:stop, :k],
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
                        baseline_weight = baseline_coefficients(
                            base_weight_t, gate_amplitude
                        )
                        weight_t = baseline_weight
                        for component in gate_components:
                            normalized_features = (
                                (
                                    gate_features[
                                        ..., component["indices"]
                                    ]
                                    - component["mean"]
                                )
                                / component["std"]
                            ).clamp(-8.0, 8.0)
                            learned_weight, _ = component[
                                "model"
                            ].coefficients(
                                normalized_features,
                                base_weight_t,
                                gate_amplitude,
                            )
                            if component["role"] == "primary":
                                weight_t = (
                                    (1.0 - component["alpha"])
                                    * baseline_weight
                                    + component["alpha"]
                                    * learned_weight
                                )
                            else:
                                weight_t += component["alpha"] * (
                                    learned_weight - baseline_weight
                                )
                    elif source["amplitude"] == "equalized":
                        weight = equalize_amplitude(
                            weight,
                            np.sqrt(
                                np.maximum(energy[:, :k], 1e-30)
                            ),
                        )
                        weight_t = torch.as_tensor(
                            weight,
                            dtype=torch.complex64,
                            device=device,
                        )
                    else:
                        weight_t = torch.as_tensor(
                            weight,
                            dtype=torch.complex64,
                            device=device,
                        )
                    sources[source["name"]][start:stop] = (
                        torch.einsum(
                            "bk,bkmns->bmns",
                            weight_t,
                            neighbors[:, :k],
                        )
                    )
                del neighbors
        for source in source_specs:
            components = source.get("hybrid_components")
            if components is None:
                continue
            sources[source["name"]] = (
                sources[components["gate"]]
                + float(components["alpha"])
                * (
                    sources[components["kriging"]]
                    - sources[components["baseline"]]
                )
            )

        baseline = torch.as_tensor(
            np.array(
                np.load(
                    Path(args.baseline_dir)
                    / f"split_s{seed}_prediction.npy",
                    mmap_mode="r",
                ),
                copy=True,
            ),
            dtype=torch.complex64,
            device=device,
        )
        baseline_pas = stable_unit(pas_spectrum_phv(baseline, spec))
        baseline_pdp = stable_unit(pdp_spectrum(baseline, spec))
        expert_pas = torch.as_tensor(
            np.load(
                Path(args.expert_dir) / f"pas_{seed}_c1.npy"
            ),
            dtype=torch.float32,
            device=device,
        )
        if transported_pas is not None:
            transport_beta: float | torch.Tensor = (
                args.pas_transport_blend
            )
            if args.pas_transport_gate_dir:
                beta_path = (
                    Path(args.pas_transport_gate_dir)
                    / f"beta_{seed}.npy"
                )
                beta_np = np.load(beta_path)
                expected_beta_shape = expert_pas.shape[:-1]
                if beta_np.shape != expected_beta_shape:
                    raise ValueError(
                        f"{beta_path}: expected beta shape "
                        f"{expected_beta_shape}, got {beta_np.shape}"
                    )
                transport_beta = torch.as_tensor(
                    beta_np,
                    dtype=torch.float32,
                    device=device,
                )[..., None]
            expert_pas = stable_unit(
                (1.0 - transport_beta) * expert_pas
                + transport_beta * transported_pas
            )
        expert_pdp = torch.as_tensor(
            np.load(
                Path(args.expert_dir) / f"pdp_{seed}_c0.8.npy"
            ),
            dtype=torch.float32,
            device=device,
        )
        target_pas = stable_unit(
            (1.0 - args.pas_blend) * baseline_pas
            + args.pas_blend * expert_pas
        )
        target_pdp = stable_unit(
            (1.0 - args.pdp_blend) * baseline_pdp
            + args.pdp_blend * expert_pdp
        )
        target_hvp_by_blend = {}
        if args.hvp_expert_dir:
            baseline_hvp = stable_unit(pas_spectrum(baseline, spec))
            hvp_expert_path = (
                Path(args.hvp_expert_dir)
                / f"pas_{seed}_c0.5.npy"
            )
            expert_hvp = stable_unit(
                torch.as_tensor(
                    np.load(hvp_expert_path),
                    dtype=torch.float32,
                    device=device,
                )
            )
            for hvp_blend in hvp_blends:
                target_hvp_by_blend[hvp_blend] = stable_unit(
                    (1.0 - hvp_blend) * baseline_hvp
                    + hvp_blend * expert_hvp
                )
        truth_pas = stable_unit(score_pas_transform(truth, spec))
        truth_pdp = stable_unit(pdp_spectrum(truth, spec))
        record(
            "baseline_none_i0",
            seed,
            baseline,
            truth,
            truth_pas,
            truth_pdp,
            keep,
        )

        for source in source_specs:
            source_name = source["name"]
            prediction_source = (
                sources[source_name]
                * torch.exp(
                    torch.tensor(
                        1j * source["initial_global_phase"],
                        dtype=torch.complex64,
                        device=device,
                    )
                )
            )
            source_rms = (
                prediction_source.abs().square().mean().sqrt()
            )
            prediction_source = prediction_source + (
                args.phase_seed_epsilon
                * baseline
                * source_rms
                / baseline.abs()
                .square()
                .mean()
                .sqrt()
                .clamp_min(1e-30)
            )
            record(
                f"{source_name}_none_i0",
                seed,
                prediction_source,
                truth,
                truth_pas,
                truth_pdp,
                keep,
            )
            dual_reference = None
            for order in ("pas_pdp", "pdp_pas"):
                prediction = prediction_source.clone()
                for iteration in range(1, max(iterations) + 1):
                    if order == "pas_pdp":
                        prediction = enforce_pas(
                            prediction, target_pas, spec
                        )
                        prediction = enforce_pdp(
                            prediction, target_pdp
                        )
                    else:
                        prediction = enforce_pdp(
                            prediction, target_pdp
                        )
                        prediction = enforce_pas(
                            prediction, target_pas, spec
                        )
                    if iteration in iterations:
                        candidate_name = (
                            f"{source_name}_{order}_i{iteration}"
                        )
                        record(
                            candidate_name,
                            seed,
                            prediction,
                            truth,
                            truth_pas,
                            truth_pdp,
                            keep,
                        )
                        for source_blend in source_blends:
                            mixed_with_source = (
                                (1.0 - source_blend)
                                * prediction_source
                                + source_blend * prediction
                            )
                            record(
                                candidate_name
                                + f"_src{source_blend:g}",
                                seed,
                                mixed_with_source,
                                truth,
                                truth_pas,
                                truth_pdp,
                                keep,
                            )
                        if (
                            order == "pas_pdp"
                            and iteration == max(iterations)
                        ):
                            dual_reference = prediction.clone()
                del prediction
            for hvp_blend, target_hvp in target_hvp_by_blend.items():
                for dual_order in dual_orders:
                    steps = dual_order.split("_")
                    if sorted(steps) not in (
                        ["hvp", "pdp"],
                        ["hvp", "pdp", "phv"],
                    ):
                        raise ValueError(
                            f"invalid dual projection order {dual_order}"
                        )
                    prediction = prediction_source.clone()
                    for iteration in range(
                        1, max(iterations) + 1
                    ):
                        for step in steps:
                            if step == "phv":
                                prediction = enforce_pas(
                                    prediction, target_pas, spec
                                )
                            elif step == "hvp":
                                prediction = enforce_pas_layout(
                                    prediction,
                                    target_hvp,
                                    spec,
                                    "hvp",
                                )
                            else:
                                prediction = enforce_pdp(
                                    prediction, target_pdp
                                )
                        if iteration in iterations:
                            candidate_name = (
                                f"{source_name}_dual_{dual_order}"
                                f"_hb{hvp_blend:g}_i{iteration}"
                            )
                            record(
                                candidate_name,
                                seed,
                                prediction,
                                truth,
                                truth_pas,
                                truth_pdp,
                                keep,
                            )
                            for source_blend in source_blends:
                                mixed_with_source = (
                                    (1.0 - source_blend)
                                    * prediction_source
                                    + source_blend * prediction
                                )
                                record(
                                    candidate_name
                                    + f"_src{source_blend:g}",
                                    seed,
                                    mixed_with_source,
                                    truth,
                                    truth_pas,
                                    truth_pdp,
                                    keep,
                                )
                            if dual_reference is not None:
                                for dual_blend in dual_blends:
                                    mixed_prediction = (
                                        (1.0 - dual_blend)
                                        * dual_reference
                                        + dual_blend * prediction
                                    )
                                    record(
                                        candidate_name
                                        + f"_mix{dual_blend:g}",
                                        seed,
                                        mixed_prediction,
                                        truth,
                                        truth_pas,
                                        truth_pdp,
                                        keep,
                                    )
                                    for source_blend in source_blends:
                                        mixed_with_source = (
                                            (1.0 - source_blend)
                                            * prediction_source
                                            + source_blend
                                            * mixed_prediction
                                        )
                                        record(
                                            candidate_name
                                            + f"_mix{dual_blend:g}"
                                            + f"_src{source_blend:g}",
                                            seed,
                                            mixed_with_source,
                                            truth,
                                            truth_pas,
                                            truth_pdp,
                                            keep,
                                        )
                    del prediction
            del dual_reference
        print(f"[neighbor-project] split={seed} done", flush=True)
        del (
            truth,
            sources,
            baseline,
            target_pas,
            target_pdp,
            truth_pas,
            truth_pdp,
            transported_pas,
        )
        if args.hvp_expert_dir:
            del baseline_hvp, expert_hvp, target_hvp_by_blend
        torch.cuda.empty_cache()

    ranked = []
    for name, rows in sufficient.items():
        tune_cross = sum(rows[seed]["cross"] for seed in tune_seeds)
        tune_prediction_energy = sum(
            rows[seed]["prediction_energy"] for seed in tune_seeds
        )
        residual_phase = float(np.angle(tune_cross))
        phase_factor = np.exp(-1j * residual_phase)
        scale = max(
            float(np.real(tune_cross * phase_factor))
            / max(tune_prediction_energy, 1e-30),
            0.0,
        )
        scores = {}
        for seed in seeds:
            row = rows[seed]
            cross_real = float(
                np.real(row["cross"] * phase_factor)
            )
            nmse = (
                row["truth_energy"]
                + scale**2 * row["prediction_energy"]
                - 2.0 * scale * cross_real
            ) / max(row["truth_energy"], 1e-30)
            combined = (
                spec.metric_weights[0] * row["PAS"]
                + spec.metric_weights[1] * row["PDP"]
                + spec.metric_weights[2] / (1.0 + nmse)
            )
            scores[str(seed)] = {
                "PAS": row["PAS"],
                "PDP": row["PDP"],
                "NMSE": float(nmse),
                "C": float(combined),
            }
            if args.report_pas_layouts:
                for layout in ("hvp", "pvh"):
                    layout_pas = row[f"PAS_{layout}"]
                    layout_combined = (
                        spec.metric_weights[0] * layout_pas
                        + spec.metric_weights[1] * row["PDP"]
                        + spec.metric_weights[2] / (1.0 + nmse)
                    )
                    scores[str(seed)][f"PAS_{layout}"] = layout_pas
                    scores[str(seed)][f"C_{layout}"] = float(
                        layout_combined
                    )
        tune = [scores[str(seed)]["C"] for seed in tune_seeds]
        ranked.append(
            {
                "name": name,
                "residual_global_phase": residual_phase,
                "scale": float(scale),
                "tune_median": float(np.median(tune)),
                "tune_mean": float(np.mean(tune)),
                "tune_worst": float(np.min(tune)),
                "audit": scores[str(args.audit_seed)]["C"],
                "scores": scores,
            }
        )
    ranked.sort(
        key=lambda row: (
            row["tune_median"],
            row["tune_mean"],
            row["tune_worst"],
        ),
        reverse=True,
    )
    payload = {
        "selection_policy": (
            "source/order/iterations/phase/scale selected on four tune "
            "splits; strict audit excludes every tune index"
            if args.strict_audit
            else "splits; audit is reported but not used for selection"
        ),
        "phase": {"k0": k0, "k1": k1},
        "pas_blend": args.pas_blend,
        "pdp_blend": args.pdp_blend,
        "pas_transport": {
            "blend": args.pas_transport_blend,
            "k": args.pas_transport_k,
            "distance_power": args.pas_transport_power,
            "gate_dir": args.pas_transport_gate_dir,
        },
        "phase_seed_epsilon": args.phase_seed_epsilon,
        "steering_result": args.steering_result,
        "source_specs": source_specs,
        "complex_gate": {
            "directory": args.complex_gate_dir,
            "source": args.complex_gate_source,
            "alpha": args.complex_gate_alpha,
            "secondary_directory": (
                args.complex_gate_secondary_dir
            ),
            "secondary_alpha": (
                args.complex_gate_secondary_alpha
            ),
        },
        "covariance_kriging": {
            "directory": args.covariance_kriging_dir,
            "source": args.covariance_kriging_source,
            "diagonal_loading": args.covariance_kriging_loading,
            "blend_grid": args.covariance_kriging_blend_grid,
            "hybrid_grid": args.covariance_kriging_hybrid_grid,
        },
        "angle_delay_consensus": {
            "source": args.angle_delay_source,
            "layout": "phv",
            "group": args.angle_delay_group,
            "gamma": args.angle_delay_gamma,
            "blend": args.angle_delay_blend,
            "ratio_clip": [
                args.angle_delay_ratio_min,
                args.angle_delay_ratio_max,
            ],
        },
        "tune_seeds": tune_seeds,
        "audit_seed": args.audit_seed,
        "strict_audit": args.strict_audit,
        "reported_pas_layouts": args.report_pas_layouts,
        "score_pas_layout": args.score_pas_layout,
        "hvp_expert_dir": args.hvp_expert_dir,
        "hvp_blend_grid": hvp_blends if args.hvp_expert_dir else [],
        "dual_orders": dual_orders if args.hvp_expert_dir else [],
        "dual_blend_grid": (
            dual_blends if args.hvp_expert_dir else []
        ),
        "source_blend_grid": source_blends,
        "strict_audit_count": (
            200 - len(
                set(
                    reproduce_val_indices(
                        2000, 0.1, args.audit_seed
                    )
                )
                & tune_union
            )
            if args.strict_audit
            else 200
        ),
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
        json.dumps(ranked[:15], ensure_ascii=False),
        flush=True,
    )
    print(
        f"GEOMETRIC_PHASE_NEIGHBORS_PROJECTION_DONE out={out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
