#!/usr/bin/env python3
"""Build the geometric-phase and moment-spectrum submission candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

from _bootstrap import ROOT  # noqa: F401
from complex_neighbor_gate import (
    FEATURE_NAMES as COMPLEX_GATE_FEATURE_NAMES,
)
from complex_neighbor_gate import (
    ComplexNeighborGate,
    baseline_coefficients,
    build_complex_features,
)
from neighbor_source_weights import (
    anisotropic_weights,
    equalize_amplitude,
    moment_correct_weights,
)
from pas_transport_gate import (
    FEATURE_NAMES as PAS_TRANSPORT_FEATURE_NAMES,
    PasTransportGate,
    build_gate_features,
)
from probe_array_steering_phase import array_coordinates
from sweep_moment_attention import SliceAttention, moment_project, stable_unit
from validate_moment_projection import enforce_pas, enforce_pdp
from validate_moment_projection import enforce_pas_layout
from wireless_twin.data.map_loader import load_point_cloud
from wireless_twin.data.setup_config import load_setup
from wireless_twin.models.raytrace2 import build_heightmap
from wireless_twin.signal import (
    pas_spectrum,
    pas_spectrum_phv,
    pdp_spectrum,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", default="Round1_Map(2)")
    parser.add_argument("--cache-dir", default="cache/teammate_knn_hvp")
    parser.add_argument(
        "--config", default="configs/gs37_geophase_moment.json"
    )
    parser.add_argument(
        "--baseline",
        default=(
            "best_submit/BLEND_GS34_PHV_FULLBANK_REF10_TF05/"
            "Round1_Test_Channel.npy"
        ),
    )
    parser.add_argument("--phase-result")
    parser.add_argument(
        "--projection-result",
    )
    parser.add_argument("--outdir")
    parser.add_argument("--pas-correction", type=float)
    parser.add_argument("--pas-radial", type=float)
    parser.add_argument("--pas-tangent", type=float)
    parser.add_argument("--pas-blend", type=float)
    parser.add_argument("--pdp-correction", type=float)
    parser.add_argument("--pdp-radial", type=float)
    parser.add_argument("--pdp-tangent", type=float)
    parser.add_argument("--pdp-blend", type=float)
    parser.add_argument("--phase-seed-epsilon", type=float)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    candidate_name = str(
        config.get("name", "BLEND_GS37_GEOPHASE_MOMENT")
    )
    seeds = [
        *[int(seed) for seed in config["tune_seeds"]],
        int(config["audit_seed"]),
    ]
    defaults = {
        "pas_correction": config["pas"]["correction"],
        "pas_radial": config["pas"]["radial_multiplier"],
        "pas_tangent": config["pas"]["tangent_multiplier"],
        "pas_blend": config["pas"]["blend"],
        "pdp_correction": config["pdp"]["correction"],
        "pdp_radial": config["pdp"]["radial_multiplier"],
        "pdp_tangent": config["pdp"]["tangent_multiplier"],
        "pdp_blend": config["pdp"]["blend"],
        "phase_seed_epsilon": config["projection"][
            "phase_seed_epsilon"
        ],
        "iterations": config["projection"]["iterations"],
    }
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)

    device = torch.device(args.device)
    datadir = Path(args.datadir)
    spec = load_setup(datadir / "Round1_Setup.json")
    train_pos = np.load(
        datadir / "Round1_Train_Pos.npy"
    ).astype(np.float64)
    test_pos = np.load(
        datadir / "Round1_Test_Pos.npy"
    ).astype(np.float64)
    train_channels = np.load(
        datadir / "Round1_Train_Channel.npy", mmap_mode="r"
    )

    baseline_path = Path(args.baseline)
    baseline_np = np.load(baseline_path, mmap_mode="r")
    baseline_rms = float(
        np.sqrt(np.mean(np.square(np.abs(baseline_np), dtype=np.float64)))
    )
    baseline = torch.as_tensor(
        np.array(baseline_np, copy=True),
        dtype=torch.complex64,
        device=device,
    )
    del baseline_np

    phase_path = Path(args.phase_result) if args.phase_result else None
    phase_selected = config["phase"]
    if phase_path is not None:
        phase_payload = json.loads(
            phase_path.read_text(encoding="utf-8")
        )
        phase_selected = phase_payload["selected"]
    k0 = float(phase_selected["k0_rad_per_meter"])
    k1 = float(
        phase_selected["k1_rad_per_meter_per_subcarrier"]
    )
    global_phase = float(phase_selected["global_phase"])

    projection_path = (
        Path(args.projection_result) if args.projection_result else None
    )
    selected_projection = {
        "tune_median": None,
        "audit": None,
        **config["projection"],
    }
    selection_policy = (
        f"parameters frozen in {config_path}"
    )
    if projection_path is not None:
        projection_payload = json.loads(
            projection_path.read_text(encoding="utf-8")
        )
        selected_projection = projection_payload["ranked"][0]
        selection_policy = projection_payload["selection_policy"]
    projection_order = str(selected_projection["order"])
    if projection_order not in {
        "pas_pdp",
        "pdp_pas",
        "hvp_phv_pdp",
        "phv_hvp_pdp",
        "hvp_pdp_phv",
        "phv_pdp_hvp",
    }:
        raise ValueError(
            "unsupported selected validation projection: "
            f"{projection_order}"
        )
    if int(selected_projection["iterations"]) != args.iterations:
        raise ValueError(
            "iteration mismatch between validation and builder: "
            f"{selected_projection['iterations']} != {args.iterations}"
        )
    output_scale = float(selected_projection["scale"])
    residual_global_phase = float(
        selected_projection.get(
            "residual_global_phase",
            config["projection"].get("residual_global_phase", 0.0),
        )
    )

    pool_idx = np.arange(len(train_pos), dtype=np.int64)
    source_config = config.get(
        "source",
        {
            "kind": "nearest",
            "k": 1,
            "radial_ratio": 1.0,
            "distance_power": 0.0,
            "amplitude": "raw",
        },
    )
    source_k = int(source_config.get("k", 1))
    complex_gate_config = config.get("complex_gate")
    complex_gate_components = []
    if complex_gate_config is not None:
        if source_k <= 1:
            raise ValueError(
                "complex-neighbor gate requires a multi-neighbor source"
            )
        if source_config.get("moment") is not None:
            raise ValueError(
                "complex-neighbor gate must be retrained for a "
                "moment-corrected source"
            )
        component_specs = [
            (
                "primary",
                complex_gate_config["checkpoint"],
                complex_gate_config["alpha"],
            )
        ]
        if complex_gate_config.get("secondary_checkpoint"):
            component_specs.append(
                (
                    "secondary",
                    complex_gate_config["secondary_checkpoint"],
                    complex_gate_config["secondary_alpha"],
                )
            )
        for role, checkpoint_name, alpha_value in component_specs:
            alpha = float(alpha_value)
            if not 0.0 <= alpha <= 1.0:
                raise ValueError(
                    f"{role} complex-neighbor gate alpha must "
                    "be in [0, 1]"
                )
            if alpha == 0.0:
                continue
            checkpoint_path = Path(checkpoint_name)
            payload = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=False,
            )
            checkpoint_k = int(
                payload.get("metadata", {}).get("k", source_k)
            )
            if checkpoint_k != source_k:
                raise ValueError(
                    f"{role} complex-neighbor gate K mismatch: "
                    f"{checkpoint_k} != {source_k}"
                )
            feature_names = tuple(payload["feature_names"])
            unknown_features = set(feature_names).difference(
                COMPLEX_GATE_FEATURE_NAMES
            )
            if unknown_features:
                raise ValueError(
                    f"unknown {role} complex-neighbor gate "
                    f"features: {sorted(unknown_features)}"
                )
            model = ComplexNeighborGate(
                len(feature_names),
                payload.get("architecture", "mlp16"),
            ).to(device)
            model.load_state_dict(payload["model_state"])
            model.eval()
            complex_gate_components.append(
                {
                    "role": role,
                    "path": checkpoint_path,
                    "alpha": alpha,
                    "indices": [
                        COMPLEX_GATE_FEATURE_NAMES.index(name)
                        for name in feature_names
                    ],
                    "model": model,
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
            )
    pas_transport = config["pas"].get("transport")
    transport_k = (
        int(pas_transport["k"]) if pas_transport is not None else 0
    )
    if transport_k > source_k:
        raise ValueError(
            "PAS transport requires at least as many source neighbors: "
            f"{transport_k} > {source_k}"
        )
    source_distance, local = cKDTree(train_pos[:, :2]).query(
        test_pos[:, :2], k=source_k
    )
    source_distance = np.asarray(
        source_distance, dtype=np.float64
    ).reshape(
        len(test_pos), source_k
    )
    neighbor_idx = pool_idx[np.asarray(local)].reshape(
        len(test_pos), source_k
    )
    nearest_idx = neighbor_idx[:, 0]
    bs = np.asarray(spec.bs_position, dtype=np.float64)
    train_radius = np.linalg.norm(train_pos - bs[None], axis=1)
    test_radius = np.linalg.norm(test_pos - bs[None], axis=1)
    train_unit = (train_pos - bs[None]) / np.maximum(
        train_radius[:, None], 1e-12
    )
    test_unit = (test_pos - bs[None]) / np.maximum(
        test_radius[:, None], 1e-12
    )
    delta_radius = (
        test_radius[:, None] - train_radius[neighbor_idx]
    )
    delta_unit = test_unit[:, None] - train_unit[neighbor_idx]
    delta_xy = (
        train_pos[neighbor_idx, :2] - test_pos[:, None, :2]
    )
    radial = test_pos[:, :2] - bs[None, :2]
    radial /= np.maximum(
        np.linalg.norm(radial, axis=1, keepdims=True), 1e-12
    )
    tangent = np.column_stack([-radial[:, 1], radial[:, 0]])
    radial_delta = np.einsum("bkd,bd->bk", delta_xy, radial)
    tangent_delta = np.einsum("bkd,bd->bk", delta_xy, tangent)
    subcarrier = np.arange(spec.s, dtype=np.float64)
    subcarrier -= subcarrier.mean()
    steering_config = config.get("steering")
    steering_horizontal = None
    steering_vertical = None
    steering_axis = None
    if steering_config is not None:
        steering_horizontal, steering_vertical = array_coordinates(
            spec, str(steering_config["layout"])
        )
        steering_axis = np.asarray(
            [
                np.cos(float(steering_config["theta"])),
                np.sin(float(steering_config["theta"])),
            ],
            dtype=np.float64,
        )
    source = torch.zeros_like(baseline)
    transported_pas = (
        torch.zeros(
            (
                len(test_pos),
                spec.n,
                spec.s,
                spec.mh * spec.mv,
            ),
            dtype=torch.float32,
            device=device,
        )
        if pas_transport is not None
        else None
    )
    inline_global_phase = (
        source_k == 1
        and source_config.get("kind", "nearest") == "nearest"
    )
    for start in range(0, len(test_pos), args.batch_size):
        stop = min(start + args.batch_size, len(test_pos))
        batch = stop - start
        neighbors = torch.as_tensor(
            np.array(
                train_channels[neighbor_idx[start:stop].reshape(-1)],
                copy=True,
            ).reshape(
                batch,
                source_k,
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
        if inline_global_phase:
            phase = phase + global_phase
        neighbors *= torch.as_tensor(
            np.exp(1j * phase),
            dtype=torch.complex64,
            device=device,
        )[:, :, None, None, :]
        if steering_config is not None:
            delta_horizontal = np.einsum(
                "bkd,d->bk",
                delta_unit[start:stop, :, :2],
                steering_axis,
            )
            steering_phase = (
                float(steering_config["horizontal_coefficient"])
                * delta_horizontal[:, :, None]
                * steering_horizontal[None, None, :]
                + float(steering_config["vertical_coefficient"])
                * delta_unit[start:stop, :, 2:3]
                * steering_vertical[None, None, :]
            )
            neighbors *= torch.as_tensor(
                np.exp(1j * steering_phase),
                dtype=torch.complex64,
                device=device,
            )[:, :, :, None, None]
        if transported_pas is not None:
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
                    source_distance[start:stop, :transport_k],
                    0.05,
                )
                ** float(pas_transport["distance_power"])
            )
            transport_weight /= np.maximum(
                transport_weight.sum(axis=1, keepdims=True),
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
        if source_k == 1:
            weight = np.ones((batch, 1), dtype=np.float64)
        else:
            radial_ratio = float(
                source_config["radial_ratio"]
            )
            distance_power = float(
                source_config["distance_power"]
            )
            weight, effective = anisotropic_weights(
                radial_delta[start:stop],
                tangent_delta[start:stop],
                radial_ratio,
                distance_power,
            )
            moment_config = source_config.get("moment")
            if moment_config is not None:
                weight = moment_correct_weights(
                    weight,
                    radial_delta[start:stop],
                    tangent_delta[start:stop],
                    ridge=float(moment_config["ridge"]),
                    strength=float(moment_config["strength"]),
                )
        if complex_gate_components:
            source_flat = neighbors.reshape(batch, source_k, -1)
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
            gate_features, gate_amplitude = build_complex_features(
                gram,
                base_weight_t,
                torch.as_tensor(
                    source_distance[start:stop],
                    dtype=torch.float32,
                    device=device,
                ),
                torch.as_tensor(
                    effective,
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
                    test_pos[start:stop],
                    dtype=torch.float32,
                    device=device,
                ),
                torch.as_tensor(
                    bs,
                    dtype=torch.float32,
                    device=device,
                ),
            )
            baseline_weight = baseline_coefficients(
                base_weight_t, gate_amplitude
            )
            weight_t = baseline_weight
            with torch.inference_mode():
                for component in complex_gate_components:
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
                            + component["alpha"] * learned_weight
                        )
                    else:
                        weight_t += component["alpha"] * (
                            learned_weight - baseline_weight
                        )
        else:
            if source_config.get("amplitude", "raw") == "equalized":
                energy = (
                    neighbors.abs()
                    .square()
                    .sum(dim=(2, 3, 4))
                    .cpu()
                    .numpy()
                )
                weight = equalize_amplitude(
                    weight,
                    np.sqrt(np.maximum(energy, 1e-30)),
                )
            weight_t = torch.as_tensor(
                weight, dtype=torch.complex64, device=device
            )
        source[start:stop] = torch.einsum(
            "bk,bkmns->bmns", weight_t, neighbors
        )
        del neighbors
    if not inline_global_phase:
        source *= torch.exp(
            torch.tensor(
                1j * global_phase,
                dtype=torch.complex64,
                device=device,
            )
        )
    source_rms_before_seed = source.abs().square().mean().sqrt()
    source = source + (
        args.phase_seed_epsilon
        * baseline
        * source_rms_before_seed
        / baseline.abs().square().mean().sqrt().clamp_min(1e-30)
    )
    source_rms = float(source.abs().square().mean().sqrt())

    points = load_point_cloud(datadir / "Round1_Map.ply")
    heightmap, x0, y0, resolution = build_heightmap(points)
    gx = np.clip(
        np.floor((train_pos[:, 0] - x0) / resolution).astype(np.int64),
        0,
        heightmap.shape[0] - 1,
    )
    gy = np.clip(
        np.floor((train_pos[:, 1] - y0) / resolution).astype(np.int64),
        0,
        heightmap.shape[1] - 1,
    )
    indoor = (heightmap[gx, gy] > 2.0).astype(np.float32)

    checkpoint_templates = {
        "pas": (
            "checkpoints/clean_panel_phv_geom_k16/s{seed}/"
            "nbrattn_clean_k16_pas_k16s0.pt"
        ),
        "pdp": (
            "checkpoints/clean_panel/s{seed}/"
            "nbrattn_clean_k32_pdp_k32s0.pt"
        ),
        "hvp": (
            "checkpoints/clean_panel/s{seed}/"
            "nbrattn_clean_k32_pas_k32s0.pt"
        ),
    }
    cache_names = {
        "pas": "train_pas_phv.npy",
        "pdp": "train_pdp.npy",
        "hvp": "train_pas_hvp.npy",
    }
    manifest = {
        "config": f"{config_path}:{sha256(config_path)}",
        "baseline": f"{baseline_path}:{sha256(baseline_path)}",
        "arms": {},
    }
    if phase_path is not None:
        manifest["phase_result"] = (
            f"{phase_path}:{sha256(phase_path)}"
        )
    if projection_path is not None:
        manifest["projection_result"] = (
            f"{projection_path}:{sha256(projection_path)}"
        )
    if complex_gate_components:
        manifest["complex_neighbor_gates"] = {
            component["role"]: (
                f"{component['path']}:{sha256(component['path'])}"
            )
            for component in complex_gate_components
        }

    def build_features(
        spectra: torch.Tensor,
        query_pos: np.ndarray,
        neighbors: np.ndarray,
        distance: np.ndarray,
        feature_set: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, k = neighbors.shape
        candidate = spectra[
            torch.as_tensor(neighbors, dtype=torch.long, device=device)
        ]
        values = candidate.reshape(batch, k, -1, candidate.shape[-1])
        mean = values.mean(dim=1, keepdim=True)
        agreement = F.cosine_similarity(values, mean, dim=-1)
        distance_t = torch.as_tensor(
            distance, dtype=torch.float32, device=device
        )
        expanded_distance = distance_t[:, :, None].expand(
            -1, -1, values.shape[2]
        )
        neighbor_indoor = torch.as_tensor(
            indoor[neighbors], dtype=torch.float32, device=device
        )[:, :, None].expand(-1, -1, values.shape[2])
        query_gx = np.clip(
            np.floor((query_pos[:, 0] - x0) / resolution).astype(np.int64),
            0,
            heightmap.shape[0] - 1,
        )
        query_gy = np.clip(
            np.floor((query_pos[:, 1] - y0) / resolution).astype(np.int64),
            0,
            heightmap.shape[1] - 1,
        )
        query_indoor_np = (
            heightmap[query_gx, query_gy] > 2.0
        ).astype(np.float32)
        query_indoor = torch.as_tensor(
            query_indoor_np, dtype=torch.float32, device=device
        )[:, None, None].expand(-1, k, values.shape[2])
        columns = [
            expanded_distance / 3.0,
            (neighbor_indoor == query_indoor).float(),
            agreement,
            agreement.square(),
            torch.ones_like(expanded_distance),
            (expanded_distance < 2.5).float(),
        ]
        if feature_set == "geometry":
            target_xy = torch.as_tensor(
                query_pos[:, :2],
                dtype=torch.float32,
                device=device,
            )
            neighbor_xy = torch.as_tensor(
                train_pos[neighbors, :2],
                dtype=torch.float32,
                device=device,
            )
            delta = neighbor_xy - target_xy[:, None]
            bs_xy = torch.as_tensor(
                spec.bs_position[:2],
                dtype=torch.float32,
                device=device,
            )
            radial = target_xy - bs_xy
            radius = radial.norm(dim=-1).clamp_min(1e-3)
            radial_unit = radial / radius[:, None]
            tangent_unit = torch.stack(
                [-radial_unit[:, 1], radial_unit[:, 0]], dim=-1
            )
            radial_delta = (delta * radial_unit[:, None]).sum(dim=-1)
            tangent_delta = (delta * tangent_unit[:, None]).sum(dim=-1)

            def expand(value: torch.Tensor) -> torch.Tensor:
                return value[:, :, None].expand(
                    -1, -1, values.shape[2]
                )

            angle = torch.atan2(radial[:, 1], radial[:, 0])
            columns.extend(
                [
                    expand(delta[..., 0] / 5.0),
                    expand(delta[..., 1] / 5.0),
                    expand(radial_delta / 5.0),
                    expand(tangent_delta / 5.0),
                    (radius / 200.0)[:, None, None].expand(
                        -1, k, values.shape[2]
                    ),
                    angle.sin()[:, None, None].expand(
                        -1, k, values.shape[2]
                    ),
                    angle.cos()[:, None, None].expand(
                        -1, k, values.shape[2]
                    ),
                    neighbor_indoor,
                    query_indoor,
                ]
            )
        elif feature_set != "basic":
            raise ValueError(f"unsupported feature_set={feature_set}")
        features = torch.stack(columns, dim=-1)
        log_distance = torch.log(
            distance_t.clamp_min(0.3)
        )[:, :, None]
        return features, values, log_distance

    def spatial_delta(
        query_pos: np.ndarray,
        neighbors: np.ndarray,
        distance: np.ndarray,
    ) -> torch.Tensor:
        delta = torch.as_tensor(
            train_pos[neighbors, :2] - query_pos[:, None, :2],
            dtype=torch.float32,
            device=device,
        )
        scale = torch.as_tensor(
            np.median(distance, axis=1),
            dtype=torch.float32,
            device=device,
        ).clamp_min(0.3)
        delta /= scale[:, None, None]
        target_xy = torch.as_tensor(
            query_pos[:, :2],
            dtype=torch.float32,
            device=device,
        )
        bs_xy = torch.as_tensor(
            spec.bs_position[:2],
            dtype=torch.float32,
            device=device,
        )
        radial = target_xy - bs_xy
        radial /= radial.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        tangent = torch.stack(
            [-radial[:, 1], radial[:, 0]], dim=-1
        )
        return torch.stack(
            [
                (delta * radial[:, None]).sum(dim=-1),
                (delta * tangent[:, None]).sum(dim=-1),
            ],
            dim=-1,
        )

    hvp_target_config = config["pas"].get("hvp")
    post_bilateral_config = config["pdp"].get("post_bilateral")
    post_bilateral_pdp = None
    experts = {}
    expert_domains = ["pas", "pdp"]
    if hvp_target_config is not None:
        expert_domains.append("hvp")
    for domain in expert_domains:
        print(f"[GS37] loading {domain} spectra", flush=True)
        cached = np.load(
            Path(args.cache_dir) / cache_names[domain], mmap_mode="r"
        )
        spectra = stable_unit(
            torch.as_tensor(
                np.array(cached, copy=True),
                dtype=torch.float32,
                device=device,
            )
        )
        del cached
        if domain == "pas":
            domain_config = config["pas"]
            k = 16
            correction = args.pas_correction
            radial_multiplier = args.pas_radial
            tangent_multiplier = args.pas_tangent
        elif domain == "pdp":
            domain_config = config["pdp"]
            k = 32
            correction = args.pdp_correction
            radial_multiplier = args.pdp_radial
            tangent_multiplier = args.pdp_tangent
        else:
            domain_config = hvp_target_config
            k = int(domain_config.get("k", 32))
            correction = float(domain_config["correction"])
            radial_multiplier = float(
                domain_config["radial_multiplier"]
            )
            tangent_multiplier = float(
                domain_config["tangent_multiplier"]
            )
        distance, local = cKDTree(train_pos[:, :2]).query(
            test_pos[:, :2], k=k
        )
        distance = np.asarray(distance, dtype=np.float32)
        neighbors = pool_idx[np.asarray(local)]
        if domain == "pdp" and post_bilateral_config is not None:
            bilateral_k = int(post_bilateral_config.get("k", 32))
            if bilateral_k > k:
                raise ValueError(
                    "post-bilateral PDP K exceeds PDP expert K: "
                    f"{bilateral_k} > {k}"
                )
            distance_power = float(
                post_bilateral_config["distance_power"]
            )
            anchor_strength = float(
                post_bilateral_config["anchor_strength"]
            )
            bilateral_chunks = []
            with torch.inference_mode():
                for start in range(0, len(test_pos), args.batch_size):
                    stop = min(start + args.batch_size, len(test_pos))
                    neighbor_t = torch.as_tensor(
                        neighbors[start:stop, :bilateral_k],
                        dtype=torch.long,
                        device=device,
                    )
                    values = spectra[neighbor_t]
                    nearest_agreement = (
                        values * values[:, :1]
                    ).sum(dim=-1).mean(dim=(2, 3))
                    distance_t = torch.as_tensor(
                        distance[start:stop, :bilateral_k],
                        dtype=torch.float32,
                        device=device,
                    ).clamp_min(0.3)
                    weight = torch.softmax(
                        -distance_power * distance_t.log()
                        + anchor_strength * nearest_agreement,
                        dim=1,
                    )
                    bilateral_chunks.append(
                        stable_unit(
                            torch.einsum(
                                "bk,bkmns->bmns", weight, values
                            )
                        )
                    )
            post_bilateral_pdp = torch.cat(bilateral_chunks)
            manifest["post_bilateral_pdp"] = {
                **post_bilateral_config,
                "selection": "source-only nearest-global agreement",
            }
        expert_sum = torch.zeros(
            (len(test_pos),) + tuple(spectra.shape[1:]),
            dtype=torch.float32,
            device=device,
        )
        axis_multiplier = torch.tensor(
            [
                radial_multiplier,
                tangent_multiplier,
            ],
            dtype=torch.float32,
            device=device,
        )
        for seed in seeds:
            checkpoint_template = str(
                domain_config.get(
                    "checkpoint_template",
                    checkpoint_templates[domain],
                )
            )
            checkpoint = Path(checkpoint_template.format(seed=seed))
            payload = torch.load(
                checkpoint, map_location=device, weights_only=False
            )
            meta = payload["meta"]
            feature_set = str(meta.get("feature_set", "basic"))
            feature_dim = int(
                meta.get(
                    "feature_dim",
                    15 if feature_set == "geometry" else 6,
                )
            )
            model = SliceAttention(feature_dim).to(device)
            model.load_state_dict(payload["model_state"])
            model.eval()
            chunks = []
            with torch.inference_mode():
                for start in range(0, len(test_pos), args.batch_size):
                    stop = min(start + args.batch_size, len(test_pos))
                    features, values, log_distance = build_features(
                        spectra,
                        test_pos[start:stop],
                        neighbors[start:stop],
                        distance[start:stop],
                        feature_set,
                    )
                    logits = model.logits(features, log_distance)
                    weight, _ = moment_project(
                        logits,
                        spatial_delta(
                            test_pos[start:stop],
                            neighbors[start:stop],
                            distance[start:stop],
                        ),
                        correction,
                        axis_multiplier,
                        12,
                        0.03,
                    )
                    chunks.append(
                        stable_unit(
                            torch.einsum(
                                "bkq,bkql->bql", weight, values
                            )
                        )
                    )
            expert_sum += torch.cat(chunks).reshape(expert_sum.shape)
            manifest["arms"][f"{domain}_{seed}"] = (
                f"{checkpoint}:{sha256(checkpoint)}"
            )
            del model, chunks
            torch.cuda.empty_cache()
            print(f"[GS37] {domain} seed={seed} done", flush=True)
        experts[domain] = stable_unit(expert_sum / len(seeds))
        del spectra, expert_sum
        torch.cuda.empty_cache()

    transport_gate_stats = None
    if transported_pas is not None:
        gate_checkpoint = pas_transport.get("gate_checkpoint")
        if gate_checkpoint:
            gate_path = Path(gate_checkpoint)
            payload = torch.load(
                gate_path, map_location=device, weights_only=False
            )
            if payload["feature_names"] != list(
                PAS_TRANSPORT_FEATURE_NAMES
            ):
                raise ValueError(
                    "PAS transport gate feature definition mismatch"
                )
            gate = PasTransportGate(
                len(PAS_TRANSPORT_FEATURE_NAMES),
                str(payload["architecture"]),
            ).to(device)
            gate.load_state_dict(payload["model_state"])
            gate.eval()
            feature_mean = torch.as_tensor(
                payload["feature_mean"],
                dtype=torch.float32,
                device=device,
            )
            feature_std = torch.as_tensor(
                payload["feature_std"],
                dtype=torch.float32,
                device=device,
            )
            beta_chunks = []
            with torch.inference_mode():
                for start in range(0, len(test_pos), args.batch_size):
                    stop = min(
                        start + args.batch_size, len(test_pos)
                    )
                    gate_features = build_gate_features(
                        experts["pas"][start:stop],
                        transported_pas[start:stop],
                        torch.as_tensor(
                            test_pos[start:stop],
                            dtype=torch.float32,
                            device=device,
                        ),
                        torch.as_tensor(
                            source_distance[
                                start:stop, :transport_k
                            ],
                            dtype=torch.float32,
                            device=device,
                        ),
                        torch.as_tensor(
                            bs, dtype=torch.float32, device=device
                        ),
                        spec.mh,
                        spec.mv,
                    )
                    normalized = (
                        gate_features - feature_mean
                    ) / feature_std
                    beta_chunks.append(
                        gate(normalized.clamp(-8.0, 8.0))
                    )
            transport_beta = torch.cat(beta_chunks)
            experts["pas"] = stable_unit(
                (1.0 - transport_beta[..., None]) * experts["pas"]
                + transport_beta[..., None] * transported_pas
            )
            transport_gate_stats = {
                "checkpoint": f"{gate_path}:{sha256(gate_path)}",
                "architecture": payload["architecture"],
                "beta_mean": float(transport_beta.mean()),
                "beta_std": float(
                    transport_beta.std(correction=0)
                ),
                "beta_min": float(transport_beta.min()),
                "beta_max": float(transport_beta.max()),
            }
            manifest["pas_transport_gate"] = (
                transport_gate_stats["checkpoint"]
            )
            del gate, gate_features, normalized, beta_chunks
        else:
            transport_blend = float(pas_transport["blend"])
            experts["pas"] = stable_unit(
                (1.0 - transport_blend) * experts["pas"]
                + transport_blend * transported_pas
            )

    baseline_pas = stable_unit(pas_spectrum_phv(baseline, spec))
    baseline_pdp = stable_unit(pdp_spectrum(baseline, spec))
    target_pas = stable_unit(
        (1.0 - args.pas_blend) * baseline_pas
        + args.pas_blend * experts["pas"]
    )
    target_pdp = stable_unit(
        (1.0 - args.pdp_blend) * baseline_pdp
        + args.pdp_blend * experts["pdp"]
    )
    baseline_hvp = None
    target_hvp = None
    if hvp_target_config is not None:
        baseline_hvp = stable_unit(pas_spectrum(baseline, spec))
        hvp_blend = float(hvp_target_config["blend"])
        target_hvp = stable_unit(
            (1.0 - hvp_blend) * baseline_hvp
            + hvp_blend * experts["hvp"]
        )
    projection_steps = {
        "pas_pdp": ("phv", "pdp"),
        "pdp_pas": ("pdp", "phv"),
        "hvp_phv_pdp": ("hvp", "phv", "pdp"),
        "phv_hvp_pdp": ("phv", "hvp", "pdp"),
        "hvp_pdp_phv": ("hvp", "pdp", "phv"),
        "phv_pdp_hvp": ("phv", "pdp", "hvp"),
    }[projection_order]

    def project_channel(
        initial: torch.Tensor, steps: tuple[str, ...]
    ) -> torch.Tensor:
        current = initial.clone()
        for _ in range(args.iterations):
            for step in steps:
                if step == "phv":
                    current = enforce_pas(
                        current, target_pas, spec
                    )
                elif step == "pdp":
                    current = enforce_pdp(current, target_pdp)
                else:
                    if target_hvp is None:
                        raise ValueError(
                            "HVP projection requested without "
                            "pas.hvp config"
                        )
                    current = enforce_pas_layout(
                        current, target_hvp, spec, "hvp"
                    )
        return current

    dual_mix = config["projection"].get("dual_mix")
    if dual_mix is not None:
        reference_prediction = project_channel(
            source, ("phv", "pdp")
        )
        dual_prediction = project_channel(source, projection_steps)
        dual_mix = float(dual_mix)
        prediction = (
            (1.0 - dual_mix) * reference_prediction
            + dual_mix * dual_prediction
        )
        del reference_prediction, dual_prediction
    else:
        prediction = project_channel(source, projection_steps)
    source_blend = config["projection"].get("source_blend")
    if source_blend is not None:
        source_blend = float(source_blend)
        if not 0.0 <= source_blend <= 1.0:
            raise ValueError("source_blend must be in [0, 1]")
        prediction = (
            (1.0 - source_blend) * source
            + source_blend * prediction
        )
    if post_bilateral_config is not None:
        if post_bilateral_pdp is None:
            raise RuntimeError("post-bilateral PDP target was not built")
        target_beta = float(post_bilateral_config["target_beta"])
        channel_mix = float(
            post_bilateral_config.get("channel_mix", 1.0)
        )
        if not 0.0 <= target_beta <= 1.0:
            raise ValueError(
                "post-bilateral PDP target_beta must be in [0, 1]"
            )
        if not 0.0 <= channel_mix <= 1.0:
            raise ValueError(
                "post-bilateral PDP channel_mix must be in [0, 1]"
            )
        current_pdp = stable_unit(pdp_spectrum(prediction, spec))
        bilateral_target = stable_unit(
            (1.0 - target_beta) * current_pdp
            + target_beta * post_bilateral_pdp
        )
        bilateral_prediction = enforce_pdp(
            prediction, bilateral_target
        )
        prediction = (
            (1.0 - channel_mix) * prediction
            + channel_mix * bilateral_prediction
        )
        del current_pdp, bilateral_target, bilateral_prediction
    prediction *= torch.tensor(
        output_scale * np.exp(1j * residual_global_phase),
        dtype=torch.complex64,
        device=device,
    )

    output = prediction.detach().cpu().numpy().astype(np.complex64)
    if output.shape != baseline.shape:
        raise ValueError(f"unexpected output shape {output.shape}")
    if not np.isfinite(output.real).all() or not np.isfinite(
        output.imag
    ).all():
        raise ValueError("output contains non-finite values")

    outdir = (
        Path(args.outdir)
        if args.outdir
        else Path("best_submit") / candidate_name
    )
    outdir.mkdir(parents=True, exist_ok=True)
    output_path = outdir / "Round1_Test_Channel.npy"
    np.save(output_path, output)
    output_rms = float(
        np.sqrt(np.mean(np.square(np.abs(output), dtype=np.float64)))
    )
    final_pas = stable_unit(pas_spectrum_phv(prediction, spec))
    final_pdp = stable_unit(pdp_spectrum(prediction, spec))
    final_hvp = (
        stable_unit(pas_spectrum(prediction, spec))
        if target_hvp is not None
        else None
    )
    pas_cosine = float((final_pas * baseline_pas).sum(dim=-1).mean())
    pdp_cosine = float((final_pdp * baseline_pdp).sum(dim=-1).mean())
    target_pas_cosine = float(
        (final_pas * target_pas).sum(dim=-1).mean()
    )
    target_pdp_cosine = float(
        (final_pdp * target_pdp).sum(dim=-1).mean()
    )
    target_hvp_cosine = (
        float((final_hvp * target_hvp).sum(dim=-1).mean())
        if final_hvp is not None
        else None
    )
    channel_cosine = float(
        (prediction.conj() * baseline).sum().abs()
        / (
            prediction.norm() * baseline.norm()
        ).clamp_min(1e-30)
    )
    nearest_distance = source_distance[:, 0]
    nearest_delta_radius = delta_radius[:, 0]
    result = {
        "name": candidate_name,
        "validation": {
            "phase": str(phase_path) if phase_path else None,
            "projection": (
                str(projection_path) if projection_path else None
            ),
            "selection_policy": selection_policy,
            "tune_median": selected_projection["tune_median"],
            "audit": selected_projection["audit"],
            "scale": output_scale,
        },
        "baseline": manifest["baseline"],
        "full_neighbor_bank_size": len(train_pos),
        "seeds": seeds,
        "geometric_phase": {
            "k0_rad_per_meter": k0,
            "k1_rad_per_meter_per_subcarrier": k1,
            "initial_global_phase": global_phase,
            "residual_global_phase": residual_global_phase,
            "phase_seed_epsilon": args.phase_seed_epsilon,
            "source": source_config,
            "complex_neighbor_gate": complex_gate_config,
            "steering": steering_config,
            "nearest_xy_distance": {
                "mean": float(nearest_distance.mean()),
                "median": float(np.median(nearest_distance)),
                "max": float(nearest_distance.max()),
            },
            "delta_bs_radius": {
                "mean": float(nearest_delta_radius.mean()),
                "std": float(nearest_delta_radius.std()),
                "min": float(nearest_delta_radius.min()),
                "max": float(nearest_delta_radius.max()),
            },
        },
        "pas": {
            "k": 16,
            "correction": args.pas_correction,
            "radial_multiplier": args.pas_radial,
            "tangent_multiplier": args.pas_tangent,
            "blend": args.pas_blend,
            "transport": pas_transport,
            "transport_gate": transport_gate_stats,
            "hvp": hvp_target_config,
        },
        "pdp": {
            "k": 32,
            "correction": args.pdp_correction,
            "radial_multiplier": args.pdp_radial,
            "tangent_multiplier": args.pdp_tangent,
            "blend": args.pdp_blend,
            "post_bilateral": post_bilateral_config,
        },
        "projection_order": projection_order,
        "dual_mix": dual_mix,
        "source_blend": source_blend,
        "iterations": args.iterations,
        "source_rms": source_rms,
        "baseline_rms": baseline_rms,
        "output_scale": output_scale,
        "output_rms": output_rms,
        "relative_to_baseline": {
            "pas_cosine": pas_cosine,
            "pdp_cosine": pdp_cosine,
            "channel_cosine": channel_cosine,
        },
        "projection_fidelity": {
            "pas_cosine_to_target": target_pas_cosine,
            "pdp_cosine_to_target": target_pdp_cosine,
            "hvp_cosine_to_target": target_hvp_cosine,
        },
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "sha256": sha256(output_path),
        "manifest": manifest,
    }
    (outdir / "manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (outdir / "说明.txt").write_text(
        f"{candidate_name}\n\n"
        "几何相位与矩约束频谱候选。\n"
        f"相位源: K={source_k}, "
        f"radial_ratio={source_config.get('radial_ratio', 1.0)}, "
        f"power={source_config.get('distance_power', 0.0)}, "
        f"amplitude={source_config.get('amplitude', 'raw')}。\n"
        + (
            "阵列转向: "
            f"layout={steering_config['layout']}, "
            f"theta={steering_config['theta']:.9f}, "
            "horizontal="
            f"{steering_config['horizontal_coefficient']:.9f}, "
            "vertical="
            f"{steering_config['vertical_coefficient']:.9f}。\n"
            if steering_config is not None
            else ""
        )
        + f"相位参数: k0={k0:.9f} rad/m, "
        f"k1={k1:.9f} rad/(m*subcarrier), "
        f"initial_global={global_phase:.9f} rad, "
        f"residual_global={residual_global_phase:.9f} rad。\n"
        "频谱目标: GS34 基线 + 五折矩约束注意力；"
        f"PAS 融合 {args.pas_blend:g}，"
        f"PDP 融合 {args.pdp_blend:g}。\n"
        + (
            "PAS 物理搬移专家: "
            f"K={transport_k}, "
            f"distance_power={pas_transport['distance_power']}, "
            + (
                "expert_blend=逐切片低容量门控"
                f"({transport_gate_stats['architecture']}, "
                f"mean={transport_gate_stats['beta_mean']:.4f})。\n"
                if transport_gate_stats is not None
                else f"expert_blend={pas_transport['blend']}。\n"
            )
            if pas_transport is not None
            else ""
        )
        + (
            "HVP 稳健目标: "
            f"K={hvp_target_config.get('k', 32)}, "
            f"correction={hvp_target_config['correction']}, "
            f"blend={hvp_target_config['blend']}。\n"
            if hvp_target_config is not None
            else ""
        )
        + (
            "PDP 谱形一致性后投影: "
            f"K={post_bilateral_config.get('k', 32)}, "
            "distance_power="
            f"{post_bilateral_config['distance_power']}, "
            "anchor_strength="
            f"{post_bilateral_config['anchor_strength']}, "
            f"target_beta={post_bilateral_config['target_beta']}。\n"
            if post_bilateral_config is not None
            else ""
        )
        + f"重建: {projection_order} 交替投影 "
        f"{args.iterations} 轮；"
        + (
            "复邻居低容量门控: "
            + ", ".join(
                f"{component['role']}="
                f"{component['alpha']:g}"
                for component in complex_gate_components
            )
            + "；"
            if complex_gate_config is not None
            else ""
        )
        + (
            f"与 PHV-PDP 参考按 {dual_mix:g} 复通道混合；"
            if dual_mix is not None
            else ""
        )
        + (
            f"投影结果按 {source_blend:g} 与源复通道混合；"
            if source_blend is not None
            else ""
        )
        + f"幅度尺度={output_scale:.9f}，来自四个 tune split，"
        "audit split 未参与选参。\n",
        encoding="utf-8",
    )
    print(
        f"GEOPHASE_DONE name={candidate_name} path={output_path} "
        f"source_rms={source_rms:.9e} output_rms={output_rms:.9e} "
        f"pas_cos={pas_cosine:.6f} pdp_cos={pdp_cosine:.6f} "
        f"channel_cos={channel_cosine:.6f} "
        f"target_pas={target_pas_cosine:.6f} "
        f"target_pdp={target_pdp_cosine:.6f} "
        + (
            f"target_hvp={target_hvp_cosine:.6f} "
            if target_hvp_cosine is not None
            else ""
        )
        + f"sha256={result['sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
