#!/usr/bin/env python3
"""Sweep complex interpolation after geometry-driven phase alignment."""
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
from sweep_moment_attention import moment_project
from wireless_twin.data.map_loader import load_point_cloud
from wireless_twin.data.setup_config import load_setup
from wireless_twin.models.raytrace2 import build_heightmap


def normalize(weight: np.ndarray) -> np.ndarray:
    weight = np.maximum(weight, 0.0)
    return weight / np.maximum(weight.sum(axis=1, keepdims=True), 1e-30)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", default="Round1_Map(2)")
    parser.add_argument(
        "--config", default="configs/gs37_geophase_moment.json"
    )
    parser.add_argument("--tune-seeds", default="1890,3716,962,1022")
    parser.add_argument("--audit-seed", type=int, default=2262)
    parser.add_argument(
        "--external-indices",
        help="optional external holdout index file used only for reporting",
    )
    parser.add_argument("--external-name", default="external")
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument(
        "--anisotropic-selection-only",
        action="store_true",
        help=(
            "evaluate neighbors reselected by anisotropic distance from "
            "the full k-max candidate pool"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steering-result")
    parser.add_argument(
        "--out",
        default="docs/geometric_phase_neighbors/result.json",
    )
    args = parser.parse_args()

    tune_seeds = [
        int(value) for value in args.tune_seeds.split(",") if value
    ]
    seeds = tune_seeds + [args.audit_seed]
    split_keys: list[int | str] = list(seeds)
    if args.external_indices:
        split_keys.append(args.external_name)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    k0 = float(config["phase"]["k0_rad_per_meter"])
    k1 = float(
        config["phase"]["k1_rad_per_meter_per_subcarrier"]
    )
    datadir = Path(args.datadir)
    spec = load_setup(datadir / "Round1_Setup.json")
    steering = None
    if args.steering_result:
        steering_payload = json.loads(
            Path(args.steering_result).read_text(encoding="utf-8")
        )
        steering = steering_payload["ranked"][0]
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

    points = load_point_cloud(datadir / "Round1_Map.ply")
    heightmap, x0, y0, resolution = build_heightmap(points)
    gx = np.clip(
        np.floor((positions[:, 0] - x0) / resolution).astype(np.int64),
        0,
        heightmap.shape[0] - 1,
    )
    gy = np.clip(
        np.floor((positions[:, 1] - y0) / resolution).astype(np.int64),
        0,
        heightmap.shape[1] - 1,
    )
    indoor = heightmap[gx, gy] > 2.0
    split_data = {}

    for seed in split_keys:
        if seed == args.external_name:
            val_idx = np.asarray(
                sorted(
                    np.load(args.external_indices)
                    .astype(np.int64)
                    .tolist()
                ),
                dtype=np.int64,
            )
        else:
            val_idx = np.asarray(
                sorted(
                    reproduce_val_indices(
                        len(positions), 0.1, int(seed)
                    )
                ),
                dtype=np.int64,
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
                    (
                        k0 + k1 * subcarrier[None, None]
                    )
                    * delta_radius[start:stop, :, None]
                )
                source *= torch.as_tensor(
                    np.exp(1j * phase),
                    dtype=torch.complex64,
                    device=device,
                )[:, :, None, None, :]
                if steering is not None:
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
        split_data[seed] = {
            "val_idx": val_idx,
            "distance": distance,
            "neighbor_idx": neighbor_idx,
            "radial_delta": radial_delta,
            "tangent_delta": tangent_delta,
            "same_indoor": indoor[neighbor_idx] == indoor[val_idx, None],
            "cross": cross,
            "gram": gram,
            "truth_energy": truth_energy,
        }
        print(
            f"[neighbor-phase] split={seed} "
            f"nearest_mean={distance[:, 0].mean():.4f}",
            flush=True,
        )
        torch.cuda.empty_cache()

    def geometry_weights(data: dict) -> dict[str, np.ndarray]:
        distance = data["distance"]
        radial_delta = data["radial_delta"]
        tangent_delta = data["tangent_delta"]
        same_indoor = data["same_indoor"]
        batch, k_max = distance.shape
        result = {}

        for rank in range(min(4, k_max)):
            weight = np.zeros((batch, k_max), dtype=np.float64)
            weight[:, rank] = 1.0
            result[f"rank{rank + 1}"] = weight

        selected_k_values = [
            value
            for value in (8, 12, 16, 24, 32)
            if value <= k_max
        ]
        for ratio in (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0):
            effective = np.sqrt(
                np.square(radial_delta * ratio)
                + np.square(tangent_delta)
            )
            for k in selected_k_values:
                selected = np.argpartition(
                    effective, k - 1, axis=1
                )[:, :k]
                for power in (1.5, 2.0, 2.5, 3.0):
                    raw = np.zeros_like(effective)
                    selected_distance = np.take_along_axis(
                        effective, selected, axis=1
                    )
                    np.put_along_axis(
                        raw,
                        selected,
                        1.0
                        / np.maximum(selected_distance, 0.05)
                        ** power,
                        axis=1,
                    )
                    result[
                        f"anisel_k{k}_r{ratio:g}_p{power:g}"
                    ] = normalize(raw)
        if args.anisotropic_selection_only:
            return result

        k_values = [
            value
            for value in (2, 3, 4, 6, 8, 12, 16)
            if value <= k_max
        ]
        for k in k_values:
            mask = np.zeros_like(distance)
            mask[:, :k] = 1.0
            for power in (0.0, 0.5, 1.0, 2.0, 3.0, 4.0):
                raw = mask / np.maximum(distance, 0.05) ** power
                result[f"idw_k{k}_p{power:g}"] = normalize(raw)
            for bandwidth in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
                raw = mask * np.exp(
                    -0.5 * np.square(distance / bandwidth)
                )
                result[f"gauss_k{k}_h{bandwidth:g}"] = normalize(raw)

        for k in k_values:
            mask = np.zeros_like(distance)
            mask[:, :k] = 1.0
            for ratio in (
                0.25,
                0.5,
                0.75,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                8.0,
                12.0,
            ):
                effective = np.sqrt(
                    np.square(radial_delta * ratio)
                    + np.square(tangent_delta)
                )
                for power in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
                    raw = (
                        mask
                        / np.maximum(effective, 0.05) ** power
                    )
                    result[
                        f"anis_k{k}_r{ratio:g}_p{power:g}"
                    ] = normalize(raw)

        base = 1.0 / np.maximum(distance, 0.05) ** 2
        for mismatch_weight in (0.0, 0.1, 0.3, 0.5, 0.75):
            raw = base * np.where(
                same_indoor, 1.0, mismatch_weight
            )
            result[
                f"indoor_idw_p2_m{mismatch_weight:g}"
            ] = normalize(raw)

        scale = np.maximum(
            np.median(distance, axis=1, keepdims=True), 0.3
        )
        delta_rt = np.stack(
            [radial_delta / scale, tangent_delta / scale], axis=-1
        )
        for k in k_values:
            mask = np.zeros_like(distance)
            mask[:, :k] = 1.0
            for power in (0.0, 1.0, 2.0):
                base = normalize(
                    mask / np.maximum(distance, 0.05) ** power
                )
                logits = torch.as_tensor(
                    np.log(np.maximum(base, 1e-30))[:, :, None],
                    dtype=torch.float32,
                )
                delta_t = torch.as_tensor(
                    delta_rt, dtype=torch.float32
                )
                for correction in (0.5, 1.0):
                    for radial_axis in (0.5, 1.0, 2.0):
                        projected, _ = moment_project(
                            logits,
                            delta_t,
                            correction,
                            torch.tensor(
                                [radial_axis, 1.0],
                                dtype=torch.float32,
                            ),
                            12,
                            0.03,
                        )
                        result[
                            f"moment_k{k}_p{power:g}"
                            f"_c{correction:g}_r{radial_axis:g}"
                        ] = projected[:, :, 0].numpy().astype(
                            np.float64
                        )
        return result

    def content_weights(
        data: dict, base_weights: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        gram = data["gram"]
        energy = np.maximum(
            np.real(np.diagonal(gram, axis1=1, axis2=2)),
            1e-30,
        )
        denominator = np.sqrt(
            energy[:, :, None] * energy[:, None, :]
        )
        real_coherence = np.real(gram) / denominator
        abs_coherence = np.abs(gram) / denominator
        eye = np.eye(args.k_max, dtype=bool)[None]
        real_agreement = np.where(
            eye, 0.0, np.maximum(real_coherence, 0.0)
        ).sum(axis=2) / max(args.k_max - 1, 1)
        abs_agreement = np.where(
            eye, 0.0, abs_coherence
        ).sum(axis=2) / max(args.k_max - 1, 1)
        result = {}
        for agreement_name, agreement in (
            ("real", real_agreement),
            ("abs", abs_agreement),
        ):
            medoid = np.zeros_like(agreement)
            medoid[
                np.arange(len(agreement)), agreement.argmax(axis=1)
            ] = 1.0
            result[f"medoid_{agreement_name}"] = medoid
            for base_name in (
                "idw_k8_p1",
                "idw_k8_p2",
                "idw_k8_p4",
            ):
                if base_name not in base_weights:
                    continue
                base = base_weights[base_name]
                centered = agreement - agreement.mean(
                    axis=1, keepdims=True
                )
                for beta in (1.0, 2.0, 5.0, 10.0):
                    result[
                        f"{base_name}_{agreement_name}_b{beta:g}"
                    ] = normalize(base * np.exp(beta * centered))
        return result

    split_weights = {}
    names = None
    for seed in split_keys:
        geometry = geometry_weights(split_data[seed])
        if not args.anisotropic_selection_only:
            geometry.update(
                content_weights(split_data[seed], geometry)
            )
        split_weights[seed] = geometry
        current_names = set(geometry)
        names = (
            current_names
            if names is None
            else names.intersection(current_names)
        )
    assert names is not None

    def coefficients(
        data: dict, weight: np.ndarray, amplitude_mode: str
    ) -> np.ndarray:
        if amplitude_mode == "raw":
            return weight
        energy = np.maximum(
            np.real(np.diagonal(
                data["gram"], axis1=1, axis2=2
            )),
            1e-30,
        )
        amplitude = np.sqrt(energy)
        target_amplitude = (weight * amplitude).sum(
            axis=1, keepdims=True
        )
        return weight * target_amplitude / amplitude

    sufficient = {}
    for name in sorted(names):
        for amplitude_mode in ("raw", "equalized"):
            full_name = f"{name}_{amplitude_mode}"
            sufficient[full_name] = {}
            for seed in split_keys:
                data = split_data[seed]
                coeff = coefficients(
                    data,
                    split_weights[seed][name],
                    amplitude_mode,
                )
                cross_per_point = np.einsum(
                    "bk,bk->b", coeff, data["cross"]
                )
                energy_per_point = np.real(
                    np.einsum(
                        "bk,bkj,bj->b",
                        coeff,
                        data["gram"],
                        coeff,
                    )
                )
                sufficient[full_name][seed] = {
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
        tune_cross = sum(rows[seed]["cross"] for seed in tune_seeds)
        tune_prediction_energy = sum(
            rows[seed]["prediction_energy"] for seed in tune_seeds
        )
        global_phase = float(np.angle(tune_cross))
        phase_factor = np.exp(-1j * global_phase)
        scale = max(
            float(np.real(tune_cross * phase_factor))
            / max(tune_prediction_energy, 1e-30),
            0.0,
        )
        scores = {}
        for seed in split_keys:
            row = rows[seed]
            cross_real = float(
                np.real(row["cross"] * phase_factor)
            )
            nmse = (
                row["truth_energy"]
                + scale**2 * row["prediction_energy"]
                - 2.0 * scale * cross_real
            ) / max(row["truth_energy"], 1e-30)
            coherence = (
                np.abs(row["cross"])
                / np.sqrt(
                    max(
                        row["prediction_energy"]
                        * row["truth_energy"],
                        1e-30,
                    )
                )
            )
            scores[str(seed)] = {
                "NMSE": float(nmse),
                "coherence": float(coherence),
                "fixed_phase_real_coherence": float(
                    cross_real
                    / np.sqrt(
                        max(
                            row["prediction_energy"]
                            * row["truth_energy"],
                            1e-30,
                        )
                    )
                ),
            }
        tune_nmse = [
            scores[str(seed)]["NMSE"] for seed in tune_seeds
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
            "phase/scale/config selected on four tune splits; "
            "audit external"
        ),
        "phase": {"k0": k0, "k1": k1},
        "tune_seeds": tune_seeds,
        "audit_seed": args.audit_seed,
        "external_indices": args.external_indices,
        "external_name": (
            args.external_name if args.external_indices else None
        ),
        "k_max": args.k_max,
        "steering_result": args.steering_result,
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
    print(f"GEOMETRIC_PHASE_NEIGHBORS_DONE out={out}", flush=True)


if __name__ == "__main__":
    main()
