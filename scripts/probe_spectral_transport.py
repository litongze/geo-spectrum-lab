#!/usr/bin/env python3
"""Probe position-aware transport of neighbor PAS/PDP spectra."""
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
from wireless_twin.signal import pas_spectrum_phv, pdp_spectrum


def stable_unit(value: torch.Tensor) -> torch.Tensor:
    maximum = value.amax(dim=-1, keepdim=True)
    scaled = value / maximum.clamp_min(
        torch.finfo(value.dtype).tiny
    )
    norm = scaled.norm(dim=-1, keepdim=True)
    return torch.where(
        norm > 0,
        scaled / norm.clamp_min(torch.finfo(value.dtype).tiny),
        torch.zeros_like(value),
    )


def get_split_indices(
    size: int,
    split: int | str,
    external_name: str,
    external_indices: str | None,
) -> np.ndarray:
    if split == external_name:
        if external_indices is None:
            raise ValueError("external split requires --external-indices")
        return np.asarray(
            sorted(
                np.load(external_indices).astype(np.int64).tolist()
            ),
            dtype=np.int64,
        )
    return np.asarray(
        sorted(reproduce_val_indices(size, 0.1, int(split))),
        dtype=np.int64,
    )


def blend_cosine(
    expert_truth: torch.Tensor,
    transported_truth: torch.Tensor,
    expert_transported: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    if beta == 0:
        return expert_truth
    if beta == 1:
        return transported_truth
    left = 1.0 - beta
    numerator = (
        left * expert_truth + beta * transported_truth
    )
    denominator = torch.sqrt(
        left * left
        + beta * beta
        + 2.0 * left * beta * expert_transported
    ).clamp_min(torch.finfo(expert_truth.dtype).tiny)
    return numerator / denominator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", default="Round1_Map(2)")
    parser.add_argument(
        "--config", default="configs/gs39_steer_k16_r5.json"
    )
    parser.add_argument(
        "--expert-dir", default="cache/moment_attention_selected"
    )
    parser.add_argument("--domain", choices=("pas", "pdp"), required=True)
    parser.add_argument("--tune-seeds", default="1890,3716,962,1022")
    parser.add_argument("--audit-seed", type=int, default=2262)
    parser.add_argument("--external-indices")
    parser.add_argument("--external-name", default="testmatched")
    parser.add_argument("--k-values", default="8,16,32")
    parser.add_argument("--distance-powers", default="1,2,3")
    parser.add_argument(
        "--pas-scales", default="0,0.25,0.5,0.75,1,1.25"
    )
    parser.add_argument("--pas-horizontal-scales", default="1")
    parser.add_argument("--pas-vertical-scales", default="1")
    parser.add_argument("--pas-theta-offsets", default="0")
    parser.add_argument(
        "--pdp-slopes",
        default=(
            "-0.01,-0.005,-0.0025,-0.00125,-0.000625,"
            "0,0.000625,0.00125,0.0025,0.005,0.01"
        ),
    )
    parser.add_argument(
        "--blend-grid", default="0,0.025,0.05,0.1,0.2,0.3,0.5,1"
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out", default="docs/spectral_transport/result.json"
    )
    args = parser.parse_args()

    tune_seeds = [
        int(value) for value in args.tune_seeds.split(",") if value
    ]
    split_keys: list[int | str] = tune_seeds + [args.audit_seed]
    if args.external_indices:
        split_keys.append(args.external_name)
    k_values = [
        int(value) for value in args.k_values.split(",") if value
    ]
    powers = [
        float(value) for value in args.distance_powers.split(",") if value
    ]
    beta_grid = [
        float(value) for value in args.blend_grid.split(",") if value
    ]
    if args.domain == "pas":
        transports = [
            float(value) for value in args.pas_scales.split(",") if value
        ]
        horizontal_scales = [
            float(value)
            for value in args.pas_horizontal_scales.split(",")
            if value
        ]
        vertical_scales = [
            float(value)
            for value in args.pas_vertical_scales.split(",")
            if value
        ]
        theta_offsets = [
            float(value)
            for value in args.pas_theta_offsets.split(",")
            if value
        ]
        transport_variants = [
            (scale, horizontal, vertical, theta)
            for scale in transports
            for horizontal in horizontal_scales
            for vertical in vertical_scales
            for theta in theta_offsets
        ]
    else:
        transports = [
            float(value) for value in args.pdp_slopes.split(",") if value
        ]
        transport_variants = [
            (slope, 1.0, 1.0, 0.0) for slope in transports
        ]
    k_max = max(k_values)

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
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
    sufficient: dict[str, dict[int | str, dict[str, float]]] = {}

    for split in split_keys:
        val_idx = get_split_indices(
            len(positions),
            split,
            args.external_name,
            args.external_indices,
        )
        pool_idx = np.setdiff1d(all_idx, val_idx)
        distance, local = cKDTree(positions[pool_idx, :2]).query(
            positions[val_idx, :2], k=k_max
        )
        distance = np.asarray(distance, dtype=np.float64)
        neighbor_idx = pool_idx[np.asarray(local)]
        delta_radius = radius[val_idx, None] - radius[neighbor_idx]
        delta_unit = unit[val_idx, None] - unit[neighbor_idx]
        weight_bank = {}
        for k in k_values:
            for power in powers:
                weight = 1.0 / np.maximum(
                    distance[:, :k], 0.05
                ) ** power
                weight /= np.maximum(
                    weight.sum(axis=1, keepdims=True), 1e-30
                )
                weight_bank[f"k{k}_p{power:g}"] = weight

        expert_name = (
            f"pas_{split}_c1.npy"
            if args.domain == "pas"
            else f"pdp_{split}_c0.8.npy"
        )
        expert = torch.as_tensor(
            np.load(Path(args.expert_dir) / expert_name),
            dtype=torch.float32,
            device=device,
        )
        expert = stable_unit(expert)
        split_sum: dict[str, float] = {}
        split_count: dict[str, int] = {}

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
                        k_max,
                        spec.m,
                        spec.n,
                        spec.s,
                    ),
                    dtype=torch.complex64,
                    device=device,
                )
                truth_h = torch.as_tensor(
                    np.array(channels[val_idx[start:stop]], copy=True),
                    dtype=torch.complex64,
                    device=device,
                )
                if args.domain == "pas":
                    truth = stable_unit(
                        pas_spectrum_phv(truth_h, spec)
                    )
                else:
                    truth = stable_unit(pdp_spectrum(truth_h, spec))
                    range_phase = torch.as_tensor(
                        (
                            delta_radius[start:stop, :, None]
                            * subcarrier[None, None, :]
                        ),
                        dtype=torch.float32,
                        device=device,
                    )
                expert_batch = expert[start:stop]
                expert_truth = (
                    expert_batch * truth
                ).sum(dim=-1)
                baseline_name = f"{args.domain}_expert"
                split_sum[baseline_name] = split_sum.get(
                    baseline_name, 0.0
                ) + float(expert_truth.sum())
                split_count[baseline_name] = (
                    split_count.get(baseline_name, 0)
                    + expert_truth.numel()
                )

                for (
                    transport,
                    horizontal_scale,
                    vertical_scale,
                    theta_offset,
                ) in transport_variants:
                    if args.domain == "pas":
                        variant = dict(steering)
                        variant["theta"] = (
                            float(variant["theta"]) + theta_offset
                        )
                        variant["horizontal_coefficient"] = (
                            float(variant["horizontal_coefficient"])
                            * horizontal_scale
                        )
                        variant["vertical_coefficient"] = (
                            float(variant["vertical_coefficient"])
                            * vertical_scale
                        )
                        steering_phase = torch.as_tensor(
                            array_steering_phase(
                                delta_unit[start:stop],
                                spec,
                                variant,
                            ),
                            dtype=torch.float32,
                            device=device,
                        )
                        moved = source * torch.exp(
                            1j
                            * transport
                            * steering_phase[..., None]
                        )
                        neighbor_spectrum = stable_unit(
                            pas_spectrum_phv(
                                moved.reshape(
                                    batch * k_max,
                                    spec.m,
                                    spec.n,
                                    spec.s,
                                ),
                                spec,
                            )
                        ).reshape(
                            batch,
                            k_max,
                            spec.n,
                            spec.s,
                            spec.mh * spec.mv,
                        )
                    else:
                        moved = source * torch.exp(
                            1j * transport * range_phase
                        )[:, :, None, None, :]
                        neighbor_spectrum = stable_unit(
                            pdp_spectrum(
                                moved.reshape(
                                    batch * k_max,
                                    spec.m,
                                    spec.n,
                                    spec.s,
                                ),
                                spec,
                            )
                        ).reshape(
                            batch,
                            k_max,
                            spec.m,
                            spec.n,
                            spec.s,
                        )

                    for weight_name, weight_np in weight_bank.items():
                        k = int(weight_name.split("_")[0][1:])
                        weight = torch.as_tensor(
                            weight_np[start:stop],
                            dtype=torch.float32,
                            device=device,
                        )
                        extra_dims = [1] * (
                            neighbor_spectrum.ndim - 2
                        )
                        transported = stable_unit(
                            (
                                weight.reshape(
                                    batch, k, *extra_dims
                                )
                                * neighbor_spectrum[:, :k]
                            ).sum(dim=1)
                        )
                        transported_truth = (
                            transported * truth
                        ).sum(dim=-1)
                        expert_transported = (
                            expert_batch * transported
                        ).sum(dim=-1)
                        transport_name = (
                            (
                                f"scale{transport:g}"
                                f"_hs{horizontal_scale:g}"
                                f"_vs{vertical_scale:g}"
                                f"_to{theta_offset:g}"
                            )
                            if args.domain == "pas"
                            else f"slope{transport:g}"
                        )
                        for beta in beta_grid:
                            name = (
                                f"{args.domain}_{weight_name}_"
                                f"{transport_name}_b{beta:g}"
                            )
                            score = blend_cosine(
                                expert_truth,
                                transported_truth,
                                expert_transported,
                                beta,
                            )
                            split_sum[name] = split_sum.get(
                                name, 0.0
                            ) + float(score.sum())
                            split_count[name] = (
                                split_count.get(name, 0)
                                + score.numel()
                            )
                    del moved, neighbor_spectrum
                del source, truth_h, truth

        for name, total in split_sum.items():
            sufficient.setdefault(name, {})[split] = {
                "score": total / split_count[name]
            }
        print(
            f"[spectral-transport] domain={args.domain} split={split} "
            f"expert={sufficient[baseline_name][split]['score']:.6f}",
            flush=True,
        )
        del expert
        torch.cuda.empty_cache()

    ranked = []
    for name, rows in sufficient.items():
        if any(split not in rows for split in split_keys):
            continue
        tune = [rows[split]["score"] for split in tune_seeds]
        ranked.append(
            {
                "name": name,
                "tune_median": float(np.median(tune)),
                "tune_mean": float(np.mean(tune)),
                "tune_worst": float(np.min(tune)),
                "audit": rows[args.audit_seed]["score"],
                "external": (
                    rows[args.external_name]["score"]
                    if args.external_indices
                    else None
                ),
                "scores": {
                    str(split): rows[split]["score"]
                    for split in split_keys
                },
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
            "transport uses only source channels and geometry; "
            "config ranked on tune splits"
        ),
        "domain": args.domain,
        "tune_seeds": tune_seeds,
        "audit_seed": args.audit_seed,
        "external_name": (
            args.external_name if args.external_indices else None
        ),
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
    print(f"SPECTRAL_TRANSPORT_DONE out={out}", flush=True)


if __name__ == "__main__":
    main()
