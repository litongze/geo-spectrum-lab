#!/usr/bin/env python3
"""Probe local barycentric triangles as a clean complex-channel source."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from _bootstrap import ROOT  # noqa: F401
from complex_neighbor_gate import baseline_coefficients
from sweep_complex_neighbor_gate_blend import coefficients_for_split
from train_complex_neighbor_gate import (
    evaluate,
    optimal_calibration,
    sufficient,
)


def parse_int_grid(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def parse_float_grid(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item]


def triangle_weights(
    data: dict[str, torch.Tensor],
    neighbor_count: int,
) -> tuple[dict[str, torch.Tensor], float]:
    """Select a positive triangle containing the query origin."""
    features = data["features"].detach().cpu().numpy()
    base = data["base_weight"].detach().cpu().numpy()
    coordinate = features[:, :neighbor_count, 3:5].astype(np.float64)
    effective = features[:, :neighbor_count, 2].astype(np.float64)
    combinations = np.asarray(
        list(itertools.combinations(range(neighbor_count), 3)),
        dtype=np.int64,
    )
    point = coordinate[:, combinations]
    x1, y1 = point[:, :, 0, 0], point[:, :, 0, 1]
    x2, y2 = point[:, :, 1, 0], point[:, :, 1, 1]
    x3, y3 = point[:, :, 2, 0], point[:, :, 2, 1]
    denominator = (
        (y2 - y3) * (x1 - x3)
        + (x3 - x2) * (y1 - y3)
    )
    stable = np.abs(denominator) > 1e-8
    safe_denominator = np.where(stable, denominator, 1.0)
    weight1 = (
        -(y2 - y3) * x3 + (x3 - x2) * -y3
    ) / safe_denominator
    weight2 = (
        -(y3 - y1) * x3 + (x1 - x3) * -y3
    ) / safe_denominator
    weight = np.stack(
        [weight1, weight2, 1.0 - weight1 - weight2],
        axis=2,
    )
    valid = stable & (weight.min(axis=2) >= -1e-7)
    triangle_effective = effective[:, combinations]
    coordinate_radius = np.linalg.norm(point, axis=3)
    area = np.abs(denominator) * 0.5
    score = {
        "minimax": triangle_effective.max(axis=2),
        "sum": triangle_effective.sum(axis=2),
        "compact": (
            np.square(coordinate_radius).sum(axis=2)
            / np.maximum(area, 1e-8)
        ),
    }

    result = {}
    row_index = np.arange(len(base))
    has_triangle = valid.any(axis=1)
    for name, current_score in score.items():
        current_score = np.where(valid, current_score, np.inf)
        selected_index = current_score.argmin(axis=1)
        selected_vertex = combinations[selected_index]
        selected_weight = weight[row_index, selected_index]
        selected_weight = np.maximum(selected_weight, 0.0)
        selected_weight /= np.maximum(
            selected_weight.sum(axis=1, keepdims=True), 1e-30
        )
        output = base.copy()
        output[has_triangle] = 0.0
        for column in range(3):
            output[
                row_index[has_triangle],
                selected_vertex[has_triangle, column],
            ] = selected_weight[has_triangle, column]
        result[name] = torch.as_tensor(
            output,
            dtype=torch.float32,
            device=data["base_weight"].device,
        )
    return result, float(has_triangle.mean())


def triangle_coefficients(
    data: dict[str, torch.Tensor],
    neighbor_counts: list[int],
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    result = {}
    coverage = {}
    for neighbor_count in neighbor_counts:
        weights, current_coverage = triangle_weights(
            data, neighbor_count
        )
        for strategy, weight in weights.items():
            name = f"k{neighbor_count}_{strategy}"
            result[name] = baseline_coefficients(
                weight, data["amplitude"]
            )
            coverage[name] = current_coverage
    return result, coverage


def subset(
    data: dict[str, torch.Tensor], mask: torch.Tensor
) -> dict[str, torch.Tensor]:
    return {name: value[mask] for name, value in data.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir", default="cache/complex_neighbor_gate_probe"
    )
    parser.add_argument(
        "--checkpoint-a",
        default="checkpoints/complex_neighbor_mlp16_geomall_e300",
    )
    parser.add_argument(
        "--checkpoint-b",
        default="checkpoints/complex_neighbor_set16_e300",
    )
    parser.add_argument("--alpha-a", type=float, default=0.5)
    parser.add_argument("--alpha-b", type=float, default=0.1)
    parser.add_argument("--neighbor-counts", default="4,6,8,12,16")
    parser.add_argument(
        "--gamma-grid", default="0,0.05,0.1,0.2,0.3,0.5,0.75,1"
    )
    parser.add_argument(
        "--blend-modes", default="convex,source_residual"
    )
    parser.add_argument(
        "--tune-seeds", default="1890,3716,962,1022"
    )
    parser.add_argument("--audit-seed", default="2262")
    parser.add_argument("--external-name", default="geomall")
    parser.add_argument(
        "--external-indices",
        default="cache/test_geometry_panel/proxy_all.npy",
    )
    parser.add_argument(
        "--external-panel-dir",
        default="cache/test_geometry_panel",
    )
    parser.add_argument("--external-panel-count", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--out",
        default="docs/local_triangle_source/result.json",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    checkpoint_a = Path(args.checkpoint_a)
    checkpoint_b = Path(args.checkpoint_b)
    tune = [item for item in args.tune_seeds.split(",") if item]
    names = [*tune, args.audit_seed]
    if args.external_name:
        names.append(args.external_name)
    neighbor_counts = parse_int_grid(args.neighbor_counts)
    gamma_grid = parse_float_grid(args.gamma_grid)
    blend_modes = [
        item for item in args.blend_modes.split(",") if item
    ]
    device = torch.device(args.device)

    external_masks = {}
    if args.external_name:
        external_order = np.load(args.external_indices).astype(np.int64)
        panel_dir = Path(args.external_panel_dir)
        for number in range(args.external_panel_count):
            panel = np.load(
                panel_dir / f"proxy_p{number}.npy"
            ).astype(np.int64)
            external_masks[f"p{number}"] = torch.as_tensor(
                np.isin(external_order, panel),
                dtype=torch.bool,
                device=device,
            )

    split_data = {}
    all_coverages = {}
    for name in names:
        (
            training,
            validation,
            train_baseline,
            train_a,
            val_baseline,
            val_a,
        ) = coefficients_for_split(
            name, cache_dir, checkpoint_a, device
        )
        (
            _,
            _,
            train_baseline_b,
            train_b,
            val_baseline_b,
            val_b,
        ) = coefficients_for_split(
            name, cache_dir, checkpoint_b, device
        )
        if not torch.equal(
            train_baseline, train_baseline_b
        ) or not torch.equal(val_baseline, val_baseline_b):
            raise RuntimeError(f"baseline mismatch for split {name}")
        train_triangle, train_coverage = triangle_coefficients(
            training, neighbor_counts
        )
        val_triangle, val_coverage = triangle_coefficients(
            validation, neighbor_counts
        )
        train_dual = (
            train_baseline
            + args.alpha_a * (train_a - train_baseline)
            + args.alpha_b * (train_b - train_baseline)
        )
        val_dual = (
            val_baseline
            + args.alpha_a * (val_a - val_baseline)
            + args.alpha_b * (val_b - val_baseline)
        )
        split_data[name] = {
            "training": training,
            "validation": validation,
            "train_baseline": train_baseline,
            "val_baseline": val_baseline,
            "train_dual": train_dual,
            "val_dual": val_dual,
            "train_triangle": train_triangle,
            "val_triangle": val_triangle,
        }
        all_coverages[name] = {
            candidate: {
                "train": train_coverage[candidate],
                "validation": val_coverage[candidate],
            }
            for candidate in train_triangle
        }
        print(
            f"[triangle] split={name} "
            f"coverage={val_coverage[f'k{max(neighbor_counts)}_minimax']:.3f}",
            flush=True,
        )

    rows = []
    candidate_names = sorted(
        next(iter(split_data.values()))["train_triangle"]
    )
    for candidate_name in candidate_names:
        for blend_mode in blend_modes:
            for gamma in gamma_grid:
                scores = {}
                for name, data in split_data.items():
                    if blend_mode == "convex":
                        train_direction = (
                            data["train_triangle"][candidate_name]
                            - data["train_dual"]
                        )
                        val_direction = (
                            data["val_triangle"][candidate_name]
                            - data["val_dual"]
                        )
                    elif blend_mode == "source_residual":
                        train_direction = (
                            data["train_triangle"][candidate_name]
                            - data["train_baseline"]
                        )
                        val_direction = (
                            data["val_triangle"][candidate_name]
                            - data["val_baseline"]
                        )
                    else:
                        raise ValueError(
                            f"unknown blend mode {blend_mode}"
                        )
                    train_coefficient = (
                        data["train_dual"] + gamma * train_direction
                    )
                    val_coefficient = (
                        data["val_dual"] + gamma * val_direction
                    )
                    cross, energy = sufficient(
                        train_coefficient, data["training"]
                    )
                    phase, scale = optimal_calibration(cross, energy)
                    scores[name] = evaluate(
                        val_coefficient,
                        data["validation"],
                        phase,
                        scale,
                    )
                    if name == args.external_name:
                        for panel_name, mask in external_masks.items():
                            scores[
                                f"{name}_{panel_name}"
                            ] = evaluate(
                                val_coefficient[mask],
                                subset(data["validation"], mask),
                                phase,
                                scale,
                            )
                rows.append(
                    {
                        "candidate": candidate_name,
                        "blend_mode": blend_mode,
                        "gamma": gamma,
                        "scores": scores,
                    }
                )

    baseline = next(
        row
        for row in rows
        if row["gamma"] == 0.0
    )
    for row in rows:
        tune_delta = {
            name: (
                row["scores"][name]["NMSE"]
                - baseline["scores"][name]["NMSE"]
            )
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
            row["scores"][args.audit_seed]["NMSE"]
            - baseline["scores"][args.audit_seed]["NMSE"]
        )
        if args.external_name:
            row["external_delta_nmse"] = float(
                row["scores"][args.external_name]["NMSE"]
                - baseline["scores"][args.external_name]["NMSE"]
            )
            row["external_panel_delta_nmse"] = {
                panel_name: float(
                    row["scores"][
                        f"{args.external_name}_{panel_name}"
                    ]["NMSE"]
                    - baseline["scores"][
                        f"{args.external_name}_{panel_name}"
                    ]["NMSE"]
                )
                for panel_name in external_masks
            }
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
        and row.get("external_delta_nmse", 0.0) <= 0.0
    ]
    payload = {
        "selection_policy": (
            "triangles use source coordinates only; gate models and "
            "calibration are fit on each split's disjoint training pool"
        ),
        "alpha_a": args.alpha_a,
        "alpha_b": args.alpha_b,
        "neighbor_counts": neighbor_counts,
        "gamma_grid": gamma_grid,
        "blend_modes": blend_modes,
        "coverages": all_coverages,
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
        "LOCAL_TRIANGLE_DONE "
        f"candidate={selected['candidate']} "
        f"mode={selected['blend_mode']} "
        f"gamma={selected['gamma']:g} "
        f"tune={selected['tune_delta_nmse_mean']:+.6f} "
        f"worst={selected['tune_delta_nmse_worst']:+.6f} "
        f"audit={selected['audit_delta_nmse']:+.6f} "
        f"external={selected.get('external_delta_nmse', 0):+.6f} "
        f"out={out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
