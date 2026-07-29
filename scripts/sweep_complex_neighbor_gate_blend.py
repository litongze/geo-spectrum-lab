#!/usr/bin/env python3
"""Sweep conservative blends between baseline and learned neighbor gates."""
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
)
from train_complex_neighbor_gate import (
    evaluate,
    load_data,
    optimal_calibration,
    sufficient,
    tensors,
)


def parse_grid(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item]


def coefficients_for_split(
    name: str,
    cache_dir: Path,
    checkpoint_dir: Path,
    device: torch.device,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    training = tensors(load_data(cache_dir / f"{name}_train.npz"), device)
    validation = tensors(load_data(cache_dir / f"{name}_val.npz"), device)
    checkpoint = torch.load(
        checkpoint_dir / f"s{name}.pt",
        map_location=device,
        weights_only=False,
    )
    checkpoint_names = tuple(checkpoint["feature_names"])
    indices = [FEATURE_NAMES.index(item) for item in checkpoint_names]
    model = ComplexNeighborGate(
        len(checkpoint_names),
        checkpoint.get("architecture", "mlp16"),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    mean = torch.as_tensor(
        checkpoint["feature_mean"], dtype=torch.float32, device=device
    )
    std = torch.as_tensor(
        checkpoint["feature_std"], dtype=torch.float32, device=device
    )

    def learned(
        data: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = (
            (data["features"][..., indices] - mean) / std
        ).clamp(-8.0, 8.0)
        learned_coefficient, _ = model.coefficients(
            normalized, data["base_weight"], data["amplitude"]
        )
        baseline_coefficient = baseline_coefficients(
            data["base_weight"], data["amplitude"]
        )
        return baseline_coefficient, learned_coefficient

    with torch.inference_mode():
        train_baseline, train_learned = learned(training)
        val_baseline, val_learned = learned(validation)
    return (
        training,
        validation,
        train_baseline,
        train_learned,
        val_baseline,
        val_learned,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir", default="cache/complex_neighbor_gate_probe"
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints/complex_neighbor_gate_e80",
    )
    parser.add_argument(
        "--tune-seeds", default="1890,3716,962,1022"
    )
    parser.add_argument("--audit-seed", default="2262")
    parser.add_argument("--external-name", default="testmatched")
    parser.add_argument(
        "--alpha-grid",
        default="0,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--out", default="docs/complex_neighbor_gate/blend_e80.json"
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    tune_names = [
        item for item in args.tune_seeds.split(",") if item
    ]
    names = [*tune_names, args.audit_seed]
    if args.external_name:
        names.append(args.external_name)
    alphas = parse_grid(args.alpha_grid)
    device = torch.device(args.device)
    split_rows: dict[str, dict[str, dict[str, float]]] = {}

    for name in names:
        (
            training,
            validation,
            train_baseline,
            train_learned,
            val_baseline,
            val_learned,
        ) = coefficients_for_split(
            name, cache_dir, checkpoint_dir, device
        )
        split_rows[name] = {}
        with torch.inference_mode():
            for alpha in alphas:
                train_coefficient = (
                    (1.0 - alpha) * train_baseline
                    + alpha * train_learned
                )
                val_coefficient = (
                    (1.0 - alpha) * val_baseline
                    + alpha * val_learned
                )
                cross, energy = sufficient(
                    train_coefficient, training
                )
                phase, scale = optimal_calibration(cross, energy)
                split_rows[name][str(alpha)] = evaluate(
                    val_coefficient,
                    validation,
                    phase,
                    scale,
                )

    ranked = []
    for alpha in alphas:
        key = str(alpha)
        tune_delta = np.asarray(
            [
                split_rows[name][key]["NMSE"]
                - split_rows[name][str(alphas[0])]["NMSE"]
                for name in tune_names
            ],
            dtype=np.float64,
        )
        row = {
            "alpha": alpha,
            "tune_delta_nmse_median": float(np.median(tune_delta)),
            "tune_delta_nmse_mean": float(np.mean(tune_delta)),
            "tune_delta_nmse_worst": float(np.max(tune_delta)),
            "tune_delta_nmse": {
                name: float(delta)
                for name, delta in zip(tune_names, tune_delta)
            },
        }
        for diagnostic_name in names[len(tune_names) :]:
            row[f"{diagnostic_name}_delta_nmse"] = float(
                split_rows[diagnostic_name][key]["NMSE"]
                - split_rows[diagnostic_name][str(alphas[0])][
                    "NMSE"
                ]
            )
        ranked.append(row)
    eligible = [
        row
        for row in ranked
        if row["tune_delta_nmse_worst"] <= 0.0
    ]
    selected = min(
        eligible or ranked,
        key=lambda row: row["tune_delta_nmse_mean"],
    )
    result = {
        "selection_policy": (
            "alpha selected by minimum four-fold tune mean NMSE among "
            "candidates improving every tune fold; audit/external are "
            "diagnostics only"
        ),
        "checkpoint_dir": str(checkpoint_dir),
        "tune_seeds": tune_names,
        "audit_seed": args.audit_seed,
        "external_name": args.external_name,
        "selected": selected,
        "ranked": sorted(
            ranked, key=lambda row: row["tune_delta_nmse_mean"]
        ),
        "splits": split_rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n"
    )
    print(
        "COMPLEX_GATE_BLEND_DONE "
        f"alpha={selected['alpha']} "
        f"tune_mean={selected['tune_delta_nmse_mean']:+.6f} "
        f"audit={selected[f'{args.audit_seed}_delta_nmse']:+.6f} "
        f"external={selected[f'{args.external_name}_delta_nmse']:+.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
