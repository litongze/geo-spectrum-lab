#!/usr/bin/env python3
"""Sweep additive blends of two clean complex-neighbor gate families."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from _bootstrap import ROOT  # noqa: F401
from sweep_complex_neighbor_gate_blend import coefficients_for_split
from train_complex_neighbor_gate import (
    evaluate,
    optimal_calibration,
    sufficient,
)


def parse_grid(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir", default="cache/complex_neighbor_gate_probe"
    )
    parser.add_argument("--checkpoint-a", required=True)
    parser.add_argument("--checkpoint-b", required=True)
    parser.add_argument(
        "--alpha-a-grid",
        default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8",
    )
    parser.add_argument(
        "--alpha-b-grid",
        default="0,0.05,0.1,0.15,0.2,0.25,0.3,0.4",
    )
    parser.add_argument(
        "--tune-seeds", default="1890,3716,962,1022"
    )
    parser.add_argument("--audit-seed", default="2262")
    parser.add_argument("--external-name", default="geomall")
    parser.add_argument(
        "--external-indices",
        help="row-order indices for the external validation cache",
    )
    parser.add_argument(
        "--external-panel-dir",
        help="optional directory containing proxy_p{number}.npy subsets",
    )
    parser.add_argument("--external-panel-count", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--out", default="docs/complex_neighbor_gate/pair.json"
    )
    args = parser.parse_args()

    tune = [
        item for item in args.tune_seeds.split(",") if item
    ]
    names = tune + [args.audit_seed]
    if args.external_name:
        names.append(args.external_name)
    alpha_a = parse_grid(args.alpha_a_grid)
    alpha_b = parse_grid(args.alpha_b_grid)
    device = torch.device(args.device)
    cache_dir = Path(args.cache_dir)
    checkpoint_a = Path(args.checkpoint_a)
    checkpoint_b = Path(args.checkpoint_b)
    external_masks = {}
    if args.external_panel_dir:
        if not args.external_name or not args.external_indices:
            raise ValueError(
                "external panels require an external name and indices"
            )
        external_order = np.load(
            args.external_indices
        ).astype(np.int64)
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
        if not torch.allclose(
            train_baseline, train_baseline_b
        ) or not torch.allclose(val_baseline, val_baseline_b):
            raise RuntimeError(
                f"baseline coefficient mismatch for split {name}"
            )
        split_data[name] = {
            "training": training,
            "validation": validation,
            "train_baseline": train_baseline,
            "train_delta_a": train_a - train_baseline,
            "train_delta_b": train_b - train_baseline,
            "val_baseline": val_baseline,
            "val_delta_a": val_a - val_baseline,
            "val_delta_b": val_b - val_baseline,
        }

    rows = []
    for a in alpha_a:
        for b in alpha_b:
            scores = {}
            for name, data in split_data.items():
                train_coefficient = (
                    data["train_baseline"]
                    + a * data["train_delta_a"]
                    + b * data["train_delta_b"]
                )
                val_coefficient = (
                    data["val_baseline"]
                    + a * data["val_delta_a"]
                    + b * data["val_delta_b"]
                )
                train_cross, train_energy = sufficient(
                    train_coefficient, data["training"]
                )
                phase, scale = optimal_calibration(
                    train_cross, train_energy
                )
                scores[name] = evaluate(
                    val_coefficient,
                    data["validation"],
                    phase,
                    scale,
                )
                if name == args.external_name:
                    for panel_name, mask in external_masks.items():
                        panel_data = {
                            key: value[mask]
                            for key, value in data[
                                "validation"
                            ].items()
                        }
                        scores[
                            f"{name}_{panel_name}"
                        ] = evaluate(
                            val_coefficient[mask],
                            panel_data,
                            phase,
                            scale,
                        )
            rows.append(
                {
                    "alpha_a": a,
                    "alpha_b": b,
                    "scores": scores,
                }
            )

    baseline = next(
        row
        for row in rows
        if row["alpha_a"] == 0.0 and row["alpha_b"] == 0.0
    )
    for row in rows:
        tune_delta = {
            name: row["scores"][name]["NMSE"]
            - baseline["scores"][name]["NMSE"]
            for name in tune
        }
        row["tune_delta_nmse"] = tune_delta
        row["tune_delta_nmse_mean"] = float(
            np.mean(list(tune_delta.values()))
        )
        row["tune_delta_nmse_median"] = float(
            np.median(list(tune_delta.values()))
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
        row for row in rows if row["tune_delta_nmse_worst"] <= 0
    ]
    payload = {
        "selection_policy": (
            "two gate families are trained independently on each clean "
            "split; coefficients selected on four tune folds only"
        ),
        "checkpoint_a": args.checkpoint_a,
        "checkpoint_b": args.checkpoint_b,
        "tune_seeds": tune,
        "audit_seed": args.audit_seed,
        "external_name": args.external_name,
        "external_panels": list(external_masks),
        "grid": {"alpha_a": alpha_a, "alpha_b": alpha_b},
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
        "COMPLEX_GATE_PAIR_DONE "
        f"a={selected['alpha_a']:g} "
        f"b={selected['alpha_b']:g} "
        f"tune={selected['tune_delta_nmse_mean']:+.6f} "
        f"audit={selected['audit_delta_nmse']:+.6f} "
        f"external={selected.get('external_delta_nmse', 0):+.6f} "
        f"out={out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
