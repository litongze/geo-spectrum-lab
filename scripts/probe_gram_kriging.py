#!/usr/bin/env python3
"""Probe regularized Gram-kriging complex neighbor coefficients."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_gate_data(path: Path) -> dict[str, np.ndarray]:
    payload = np.load(path)
    return {
        name: payload[name]
        for name in (
            "amplitude",
            "base_weight",
            "gram",
            "cross",
            "truth_energy",
        )
    }


def baseline_coefficients(data: dict[str, np.ndarray]) -> np.ndarray:
    base = data["base_weight"].astype(np.float64)
    amplitude = data["amplitude"].astype(np.float64)
    target = (base * amplitude).sum(axis=1, keepdims=True)
    return (base * target / np.maximum(amplitude, 1e-30)).astype(
        np.complex128
    )


def gram_kriging_coefficients(
    data: dict[str, np.ndarray],
    diagonal_loading: float,
    prior_power: float,
    blend: float,
    amplitude_mode: str,
) -> np.ndarray:
    amplitude = data["amplitude"].astype(np.float64)
    gram = data["gram"].astype(np.complex128)
    correlation = gram / np.maximum(
        amplitude[:, :, None] * amplitude[:, None, :],
        1e-30,
    )
    k = correlation.shape[1]
    system = correlation + diagonal_loading * np.eye(
        k, dtype=np.complex128
    )[None]
    base = data["base_weight"].astype(np.float64)
    prior = np.maximum(base, 1e-30) ** prior_power
    prior /= np.maximum(prior.sum(axis=1, keepdims=True), 1e-30)
    solved = np.linalg.solve(system, prior[..., None])[..., 0]
    denominator = solved.sum(axis=1, keepdims=True)
    solved /= np.where(
        np.abs(denominator) > 1e-8, denominator, 1.0
    )
    spatial = (1.0 - blend) * base + blend * solved
    if amplitude_mode == "base":
        target_amplitude = (
            base * amplitude
        ).sum(axis=1, keepdims=True)
    elif amplitude_mode == "magnitude":
        magnitude = np.abs(spatial)
        target_amplitude = (
            magnitude * amplitude
        ).sum(axis=1, keepdims=True) / np.maximum(
            magnitude.sum(axis=1, keepdims=True), 1e-30
        )
    else:
        raise ValueError(f"unsupported amplitude mode {amplitude_mode}")
    return spatial * target_amplitude / np.maximum(
        amplitude, 1e-30
    )


def sufficient(
    coefficient: np.ndarray,
    data: dict[str, np.ndarray],
) -> dict[str, complex | float]:
    cross = np.einsum(
        "bi,bi->b",
        coefficient.conj(),
        data["cross"].astype(np.complex128),
    )
    energy = np.real(
        np.einsum(
            "bi,bij,bj->b",
            coefficient.conj(),
            data["gram"].astype(np.complex128),
            coefficient,
        )
    )
    return {
        "cross": complex(cross.sum()),
        "prediction_energy": float(energy.sum()),
        "truth_energy": float(
            data["truth_energy"].astype(np.float64).sum()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir", default="cache/complex_neighbor_gate_probe"
    )
    parser.add_argument(
        "--tune-seeds", default="1890,3716,962,1022"
    )
    parser.add_argument("--audit-seed", default="2262")
    parser.add_argument(
        "--external-names",
        default="testmatched,geomp0,geomp1,geomp2",
    )
    parser.add_argument(
        "--diagonal-loading-grid",
        default="0.03,0.1,0.3,1,3,10",
    )
    parser.add_argument(
        "--prior-power-grid", default="0.25,0.5,1"
    )
    parser.add_argument(
        "--blend-grid",
        default="0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.8,1",
    )
    parser.add_argument(
        "--amplitude-modes", default="base,magnitude"
    )
    parser.add_argument(
        "--out", default="docs/gram_kriging/result.json"
    )
    args = parser.parse_args()

    tune = [
        value for value in args.tune_seeds.split(",") if value
    ]
    external = [
        value for value in args.external_names.split(",") if value
    ]
    names = tune + [args.audit_seed] + external
    cache_dir = Path(args.cache_dir)
    data = {
        name: load_gate_data(cache_dir / f"{name}_val.npz")
        for name in names
    }
    loadings = [
        float(value)
        for value in args.diagonal_loading_grid.split(",")
        if value
    ]
    prior_powers = [
        float(value)
        for value in args.prior_power_grid.split(",")
        if value
    ]
    blends = [
        float(value)
        for value in args.blend_grid.split(",")
        if value
    ]
    amplitude_modes = [
        value for value in args.amplitude_modes.split(",") if value
    ]

    candidates: dict[str, dict[str, dict]] = {"baseline": {}}
    for name in names:
        candidates["baseline"][name] = sufficient(
            baseline_coefficients(data[name]), data[name]
        )
    for loading in loadings:
        for prior_power in prior_powers:
            for blend in blends:
                for amplitude_mode in amplitude_modes:
                    candidate_name = (
                        f"load{loading:g}_prior{prior_power:g}"
                        f"_blend{blend:g}_{amplitude_mode}"
                    )
                    candidates[candidate_name] = {}
                    for split_name in names:
                        coefficient = gram_kriging_coefficients(
                            data[split_name],
                            loading,
                            prior_power,
                            blend,
                            amplitude_mode,
                        )
                        candidates[candidate_name][split_name] = (
                            sufficient(
                                coefficient, data[split_name]
                            )
                        )

    ranked = []
    for candidate_name, rows in candidates.items():
        tune_cross = sum(rows[name]["cross"] for name in tune)
        tune_energy = sum(
            rows[name]["prediction_energy"] for name in tune
        )
        phase = float(np.angle(tune_cross))
        rotation = np.exp(-1j * phase)
        scale = max(
            float(np.real(tune_cross * rotation))
            / max(tune_energy, 1e-30),
            0.0,
        )
        scores = {}
        for name in names:
            row = rows[name]
            cross_real = float(
                np.real(row["cross"] * rotation)
            )
            nmse = (
                row["truth_energy"]
                + scale**2 * row["prediction_energy"]
                - 2.0 * scale * cross_real
            ) / max(row["truth_energy"], 1e-30)
            scores[name] = float(nmse)
        tune_nmse = [scores[name] for name in tune]
        ranked.append(
            {
                "name": candidate_name,
                "global_phase": phase,
                "scale": scale,
                "tune_mean_nmse": float(np.mean(tune_nmse)),
                "tune_median_nmse": float(np.median(tune_nmse)),
                "tune_worst_nmse": float(np.max(tune_nmse)),
                "audit_nmse": scores[args.audit_seed],
                "external_nmse": {
                    name: scores[name] for name in external
                },
                "scores": scores,
            }
        )
    baseline = next(
        row for row in ranked if row["name"] == "baseline"
    )
    for row in ranked:
        row["tune_delta_nmse"] = {
            name: row["scores"][name] - baseline["scores"][name]
            for name in tune
        }
        row["audit_delta_nmse"] = (
            row["audit_nmse"] - baseline["audit_nmse"]
        )
        row["external_delta_nmse"] = {
            name: row["external_nmse"][name]
            - baseline["external_nmse"][name]
            for name in external
        }
    ranked.sort(
        key=lambda row: (
            row["tune_mean_nmse"],
            row["tune_worst_nmse"],
            row["audit_nmse"],
        )
    )
    robust = [
        row
        for row in ranked
        if max(row["tune_delta_nmse"].values()) <= 0.0
    ]
    payload = {
        "selection_policy": (
            "ranked on four random tune splits; audit and label-free "
            "geometry-selected external panels are diagnostics only"
        ),
        "cache_dir": args.cache_dir,
        "tune_seeds": tune,
        "audit_seed": args.audit_seed,
        "external_names": external,
        "grid": {
            "diagonal_loading": loadings,
            "prior_power": prior_powers,
            "blend": blends,
            "amplitude_mode": amplitude_modes,
        },
        "baseline": baseline,
        "best_robust": robust[0] if robust else None,
        "ranked": ranked,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "baseline": baseline,
                "best": ranked[0],
                "best_robust": robust[0] if robust else None,
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"GRAM_KRIGING_DONE out={out}", flush=True)


if __name__ == "__main__":
    main()
