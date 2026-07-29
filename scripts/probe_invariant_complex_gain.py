#!/usr/bin/env python3
"""Probe spectrum-invariant complex gains for NMSE calibration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from _bootstrap import ROOT  # noqa: F401
from score_holdout import reproduce_val_indices
from wireless_twin.data.setup_config import load_setup
from wireless_twin.signal import pas_spectrum_phv, pdp_spectrum


def stable_unit(value: torch.Tensor) -> torch.Tensor:
    return value / value.norm(dim=-1, keepdim=True).clamp_min(
        torch.finfo(value.dtype).tiny
    )


def score(
    prediction: torch.Tensor, truth: torch.Tensor, spec
) -> dict[str, float]:
    pas = float(
        (
            stable_unit(pas_spectrum_phv(prediction, spec))
            * stable_unit(pas_spectrum_phv(truth, spec))
        )
        .sum(dim=-1)
        .mean()
    )
    pdp = float(
        (
            stable_unit(pdp_spectrum(prediction, spec))
            * stable_unit(pdp_spectrum(truth, spec))
        )
        .sum(dim=-1)
        .mean()
    )
    nmse = float(
        (prediction - truth).abs().square().sum()
        / truth.abs().square().sum().clamp_min(1e-30)
    )
    combined = (
        spec.metric_weights[0] * pas
        + spec.metric_weights[1] * pdp
        + spec.metric_weights[2] / (1.0 + nmse)
    )
    return {"PAS": pas, "PDP": pdp, "NMSE": nmse, "C": combined}


def sufficient_statistics(
    prediction: torch.Tensor, truth: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cross = (prediction.conj() * truth).sum(dim=(0, 1, 3))
    energy = prediction.abs().square().sum(dim=(0, 1, 3))
    return cross.to(torch.complex128), energy.to(torch.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", default="Round1_Map(2)")
    parser.add_argument(
        "--prediction-dir",
        default="docs/clean_noeps_panel_phv_geom_k16",
    )
    parser.add_argument("--tune-seeds", default="1890,3716,962,1022")
    parser.add_argument("--audit-seed", type=int, default=2262)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out", default="docs/invariant_complex_gain/result.json"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    datadir = Path(args.datadir)
    spec = load_setup(datadir / "Round1_Setup.json")
    positions = np.load(
        datadir / "Round1_Train_Pos.npy", mmap_mode="r"
    )
    channels = np.load(
        datadir / "Round1_Train_Channel.npy", mmap_mode="r"
    )
    tune_seeds = [
        int(value) for value in args.tune_seeds.split(",") if value
    ]
    seeds = tune_seeds + [args.audit_seed]
    tensors = {}
    statistics = {}

    for seed in seeds:
        val_idx = np.asarray(
            sorted(reproduce_val_indices(len(positions), 0.1, seed)),
            dtype=np.int64,
        )
        prediction = torch.as_tensor(
            np.array(
                np.load(
                    Path(args.prediction_dir)
                    / f"split_s{seed}_prediction.npy",
                    mmap_mode="r",
                ),
                copy=True,
            ),
            dtype=torch.complex64,
            device=device,
        )
        truth = torch.as_tensor(
            np.array(channels[val_idx], copy=True),
            dtype=torch.complex64,
            device=device,
        )
        cross, energy = sufficient_statistics(prediction, truth)
        tensors[seed] = (prediction, truth)
        statistics[seed] = (cross, energy)

    pooled_cross = sum(
        statistics[seed][0] for seed in tune_seeds
    )
    pooled_energy = sum(
        statistics[seed][1] for seed in tune_seeds
    )
    per_n_gain = pooled_cross / pooled_energy.clamp_min(1e-30)
    global_gain = (
        pooled_cross.sum() / pooled_energy.sum().clamp_min(1e-30)
    )
    real_per_n_gain = (
        pooled_cross.real / pooled_energy.clamp_min(1e-30)
    ).clamp_min(0.0)
    real_global_gain = (
        pooled_cross.real.sum() / pooled_energy.sum().clamp_min(1e-30)
    ).clamp_min(0.0)
    unit_per_n_phase = per_n_gain / per_n_gain.abs().clamp_min(1e-30)
    unit_global_phase = global_gain / global_gain.abs().clamp_min(1e-30)

    calibrations = {
        "baseline": torch.ones(
            spec.n, dtype=torch.complex128, device=device
        ),
        "complex_global": unit_global_phase.repeat(spec.n)
        * global_gain.abs(),
        "complex_per_n": per_n_gain,
        "real_global": real_global_gain.to(torch.complex128).repeat(
            spec.n
        ),
        "real_per_n": real_per_n_gain.to(torch.complex128),
        "phase_global_only": unit_global_phase.repeat(spec.n),
        "phase_per_n_only": unit_per_n_phase,
    }
    rows = {}
    for name, gain in calibrations.items():
        split_metrics = {}
        for seed in seeds:
            prediction, truth = tensors[seed]
            calibrated = prediction * gain.to(
                torch.complex64
            )[None, None, :, None]
            split_metrics[str(seed)] = score(
                calibrated, truth, spec
            )
        tune = [
            split_metrics[str(seed)]["C"] for seed in tune_seeds
        ]
        rows[name] = {
            "gain_real": gain.real.detach().cpu().tolist(),
            "gain_imag": gain.imag.detach().cpu().tolist(),
            "gain_abs": gain.abs().detach().cpu().tolist(),
            "gain_phase": gain.angle().detach().cpu().tolist(),
            "tune_median": float(np.median(tune)),
            "tune_mean": float(np.mean(tune)),
            "tune_worst": float(np.min(tune)),
            "audit": split_metrics[str(args.audit_seed)]["C"],
            "scores": split_metrics,
        }

    oracle = {}
    for seed in seeds:
        prediction, truth = tensors[seed]
        cross, energy = statistics[seed]
        gain_n = cross / energy.clamp_min(1e-30)
        gain_global = cross.sum() / energy.sum().clamp_min(1e-30)
        oracle[str(seed)] = {
            "complex_global": score(
                prediction
                * gain_global.to(torch.complex64),
                truth,
                spec,
            ),
            "complex_per_n": score(
                prediction
                * gain_n.to(torch.complex64)[None, None, :, None],
                truth,
                spec,
            ),
            "global_gain_abs": float(gain_global.abs()),
            "global_gain_phase": float(gain_global.angle()),
            "per_n_gain_abs": gain_n.abs().cpu().tolist(),
            "per_n_gain_phase": gain_n.angle().cpu().tolist(),
        }

    payload = {
        "selection_policy": "calibration fitted on four tune splits",
        "tune_seeds": tune_seeds,
        "audit_seed": args.audit_seed,
        "calibrations": rows,
        "split_oracle_diagnostics": oracle,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary = {
        name: {
            "tune_median": row["tune_median"],
            "audit": row["audit"],
            "audit_nmse": row["scores"][str(args.audit_seed)]["NMSE"],
        }
        for name, row in rows.items()
    }
    print(json.dumps(summary, indent=2), flush=True)
    print(
        "[oracle]",
        json.dumps(oracle, ensure_ascii=False),
        flush=True,
    )
    print(f"INVARIANT_GAIN_DONE out={out}", flush=True)


if __name__ == "__main__":
    main()
