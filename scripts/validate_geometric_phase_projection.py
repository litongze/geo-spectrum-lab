#!/usr/bin/env python3
"""Combine geometric nearest-neighbor phase with moment-attention spectra."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from _bootstrap import ROOT  # noqa: F401
from score_holdout import reproduce_val_indices
from validate_moment_projection import (
    enforce_pas,
    enforce_pdp,
    stable_unit,
)
from wireless_twin.data.setup_config import load_setup
from wireless_twin.signal import pas_spectrum_phv, pdp_spectrum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", default="Round1_Map(2)")
    parser.add_argument(
        "--baseline-dir",
        default="docs/clean_noeps_panel_phv_geom_k16",
    )
    parser.add_argument(
        "--expert-dir", default="cache/moment_attention_selected"
    )
    parser.add_argument(
        "--phase-result", default="docs/geometric_phase/result.json"
    )
    parser.add_argument("--tune-seeds", default="1890,3716,962,1022")
    parser.add_argument("--audit-seed", type=int, default=2262)
    parser.add_argument("--pas-blend", type=float, default=0.8)
    parser.add_argument("--pdp-blend", type=float, default=0.3)
    parser.add_argument(
        "--phase-seed-epsilon",
        type=float,
        default=0.0,
        help=(
            "RMS-normalized dense baseline phase mixed into the sparse "
            "geometric nearest-neighbor reference"
        ),
    )
    parser.add_argument("--iteration-grid", default="1,2,3,5,8")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out",
        default="docs/geometric_phase_projection/result.json",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    datadir = Path(args.datadir)
    spec = load_setup(datadir / "Round1_Setup.json")
    positions = np.load(
        datadir / "Round1_Train_Pos.npy"
    ).astype(np.float64)
    channels = np.load(
        datadir / "Round1_Train_Channel.npy", mmap_mode="r"
    )
    tune_seeds = [
        int(value) for value in args.tune_seeds.split(",") if value
    ]
    seeds = tune_seeds + [args.audit_seed]
    iterations = [
        int(value) for value in args.iteration_grid.split(",") if value
    ]
    configurations = [("none", 0)]
    configurations.extend(
        (order, count)
        for order in ("pdp_pas", "pas_pdp")
        for count in iterations
    )
    phase_payload = json.loads(
        Path(args.phase_result).read_text(encoding="utf-8")
    )
    selected = phase_payload["selected"]
    k0 = float(selected["k0_rad_per_meter"])
    k1 = float(selected["k1_rad_per_meter_per_subcarrier"])
    global_phase = float(selected["global_phase"])
    bs = np.asarray(spec.bs_position, dtype=np.float64)
    radius = np.linalg.norm(positions - bs[None], axis=1)
    subcarrier = np.arange(spec.s, dtype=np.float64)
    subcarrier -= subcarrier.mean()
    all_idx = np.arange(len(positions), dtype=np.int64)
    sufficient = {
        f"{order}_i{count}": {}
        for order, count in configurations
    }

    for seed in seeds:
        val_idx = np.asarray(
            sorted(reproduce_val_indices(len(positions), 0.1, seed)),
            dtype=np.int64,
        )
        pool_idx = np.setdiff1d(all_idx, val_idx)
        _, local = cKDTree(positions[pool_idx, :2]).query(
            positions[val_idx, :2], k=1
        )
        neighbor_idx = pool_idx[np.asarray(local)]
        source = torch.as_tensor(
            np.array(channels[neighbor_idx], copy=True),
            dtype=torch.complex64,
            device=device,
        )
        truth = torch.as_tensor(
            np.array(channels[val_idx], copy=True),
            dtype=torch.complex64,
            device=device,
        )
        delta_radius = radius[val_idx] - radius[neighbor_idx]
        phase = (
            (
                k0 + k1 * subcarrier[None]
            )
            * delta_radius[:, None]
            + global_phase
        )
        source *= torch.as_tensor(
            np.exp(1j * phase),
            dtype=torch.complex64,
            device=device,
        )[:, None, None, :]

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
        if args.phase_seed_epsilon > 0:
            source_rms = source.abs().square().mean().sqrt()
            baseline_rms = baseline.abs().square().mean().sqrt()
            source = source + (
                args.phase_seed_epsilon
                * baseline
                * source_rms
                / baseline_rms.clamp_min(1e-30)
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
        truth_pas = stable_unit(pas_spectrum_phv(truth, spec))
        truth_pdp = stable_unit(pdp_spectrum(truth, spec))

        for order, count in configurations:
            name = f"{order}_i{count}"
            prediction = source.clone()
            for _ in range(count):
                if order == "pdp_pas":
                    prediction = enforce_pdp(prediction, target_pdp)
                    prediction = enforce_pas(
                        prediction, target_pas, spec
                    )
                elif order == "pas_pdp":
                    prediction = enforce_pas(
                        prediction, target_pas, spec
                    )
                    prediction = enforce_pdp(prediction, target_pdp)
            prediction_pas = stable_unit(
                pas_spectrum_phv(prediction, spec)
            )
            prediction_pdp = stable_unit(
                pdp_spectrum(prediction, spec)
            )
            sufficient[name][seed] = {
                "PAS": float(
                    (prediction_pas * truth_pas).sum(dim=-1).mean()
                ),
                "PDP": float(
                    (prediction_pdp * truth_pdp).sum(dim=-1).mean()
                ),
                "cross_real": float(
                    (prediction.conj() * truth).sum().real
                ),
                "prediction_energy": float(
                    prediction.abs().square().sum()
                ),
                "truth_energy": float(truth.abs().square().sum()),
                "coherence": float(
                    (prediction.conj() * truth).sum().abs()
                    / (
                        prediction.norm() * truth.norm()
                    ).clamp_min(1e-30)
                ),
            }
        print(f"[geo-project] split={seed} done", flush=True)
        del source, truth, baseline, target_pas, target_pdp
        torch.cuda.empty_cache()

    ranked = []
    for order, count in configurations:
        name = f"{order}_i{count}"
        pooled_cross = sum(
            sufficient[name][seed]["cross_real"]
            for seed in tune_seeds
        )
        pooled_energy = sum(
            sufficient[name][seed]["prediction_energy"]
            for seed in tune_seeds
        )
        scale = max(pooled_cross / max(pooled_energy, 1e-30), 0.0)
        scores = {}
        for seed in seeds:
            row = sufficient[name][seed]
            nmse = (
                row["truth_energy"]
                + scale**2 * row["prediction_energy"]
                - 2.0 * scale * row["cross_real"]
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
                "coherence": row["coherence"],
            }
        tune = [scores[str(seed)]["C"] for seed in tune_seeds]
        ranked.append(
            {
                "name": name,
                "order": order,
                "iterations": count,
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
            "phase and scale fitted on four tune splits; audit external"
        ),
        "phase": selected,
        "pas_blend": args.pas_blend,
        "pdp_blend": args.pdp_blend,
        "phase_seed_epsilon": args.phase_seed_epsilon,
        "tune_seeds": tune_seeds,
        "audit_seed": args.audit_seed,
        "ranked": ranked,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("[TOP]", json.dumps(ranked[:6], ensure_ascii=False), flush=True)
    print(f"GEOMETRIC_PHASE_PROJECTION_DONE out={out}", flush=True)


if __name__ == "__main__":
    main()
