#!/usr/bin/env python3
"""Validate channel reconstruction from selected moment-attention spectra."""
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
    maximum = value.amax(dim=-1, keepdim=True)
    scaled = torch.where(
        maximum > 0,
        value / maximum.clamp_min(torch.finfo(value.dtype).tiny),
        torch.zeros_like(value),
    )
    return scaled / scaled.norm(dim=-1, keepdim=True).clamp_min(
        torch.finfo(value.dtype).tiny
    )


def enforce_pas(
    channel: torch.Tensor, target: torch.Tensor, spec
) -> torch.Tensor:
    return enforce_pas_layout(channel, target, spec, "phv")


def enforce_pas_layout(
    channel: torch.Tensor,
    target: torch.Tensor,
    spec,
    layout: str,
) -> torch.Tensor:
    """Enforce a PAS target under one explicit BS-array flattening."""
    batch = channel.shape[0]
    if layout == "phv":
        shaped = channel.reshape(
            batch,
            spec.mp,
            spec.mh,
            spec.mv,
            spec.n,
            spec.s,
        )
        fft_dims = (2, 3)
        polarization_dim = 1
        gain_shape = (
            batch,
            1,
            spec.mh,
            spec.mv,
            spec.n,
            spec.s,
        )
    elif layout == "hvp":
        shaped = channel.reshape(
            batch,
            spec.mh,
            spec.mv,
            spec.mp,
            spec.n,
            spec.s,
        )
        fft_dims = (1, 2)
        polarization_dim = 3
        gain_shape = (
            batch,
            spec.mh,
            spec.mv,
            1,
            spec.n,
            spec.s,
        )
    elif layout == "pvh":
        shaped = channel.reshape(
            batch,
            spec.mp,
            spec.mv,
            spec.mh,
            spec.n,
            spec.s,
        )
        fft_dims = (2, 3)
        polarization_dim = 1
        gain_shape = (
            batch,
            1,
            spec.mv,
            spec.mh,
            spec.n,
            spec.s,
        )
    else:
        raise ValueError(f"unsupported PAS layout {layout}")

    angular = torch.fft.fft2(
        shaped, dim=fft_dims, norm="ortho"
    )
    current = angular.abs().square().sum(
        dim=polarization_dim
    ).reshape(batch, spec.mh * spec.mv, spec.n, spec.s)
    target_first = target.permute(0, 3, 1, 2)
    desired = target_first * current.norm(
        dim=1, keepdim=True
    ).clamp_min(torch.finfo(torch.float32).tiny)
    gain = torch.sqrt(
        desired.clamp_min(0.0) / current.clamp_min(1e-38)
    ).reshape(gain_shape)
    return torch.fft.ifft2(
        angular * gain, dim=fft_dims, norm="ortho"
    ).reshape(batch, spec.m, spec.n, spec.s)


def enforce_pdp(
    channel: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    delay = torch.fft.ifft(channel, dim=-1, norm="ortho")
    current = delay.abs().square()
    desired = target * current.norm(
        dim=-1, keepdim=True
    ).clamp_min(torch.finfo(torch.float32).tiny)
    phase = delay / delay.abs().clamp_min(1e-30)
    return torch.fft.fft(
        torch.sqrt(desired.clamp_min(0.0)) * phase,
        dim=-1,
        norm="ortho",
    )


def metrics(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    spec,
) -> dict[str, float]:
    prediction_pas = stable_unit(pas_spectrum_phv(prediction, spec))
    truth_pas = stable_unit(pas_spectrum_phv(truth, spec))
    prediction_pdp = stable_unit(pdp_spectrum(prediction, spec))
    truth_pdp = stable_unit(pdp_spectrum(truth, spec))
    pas = float((prediction_pas * truth_pas).sum(dim=-1).mean())
    pdp = float((prediction_pdp * truth_pdp).sum(dim=-1).mean())
    nmse = float(
        (prediction - truth).abs().square().sum()
        / truth.abs().square().sum().clamp_min(1e-30)
    )
    score = (
        spec.metric_weights[0] * pas
        + spec.metric_weights[1] * pdp
        + spec.metric_weights[2] / (1.0 + nmse)
    )
    return {
        "PAS": pas,
        "PDP": pdp,
        "NMSE": nmse,
        "C": score,
    }


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
    parser.add_argument("--tune-seeds", default="1890,3716,962,1022")
    parser.add_argument("--audit-seed", type=int, default=2262)
    parser.add_argument("--pas-blend", type=float, default=0.8)
    parser.add_argument("--pdp-blend", type=float, default=0.3)
    parser.add_argument("--iteration-grid", default="1,2,3,5,8")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out", default="docs/moment_projection/result.json"
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
    iteration_grid = [
        int(value) for value in args.iteration_grid.split(",") if value
    ]
    configurations = [
        (order, iterations)
        for order in ("pas_pdp", "pdp_pas")
        for iterations in iteration_grid
    ]
    split_scores: dict[str, dict[str, dict[str, float]]] = {}

    for seed in seeds:
        val_idx = np.asarray(
            sorted(reproduce_val_indices(len(positions), 0.1, seed)),
            dtype=np.int64,
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
        truth = torch.as_tensor(
            np.array(channels[val_idx], copy=True),
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
        direct = {
            "PAS": float(
                (
                    target_pas
                    * stable_unit(pas_spectrum_phv(truth, spec))
                )
                .sum(dim=-1)
                .mean()
            ),
            "PDP": float(
                (
                    target_pdp
                    * stable_unit(pdp_spectrum(truth, spec))
                )
                .sum(dim=-1)
                .mean()
            ),
        }
        baseline_metrics = metrics(baseline, truth, spec)
        split_scores[str(seed)] = {
            "baseline": baseline_metrics,
            "direct_target": direct,
        }
        for order, iterations in configurations:
            prediction = baseline.clone()
            for _ in range(iterations):
                if order == "pas_pdp":
                    prediction = enforce_pas(
                        prediction, target_pas, spec
                    )
                    prediction = enforce_pdp(prediction, target_pdp)
                else:
                    prediction = enforce_pdp(prediction, target_pdp)
                    prediction = enforce_pas(
                        prediction, target_pas, spec
                    )
            # Preserve the baseline's global scale so NMSE comparisons remain
            # attributable to spectral reconstruction rather than scale drift.
            prediction *= (
                baseline.abs().square().mean().sqrt()
                / prediction.abs().square().mean().sqrt().clamp_min(1e-30)
            )
            name = f"{order}_i{iterations}"
            split_scores[str(seed)][name] = metrics(
                prediction, truth, spec
            )
        print(
            f"[moment-project] split={seed} "
            f"base={baseline_metrics['C']:.6f} "
            f"target=({direct['PAS']:.6f},{direct['PDP']:.6f})",
            flush=True,
        )
        del baseline, truth, target_pas, target_pdp
        torch.cuda.empty_cache()

    ranked = []
    for order, iterations in configurations:
        name = f"{order}_i{iterations}"
        tune = [
            split_scores[str(seed)][name]["C"]
            for seed in tune_seeds
        ]
        ranked.append(
            {
                "name": name,
                "order": order,
                "iterations": iterations,
                "tune_median": float(np.median(tune)),
                "tune_mean": float(np.mean(tune)),
                "tune_worst": float(np.min(tune)),
                "audit": split_scores[str(args.audit_seed)][name]["C"],
                "scores": {
                    str(seed): split_scores[str(seed)][name]
                    for seed in seeds
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
        "selection_policy": "four tune splits only",
        "pas_blend": args.pas_blend,
        "pdp_blend": args.pdp_blend,
        "tune_seeds": tune_seeds,
        "audit_seed": args.audit_seed,
        "ranked": ranked,
        "splits": split_scores,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("[TOP]", json.dumps(ranked[:5], ensure_ascii=False), flush=True)
    print(f"MOMENT_PROJECTION_DONE out={out}", flush=True)


if __name__ == "__main__":
    main()
