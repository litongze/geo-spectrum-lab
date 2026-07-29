#!/usr/bin/env python3
"""Compare saved clean predictions under every plausible PAS layout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from _bootstrap import ROOT  # noqa: F401
from wireless_twin.data.setup_config import load_setup
from wireless_twin.signal import (
    cosine_similarity_along_last,
    pas_spectrum,
    pas_spectrum_phv,
    pas_spectrum_pvh,
    pdp_spectrum,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", default="Round1_Map(2)")
    parser.add_argument("--panel", default="1890,3716,962,1022,2262")
    parser.add_argument(
        "--team-pattern", default="docs/teammate_knn/s{seed}_k456.npy"
    )
    parser.add_argument(
        "--candidate-pattern",
        default="docs/clean_noeps_panel_phv_geom_k16/"
        "split_s{seed}_prediction.npy",
    )
    parser.add_argument(
        "--indices-pattern",
        default="docs/teammate_knn/s{seed}_indices.npy",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out", default="docs/metric_layout_audit/result.json"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    spec = load_setup(Path(args.datadir) / "Round1_Setup.json")
    truth_all = np.load(
        Path(args.datadir) / "Round1_Train_Channel.npy", mmap_mode="r"
    )
    layouts = {
        "hvp": pas_spectrum,
        "pvh": pas_spectrum_pvh,
        "phv": pas_spectrum_phv,
    }
    result: dict[str, dict] = {}
    for seed in (
        int(value) for value in args.panel.split(",") if value.strip()
    ):
        indices = np.load(args.indices_pattern.format(seed=seed)).astype(
            np.int64
        )
        truth = torch.as_tensor(
            np.array(truth_all[indices], copy=True),
            dtype=torch.complex64,
            device=device,
        )
        split = {}
        predictions = {}
        for name, pattern in (
            ("team_k456", args.team_pattern),
            ("candidate", args.candidate_pattern),
        ):
            prediction = torch.as_tensor(
                np.array(
                    np.load(pattern.format(seed=seed), mmap_mode="r"),
                    copy=True,
                ),
                dtype=torch.complex64,
                device=device,
            )
            predictions[name] = prediction
            row = {
                layout: float(
                    cosine_similarity_along_last(
                        transform(prediction, spec),
                        transform(truth, spec),
                    )
                )
                for layout, transform in layouts.items()
            }
            row["pdp"] = float(
                cosine_similarity_along_last(
                    pdp_spectrum(prediction, spec),
                    pdp_spectrum(truth, spec),
                )
            )
            row["nmse"] = float(
                (prediction - truth).abs().square().sum()
                / truth.abs().square().sum()
            )
            split[name] = row
        split["candidate_minus_team"] = {
            metric: split["candidate"][metric] - split["team_k456"][metric]
            for metric in split["candidate"]
        }
        team = predictions["team_k456"]
        candidate = predictions["candidate"]
        candidate = candidate * torch.sqrt(
            team.abs().square().mean()
            / candidate.abs().square().mean().clamp_min(1e-30)
        )
        blends = []
        for beta in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
            prediction = (1.0 - beta) * team + beta * candidate
            row = {
                "candidate_beta": beta,
                **{
                    layout: float(
                        cosine_similarity_along_last(
                            transform(prediction, spec),
                            transform(truth, spec),
                        )
                    )
                    for layout, transform in layouts.items()
                },
                "pdp": float(
                    cosine_similarity_along_last(
                        pdp_spectrum(prediction, spec),
                        pdp_spectrum(truth, spec),
                    )
                ),
                "nmse": float(
                    (prediction - truth).abs().square().sum()
                    / truth.abs().square().sum()
                ),
            }
            blends.append(row)
        split["matched_channel_blend"] = blends
        result[str(seed)] = split
        del truth, predictions, team, candidate, prediction
        torch.cuda.empty_cache()
        print(f"[layout] split={seed} {split}", flush=True)

    summary = {}
    for metric in ("hvp", "pvh", "phv", "pdp", "nmse"):
        values = [
            result[seed]["candidate_minus_team"][metric]
            for seed in result
        ]
        summary[metric] = {
            "median_delta": float(np.median(values)),
            "mean_delta": float(np.mean(values)),
            "worst_delta": float(np.min(values)),
            "values": values,
        }
    payload = {"splits": result, "delta_summary": summary}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"METRIC_LAYOUT_AUDIT_DONE out={out}", flush=True)


if __name__ == "__main__":
    main()
