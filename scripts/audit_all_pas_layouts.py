#!/usr/bin/env python3
"""Audit every BS-array axis order on saved clean validation predictions."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from _bootstrap import ROOT  # noqa: F401
from score_holdout import reproduce_val_indices
from wireless_twin.data.setup_config import load_setup
from wireless_twin.signal import pdp_spectrum


LAYOUTS = tuple(
    "".join(order)
    for order in itertools.permutations(("h", "v", "p"))
)


def stable_cosine(
    prediction: torch.Tensor, truth: torch.Tensor
) -> torch.Tensor:
    numerator = (prediction * truth).sum(dim=-1)
    denominator = prediction.norm(dim=-1) * truth.norm(dim=-1)
    return (
        numerator
        / denominator.clamp_min(torch.finfo(denominator.dtype).tiny)
    ).clamp(-1.0, 1.0)


def pas_spectrum_layout(
    channel: torch.Tensor, spec, layout: str
) -> torch.Tensor:
    if layout not in LAYOUTS:
        raise ValueError(f"unknown PAS layout {layout}")
    size = {"h": spec.mh, "v": spec.mv, "p": spec.mp}
    batch = channel.shape[0]
    shaped = channel.reshape(
        batch,
        size[layout[0]],
        size[layout[1]],
        size[layout[2]],
        spec.n,
        spec.s,
    )
    axis = {
        name: 1 + layout.index(name) for name in ("h", "v", "p")
    }
    canonical = shaped.permute(
        0, axis["h"], axis["v"], axis["p"], 4, 5
    )
    power = torch.fft.fft2(
        canonical, dim=(1, 2)
    ).abs().square().sum(dim=3)
    return power.reshape(
        batch, spec.mh * spec.mv, spec.n, spec.s
    ).permute(0, 2, 3, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", default="Round1_Map(2)")
    parser.add_argument(
        "--prediction-template",
        default=(
            "cache/geometric_phase_neighbors_projection_transport/"
            "split_s{seed}_prediction.npy"
        ),
    )
    parser.add_argument(
        "--tune-seeds", default="1890,3716,962,1022"
    )
    parser.add_argument("--audit-seed", default="2262")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out", default="docs/metric_layout_audit/all_six.json"
    )
    args = parser.parse_args()

    tune = [item for item in args.tune_seeds.split(",") if item]
    names = [*tune, args.audit_seed]
    datadir = Path(args.datadir)
    spec = load_setup(datadir / "Round1_Setup.json")
    channels = np.load(
        datadir / "Round1_Train_Channel.npy", mmap_mode="r"
    )
    device = torch.device(args.device)
    rows = {}
    with torch.inference_mode():
        for name in names:
            val_idx = np.asarray(
                sorted(
                    reproduce_val_indices(
                        len(channels), 0.1, int(name)
                    )
                ),
                dtype=np.int64,
            )
            prediction = np.load(
                args.prediction_template.format(seed=name),
                mmap_mode="r",
            )
            if len(prediction) != len(val_idx):
                raise ValueError(
                    f"prediction length mismatch for split {name}"
                )
            totals = {
                layout: 0.0 for layout in LAYOUTS
            }
            pdp_total = 0.0
            error_energy = 0.0
            truth_energy = 0.0
            count = 0
            for start in range(0, len(val_idx), args.batch_size):
                stop = min(start + args.batch_size, len(val_idx))
                predicted = torch.as_tensor(
                    np.array(prediction[start:stop], copy=True),
                    dtype=torch.complex64,
                    device=device,
                )
                truth = torch.as_tensor(
                    np.array(channels[val_idx[start:stop]], copy=True),
                    dtype=torch.complex64,
                    device=device,
                )
                for layout in LAYOUTS:
                    totals[layout] += float(
                        stable_cosine(
                            pas_spectrum_layout(
                                predicted, spec, layout
                            ),
                            pas_spectrum_layout(truth, spec, layout),
                        ).sum()
                    )
                pdp_total += float(
                    stable_cosine(
                        pdp_spectrum(predicted, spec),
                        pdp_spectrum(truth, spec),
                    ).sum()
                )
                error_energy += float(
                    (predicted - truth).abs().square().sum()
                )
                truth_energy += float(
                    truth.abs().square().sum()
                )
                count += stop - start
            pdp = pdp_total / (
                count * spec.m * spec.n
            )
            nmse = error_energy / max(truth_energy, 1e-30)
            split_rows = {}
            for layout in LAYOUTS:
                pas = totals[layout] / (
                    count * spec.n * spec.s
                )
                split_rows[layout] = {
                    "PAS": pas,
                    "PDP": pdp,
                    "NMSE": nmse,
                    "C": (
                        0.4 * pas
                        + 0.4 * pdp
                        + 0.2 / (1.0 + nmse)
                    ),
                }
            rows[name] = split_rows
            print(
                f"[layout] split={name} "
                + " ".join(
                    f"{layout}={split_rows[layout]['C']:.6f}"
                    for layout in LAYOUTS
                ),
                flush=True,
            )

    ranked = []
    for layout in LAYOUTS:
        tune_scores = [
            rows[name][layout]["C"] for name in tune
        ]
        ranked.append(
            {
                "layout": layout,
                "tune_mean": float(np.mean(tune_scores)),
                "tune_median": float(np.median(tune_scores)),
                "tune_worst": float(np.min(tune_scores)),
                "audit": rows[args.audit_seed][layout]["C"],
                "scores": {
                    name: rows[name][layout] for name in names
                },
            }
        )
    ranked.sort(
        key=lambda row: (
            row["tune_mean"],
            row["tune_worst"],
        ),
        reverse=True,
    )
    payload = {
        "selection_policy": (
            "layout audit only; no parameters are selected from the "
            "audit split"
        ),
        "prediction_template": args.prediction_template,
        "tune_seeds": tune,
        "audit_seed": args.audit_seed,
        "ranked": ranked,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(
        "ALL_LAYOUTS_DONE "
        f"best={ranked[0]['layout']} "
        f"tune={ranked[0]['tune_mean']:.6f} "
        f"audit={ranked[0]['audit']:.6f} "
        f"out={out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
