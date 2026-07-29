#!/usr/bin/env python3
"""Sweep PAS/PDP projection order and strength on a clean validation panel."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from _bootstrap import ROOT  # noqa: F401
from rebuild_teammate_knn import knn_indices
from score_holdout import reproduce_val_indices
from wireless_twin.data.setup_config import load_setup
from wireless_twin.signal import (
    cosine_similarity_along_last,
    pas_spectrum,
    pas_spectrum_phv,
    pas_spectrum_pvh,
    pdp_spectrum,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", default="Round1_Map(2)")
    parser.add_argument("--cache-dir", default="cache/teammate_knn_hvp")
    parser.add_argument(
        "--pas-layout",
        choices=("hvp", "pvh", "phv"),
        default="hvp",
    )
    parser.add_argument("--tune-seeds", default="1890,3716,962,1022")
    parser.add_argument("--audit-seed", type=int, default=2262)
    parser.add_argument(
        "--testmatched",
        default="handoff_to_teammate_20260727/splits/"
        "test_matched_seed2026_val.npy",
    )
    parser.add_argument("--pas-k", type=int, default=2)
    parser.add_argument("--pas-distance-power", type=float, default=4.0)
    parser.add_argument("--pdp-k", type=int, default=4)
    parser.add_argument("--pdp-distance-power", type=float, default=5.0)
    parser.add_argument("--scale", type=float, default=0.02)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out", default="docs/reconstruction_sweep/result.json"
    )
    return parser.parse_args()


def configurations() -> list[dict]:
    result = [
        {"name": "pas_r1", "order": ["pas"], "pas": 1.0, "pdp": 1.0, "rank": 1},
        {"name": "pas_r2", "order": ["pas"], "pas": 1.0, "pdp": 1.0, "rank": 2},
        {"name": "pdp_r1", "order": ["pdp"], "pas": 1.0, "pdp": 1.0, "rank": 1},
        {"name": "pdp_r2", "order": ["pdp"], "pas": 1.0, "pdp": 1.0, "rank": 2},
    ]
    for strength in (0.25, 0.5, 0.75, 1.0):
        suffix = str(strength).replace(".", "")
        result.append(
            {
                "name": f"pas_pdp_d{suffix}",
                "order": ["pas", "pdp"],
                "pas": 1.0,
                "pdp": strength,
                "rank": 1,
            }
        )
        result.append(
            {
                "name": f"pdp_pas_p{suffix}",
                "order": ["pdp", "pas"],
                "pas": strength,
                "pdp": 1.0,
                "rank": 1,
            }
        )
    for strength in (0.1, 0.25, 0.5, 0.75, 1.0):
        suffix = str(strength).replace(".", "")
        result.append(
            {
                "name": f"pas_pdp_pas_d{suffix}",
                "order": ["pas", "pdp", "pas"],
                "pas": 1.0,
                "pdp": strength,
                "rank": 1,
            }
        )
        result.append(
            {
                "name": f"pdp_pas_pdp_p{suffix}",
                "order": ["pdp", "pas", "pdp"],
                "pas": strength,
                "pdp": 1.0,
                "rank": 1,
            }
        )
    for pas_strength in (0.25, 0.5, 0.75):
        for pdp_strength in (0.25, 0.5, 0.75):
            p = str(pas_strength).replace(".", "")
            d = str(pdp_strength).replace(".", "")
            result.append(
                {
                    "name": f"soft_pas_pdp_pas_p{p}_d{d}",
                    "order": ["pas", "pdp", "pas"],
                    "pas": pas_strength,
                    "pdp": pdp_strength,
                    "rank": 1,
                }
            )
    return result


def enforce_pas(
    h: torch.Tensor,
    target: torch.Tensor,
    spec,
    pas_layout: str,
    strength: float,
) -> torch.Tensor:
    batch = h.shape[0]
    if pas_layout == "hvp":
        hm = h.reshape(
            batch, spec.mh, spec.mv, spec.mp, spec.n, spec.s
        )
        angular = torch.fft.fft2(hm, dim=(1, 2))
        current = angular.abs().square().sum(dim=3)
        gain_shape = (
            batch, spec.mh, spec.mv, 1, spec.n, spec.s
        )
        transform_dims = (1, 2)
    else:
        spatial_shape = (
            (spec.mv, spec.mh)
            if pas_layout == "pvh"
            else (spec.mh, spec.mv)
        )
        hm = h.reshape(
            batch, spec.mp, *spatial_shape, spec.n, spec.s
        )
        angular = torch.fft.fft2(hm, dim=(2, 3))
        current = angular.abs().square().sum(dim=1)
        gain_shape = (
            batch, 1, *spatial_shape, spec.n, spec.s
        )
        transform_dims = (2, 3)
    current = current.reshape(
        batch, spec.mh * spec.mv, spec.n, spec.s
    )
    desired = target.permute(0, 3, 1, 2)
    if strength < 1.0:
        desired = current + strength * (desired - current)
    gain = (desired / current.clamp_min(1e-30)).sqrt().reshape(gain_shape)
    return torch.fft.ifft2(
        angular * gain, dim=transform_dims
    ).reshape(batch, spec.m, spec.n, spec.s)


def enforce_pdp(
    h: torch.Tensor,
    target: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    delay = torch.fft.ifft(h, dim=-1)
    current = delay.abs().square()
    desired = target
    if strength < 1.0:
        desired = current + strength * (desired - current)
    gain = (desired / current.clamp_min(1e-30)).sqrt()
    return torch.fft.fft(delay * gain, dim=-1)


def score(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    spec,
    pas_layout: str,
) -> dict[str, float]:
    pas_transform = {
        "hvp": pas_spectrum,
        "pvh": pas_spectrum_pvh,
        "phv": pas_spectrum_phv,
    }[pas_layout]
    c1 = float(
        cosine_similarity_along_last(
            pas_transform(prediction, spec),
            pas_transform(truth, spec),
        )
    )
    c2 = float(
        cosine_similarity_along_last(
            pdp_spectrum(prediction, spec), pdp_spectrum(truth, spec)
        )
    )
    c3 = float(
        (prediction - truth).abs().square().sum()
        / truth.abs().square().sum()
    )
    combined = (
        spec.metric_weights[0] * c1
        + spec.metric_weights[1] * c2
        + spec.metric_weights[2] / (1.0 + c3)
    )
    return {"C1_PAS": c1, "C2_PDP": c2, "C3_NMSE": c3, "C": combined}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    tune_seeds = [
        int(value) for value in args.tune_seeds.split(",") if value.strip()
    ]
    datadir = Path(args.datadir)
    setup_path = next(datadir.glob("Round*_Setup.json"))
    tag = setup_path.name.removesuffix("_Setup.json")
    spec = load_setup(setup_path)
    train_pos = np.load(datadir / f"{tag}_Train_Pos.npy").astype(np.float32)
    channels = np.load(
        datadir / f"{tag}_Train_Channel.npy", mmap_mode="r"
    )
    cache_dir = Path(args.cache_dir)
    train_pas = np.load(
        cache_dir / f"train_pas_{args.pas_layout}.npy", mmap_mode="r"
    )
    train_pdp = np.load(cache_dir / "train_pdp.npy", mmap_mode="r")
    all_idx = np.arange(len(train_pos), dtype=np.int64)
    splits = {
        str(seed): np.asarray(
            sorted(reproduce_val_indices(len(train_pos), 0.1, seed)),
            dtype=np.int64,
        )
        for seed in tune_seeds + [args.audit_seed]
    }
    splits["testmatched"] = np.load(args.testmatched).astype(np.int64)
    configs = configurations()
    rows = {
        config["name"]: {**config, "scores": {}} for config in configs
    }

    for split_name, val_idx in splits.items():
        print(f"[split] {split_name}", flush=True)
        pool_idx = np.setdiff1d(all_idx, val_idx, assume_unique=False)
        pas_neighbors, pas_weights = knn_indices(
            train_pos,
            train_pos[val_idx],
            pool_idx,
            args.pas_k,
            args.pas_distance_power,
            "inverse",
            1.0,
            0.0,
        )
        pdp_neighbors, pdp_weights = knn_indices(
            train_pos,
            train_pos[val_idx],
            pool_idx,
            args.pdp_k,
            args.pdp_distance_power,
            "inverse",
            1.0,
            0.0,
        )
        target_pas = np.stack(
            [
                np.tensordot(
                    pas_weights[row],
                    np.asarray(train_pas[pas_neighbors[row]]),
                    axes=(0, 0),
                )
                for row in range(len(val_idx))
            ]
        ).astype(np.float32)
        target_pdp = np.stack(
            [
                np.tensordot(
                    pdp_weights[row],
                    np.asarray(train_pdp[pdp_neighbors[row]]),
                    axes=(0, 0),
                )
                for row in range(len(val_idx))
            ]
        ).astype(np.float32)
        pas_target = torch.from_numpy(target_pas).to(device)
        pdp_target = torch.from_numpy(target_pdp).to(device)
        truth = torch.from_numpy(
            np.asarray(channels[val_idx]).copy()
        ).to(device)
        sources = {
            rank: torch.from_numpy(
                np.asarray(channels[pas_neighbors[:, rank - 1]]).copy()
            ).to(device)
            for rank in {config["rank"] for config in configs}
        }
        del target_pas, target_pdp

        for config in configs:
            h = sources[config["rank"]]
            for domain in config["order"]:
                if domain == "pas":
                    h = enforce_pas(
                        h,
                        pas_target,
                        spec,
                        args.pas_layout,
                        config["pas"],
                    )
                else:
                    h = enforce_pdp(h, pdp_target, config["pdp"])
            metrics = score(
                h * args.scale, truth, spec, args.pas_layout
            )
            rows[config["name"]]["scores"][split_name] = metrics
            print(
                f"[score] {split_name} {config['name']} "
                f"C={metrics['C']:.9f}",
                flush=True,
            )
            del h
        del pas_target, pdp_target, truth, sources
        torch.cuda.empty_cache()

    ranked = []
    for row in rows.values():
        tune = [row["scores"][str(seed)]["C"] for seed in tune_seeds]
        ranked.append(
            {
                **row,
                "tune_median": float(np.median(tune)),
                "tune_mean": float(np.mean(tune)),
                "tune_worst": float(np.min(tune)),
                "audit": row["scores"][str(args.audit_seed)]["C"],
                "testmatched": row["scores"]["testmatched"]["C"],
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
            "ranked only on tune seeds; audit and testmatched are external"
        ),
        "tune_seeds": tune_seeds,
        "audit_seed": args.audit_seed,
        "pas_layout": args.pas_layout,
        "pas_k": args.pas_k,
        "pas_distance_power": args.pas_distance_power,
        "pdp_k": args.pdp_k,
        "pdp_distance_power": args.pdp_distance_power,
        "ranked": ranked,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("[TOP]", json.dumps(ranked[:8]), flush=True)
    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
