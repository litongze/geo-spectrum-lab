#!/usr/bin/env python3
"""Train a clean per-slice gate for radial-neighbor PDP correction."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from _bootstrap import ROOT  # noqa: F401
from score_holdout import reproduce_val_indices
from sweep_rayprofile_spectrum_knn import local_radial_profiles
from validate_moment_projection import stable_unit
from wireless_twin.data.map_loader import load_point_cloud
from wireless_twin.data.setup_config import load_setup
from wireless_twin.models.raytrace2 import build_heightmap
from wireless_twin.signal import pdp_spectrum


FEATURE_NAMES = (
    "baseline_target_cosine",
    "baseline_nearest_cosine",
    "target_nearest_cosine",
    "target_neighbor_consensus",
    "target_neighbor_spread",
    "baseline_peak",
    "target_peak",
    "peak_shift",
    "baseline_entropy",
    "target_entropy",
    "centroid_shift",
    "log_nearest_distance",
    "log_mean_distance",
    "nearest_profile_distance",
    "mean_profile_distance",
    "maximum_weight",
    "weight_entropy",
    "nearest_global_agreement",
)


@dataclass
class FoldData:
    name: str
    val_idx: np.ndarray
    slices_per_query: int
    features: np.ndarray
    truth_baseline: np.ndarray
    truth_target: np.ndarray
    baseline_target: np.ndarray


class SliceGate(nn.Module):
    def __init__(self, feature_dim: int, hidden: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(
            self.network[-1].bias,
            float(np.log(0.35 / (1.0 - 0.35))),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(features))[..., 0]


def radial_neighbors(
    query_idx: np.ndarray,
    pool_idx: np.ndarray,
    positions: np.ndarray,
    profiles: np.ndarray,
    profile_lambda: float,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xy_delta = (
        positions[query_idx, None, :2]
        - positions[pool_idx][None, :, :2]
    )
    spatial_distance2 = np.square(xy_delta).sum(axis=-1)
    profile_scale = profiles[pool_idx].std(axis=0).clip(0.05)
    profile_delta = (
        profiles[query_idx, None] - profiles[pool_idx][None]
    ) / profile_scale
    profile_distance2 = np.square(profile_delta).mean(axis=-1)
    effective_distance2 = (
        spatial_distance2
        + profile_lambda**2 * profile_distance2
    )
    local = np.argpartition(
        effective_distance2, kth=k - 1, axis=1
    )[:, :k]
    selected_effective = np.take_along_axis(
        effective_distance2, local, axis=1
    )
    order = np.argsort(selected_effective, axis=1)
    local = np.take_along_axis(local, order, axis=1)
    spatial_distance = np.sqrt(
        np.maximum(
            np.take_along_axis(spatial_distance2, local, axis=1),
            1e-6,
        )
    ).astype(np.float32)
    profile_distance = np.sqrt(
        np.maximum(
            np.take_along_axis(profile_distance2, local, axis=1),
            0.0,
        )
    ).astype(np.float32)
    return pool_idx[local], spatial_distance, profile_distance


def distribution_features(
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probability = value / value.sum(
        dim=-1, keepdim=True
    ).clamp_min(torch.finfo(value.dtype).tiny)
    axis = torch.linspace(
        0.0,
        1.0,
        value.shape[-1],
        dtype=value.dtype,
        device=value.device,
    )
    peak = value.argmax(dim=-1).to(value.dtype) / max(
        value.shape[-1] - 1, 1
    )
    entropy = -(
        probability
        * probability.clamp_min(1e-30).log()
    ).sum(dim=-1) / np.log(value.shape[-1])
    centroid = (probability * axis).sum(dim=-1)
    return peak, entropy, centroid


def expand_query_feature(
    value: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    return value[:, None, None].expand_as(reference)


def prepare_fold(
    name: str,
    val_idx: np.ndarray,
    positions: np.ndarray,
    profiles: np.ndarray,
    spectra: torch.Tensor,
    prediction_template: str,
    profile_lambda: float,
    k: int,
    distance_power: float,
    anchor_strength: float,
    spec,
    batch_size: int,
    device: torch.device,
) -> FoldData:
    all_idx = np.arange(len(positions), dtype=np.int64)
    pool_idx = np.setdiff1d(all_idx, val_idx)
    neighbors, distance, profile_distance = radial_neighbors(
        val_idx,
        pool_idx,
        positions,
        profiles,
        profile_lambda,
        k,
    )
    prediction = np.load(
        prediction_template.format(seed=name), mmap_mode="r"
    )
    feature_rows = []
    truth_baseline_rows = []
    truth_target_rows = []
    baseline_target_rows = []
    with torch.inference_mode():
        for start in range(0, len(val_idx), batch_size):
            stop = min(start + batch_size, len(val_idx))
            predicted = torch.as_tensor(
                np.array(prediction[start:stop], copy=True),
                dtype=torch.complex64,
                device=device,
            )
            baseline = stable_unit(pdp_spectrum(predicted, spec))
            neighbor_t = torch.as_tensor(
                neighbors[start:stop],
                dtype=torch.long,
                device=device,
            )
            values = spectra[neighbor_t]
            nearest_global = (
                values * values[:, :1]
            ).sum(dim=-1).mean(dim=(2, 3))
            distance_t = torch.as_tensor(
                distance[start:stop],
                dtype=torch.float32,
                device=device,
            ).clamp_min(0.3)
            weight = torch.softmax(
                -distance_power * distance_t.log()
                + anchor_strength * nearest_global,
                dim=1,
            )
            target = stable_unit(
                torch.einsum(
                    "bk,bkmns->bmns", weight, values
                )
            )
            truth = spectra[
                torch.as_tensor(
                    val_idx[start:stop],
                    dtype=torch.long,
                    device=device,
                )
            ]
            nearest = values[:, 0]
            baseline_target = (baseline * target).sum(dim=-1)
            baseline_nearest = (baseline * nearest).sum(dim=-1)
            target_nearest = (target * nearest).sum(dim=-1)
            neighbor_cosine = (
                values * target[:, None]
            ).sum(dim=-1)
            consensus = torch.einsum(
                "bk,bkmn->bmn", weight, neighbor_cosine
            )
            spread = torch.sqrt(
                torch.einsum(
                    "bk,bkmn->bmn",
                    weight,
                    (neighbor_cosine - consensus[:, None]).square(),
                ).clamp_min(0.0)
            )
            baseline_peak, baseline_entropy, baseline_centroid = (
                distribution_features(baseline)
            )
            target_peak, target_entropy, target_centroid = (
                distribution_features(target)
            )
            profile_t = torch.as_tensor(
                profile_distance[start:stop],
                dtype=torch.float32,
                device=device,
            )
            weight_entropy = -(
                weight * weight.clamp_min(1e-30).log()
            ).sum(dim=1) / np.log(k)
            nearest_agreement = nearest_global.mean(dim=1)
            features = torch.stack(
                [
                    baseline_target,
                    baseline_nearest,
                    target_nearest,
                    consensus,
                    spread,
                    baseline_peak,
                    target_peak,
                    (target_peak - baseline_peak).abs(),
                    baseline_entropy,
                    target_entropy,
                    (target_centroid - baseline_centroid).abs(),
                    expand_query_feature(
                        distance_t[:, 0].log(), baseline_target
                    ),
                    expand_query_feature(
                        distance_t.mean(dim=1).log(),
                        baseline_target,
                    ),
                    expand_query_feature(
                        profile_t[:, 0], baseline_target
                    ),
                    expand_query_feature(
                        profile_t.mean(dim=1), baseline_target
                    ),
                    expand_query_feature(
                        weight.max(dim=1).values, baseline_target
                    ),
                    expand_query_feature(
                        weight_entropy, baseline_target
                    ),
                    expand_query_feature(
                        nearest_agreement, baseline_target
                    ),
                ],
                dim=-1,
            )
            feature_rows.append(
                features.reshape(-1, len(FEATURE_NAMES))
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            truth_baseline_rows.append(
                (truth * baseline)
                .sum(dim=-1)
                .reshape(-1)
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            truth_target_rows.append(
                (truth * target)
                .sum(dim=-1)
                .reshape(-1)
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            baseline_target_rows.append(
                baseline_target.reshape(-1)
                .cpu()
                .numpy()
                .astype(np.float32)
            )
    slices_per_query = int(
        np.prod(spectra.shape[1:-1], dtype=np.int64)
    )
    return FoldData(
        name=name,
        val_idx=val_idx,
        slices_per_query=slices_per_query,
        features=np.concatenate(feature_rows),
        truth_baseline=np.concatenate(truth_baseline_rows),
        truth_target=np.concatenate(truth_target_rows),
        baseline_target=np.concatenate(baseline_target_rows),
    )


def blend_score(
    beta: torch.Tensor,
    truth_baseline: torch.Tensor,
    truth_target: torch.Tensor,
    baseline_target: torch.Tensor,
) -> torch.Tensor:
    numerator = (
        (1.0 - beta) * truth_baseline
        + beta * truth_target
    )
    denominator2 = (
        (1.0 - beta).square()
        + beta.square()
        + 2.0
        * beta
        * (1.0 - beta)
        * baseline_target
    )
    return numerator / denominator2.clamp_min(1e-12).sqrt()


def train_gate(
    folds: list[FoldData],
    hidden: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
    exclude_indices: set[int] | None = None,
) -> tuple[SliceGate, np.ndarray, np.ndarray]:
    row_masks = []
    for fold in folds:
        if exclude_indices:
            query_keep = np.asarray(
                [
                    int(index) not in exclude_indices
                    for index in fold.val_idx
                ],
                dtype=bool,
            )
            row_masks.append(
                np.repeat(query_keep, fold.slices_per_query)
            )
        else:
            row_masks.append(
                np.ones(len(fold.features), dtype=bool)
            )
    features = np.concatenate(
        [
            fold.features[keep]
            for fold, keep in zip(folds, row_masks)
        ]
    )
    truth_baseline = np.concatenate(
        [
            fold.truth_baseline[keep]
            for fold, keep in zip(folds, row_masks)
        ]
    )
    truth_target = np.concatenate(
        [
            fold.truth_target[keep]
            for fold, keep in zip(folds, row_masks)
        ]
    )
    baseline_target = np.concatenate(
        [
            fold.baseline_target[keep]
            for fold, keep in zip(folds, row_masks)
        ]
    )
    mean = features.mean(axis=0).astype(np.float32)
    std = features.std(axis=0).clip(1e-4).astype(np.float32)
    feature_t = torch.as_tensor(
        (features - mean) / std,
        dtype=torch.float32,
        device=device,
    )
    truth_baseline_t = torch.as_tensor(
        truth_baseline, dtype=torch.float32, device=device
    )
    truth_target_t = torch.as_tensor(
        truth_target, dtype=torch.float32, device=device
    )
    baseline_target_t = torch.as_tensor(
        baseline_target, dtype=torch.float32, device=device
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    torch.manual_seed(seed)
    model = SliceGate(len(FEATURE_NAMES), hidden).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    for _ in range(epochs):
        order = torch.randperm(
            len(feature_t), generator=generator, device=device
        )
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            beta = model(feature_t[index])
            score = blend_score(
                beta,
                truth_baseline_t[index],
                truth_target_t[index],
                baseline_target_t[index],
            )
            loss = -score.mean() + 0.001 * (
                beta - 0.35
            ).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model, mean, std


def evaluate_gate(
    fold: FoldData,
    model: SliceGate,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    strengths: list[float],
    query_keep: np.ndarray | None = None,
) -> dict:
    features = torch.as_tensor(
        (fold.features - mean) / std,
        dtype=torch.float32,
        device=device,
    )
    truth_baseline = torch.as_tensor(
        fold.truth_baseline, dtype=torch.float32, device=device
    )
    truth_target = torch.as_tensor(
        fold.truth_target, dtype=torch.float32, device=device
    )
    baseline_target = torch.as_tensor(
        fold.baseline_target, dtype=torch.float32, device=device
    )
    with torch.inference_mode():
        raw_beta = model(features)
        fixed = blend_score(
            torch.full_like(raw_beta, 0.35),
            truth_baseline,
            truth_target,
            baseline_target,
        )
        grid = torch.linspace(0.0, 1.0, 41, device=device)
        oracle_scores = blend_score(
            grid[:, None],
            truth_baseline[None],
            truth_target[None],
            baseline_target[None],
        )
        oracle, oracle_index = oracle_scores.max(dim=0)
        if query_keep is not None:
            keep = torch.as_tensor(
                np.repeat(query_keep, fold.slices_per_query),
                dtype=torch.bool,
                device=device,
            )
            raw_beta = raw_beta[keep]
            fixed = fixed[keep]
            oracle = oracle[keep]
            oracle_index = oracle_index[keep]
            truth_baseline = truth_baseline[keep]
            truth_target = truth_target[keep]
            baseline_target = baseline_target[keep]
        strength_scores = {}
        strength_betas = {}
        for strength in strengths:
            beta = 0.35 + strength * (raw_beta - 0.35)
            score = blend_score(
                beta,
                truth_baseline,
                truth_target,
                baseline_target,
            )
            strength_scores[f"{strength:g}"] = float(score.mean())
            strength_betas[f"{strength:g}"] = {
                "mean": float(beta.mean()),
                "std": float(beta.std()),
                "p10": float(torch.quantile(beta, 0.1)),
                "median": float(torch.quantile(beta, 0.5)),
                "p90": float(torch.quantile(beta, 0.9)),
            }
    fixed_score = float(fixed.mean())
    return {
        "fixed_beta035_score": fixed_score,
        "strength_scores": strength_scores,
        "strength_gains": {
            key: value - fixed_score
            for key, value in strength_scores.items()
        },
        "strength_betas": strength_betas,
        "oracle_score": float(oracle.mean()),
        "oracle_gain_over_fixed": float(
            oracle.mean() - fixed.mean()
        ),
        "oracle_beta_mean": float(
            grid[oracle_index].mean()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", default="Round1_Map(2)")
    parser.add_argument(
        "--cache-dir", default="cache/teammate_knn_hvp"
    )
    parser.add_argument(
        "--prediction-template",
        default=(
            "cache/dual_gate50_10_projection/"
            "split_s{seed}_prediction.npy"
        ),
    )
    parser.add_argument(
        "--tune-seeds", default="1890,3716,962,1022"
    )
    parser.add_argument("--audit-seed", default="2262")
    parser.add_argument("--profile-lambda", type=float, default=24.0)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--distance-power", type=float, default=0.5)
    parser.add_argument("--anchor-strength", type=float, default=4.0)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train-batch-size", type=int, default=65536)
    parser.add_argument("--channel-batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--strength-grid", default="0,0.1,0.2,0.3,0.4,0.5,0.75,1"
    )
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out", default="docs/radial_pdp_gate/result.json"
    )
    args = parser.parse_args()

    tune = [item for item in args.tune_seeds.split(",") if item]
    names = [*tune, args.audit_seed]
    strengths = [
        float(item)
        for item in args.strength_grid.split(",")
        if item
    ]
    datadir = Path(args.datadir)
    spec = load_setup(datadir / "Round1_Setup.json")
    positions = np.load(
        datadir / "Round1_Train_Pos.npy"
    ).astype(np.float32)
    points = load_point_cloud(datadir / "Round1_Map.ply")
    heightmap, x0, y0, resolution = build_heightmap(points)
    profiles = local_radial_profiles(
        positions,
        np.asarray(spec.bs_position, dtype=np.float32),
        heightmap,
        x0,
        y0,
        resolution,
    )["radial_height"]
    del points, heightmap
    device = torch.device(args.device)
    cached = np.load(
        Path(args.cache_dir) / "train_pdp.npy", mmap_mode="r"
    )
    spectra = stable_unit(
        torch.as_tensor(
            np.array(cached, copy=True),
            dtype=torch.float32,
            device=device,
        )
    )
    del cached

    folds = {}
    for name in names:
        val_idx = np.asarray(
            sorted(
                reproduce_val_indices(
                    len(positions), 0.1, int(name)
                )
            ),
            dtype=np.int64,
        )
        folds[name] = prepare_fold(
            name,
            val_idx,
            positions,
            profiles,
            spectra,
            args.prediction_template,
            args.profile_lambda,
            args.k,
            args.distance_power,
            args.anchor_strength,
            spec,
            args.channel_batch_size,
            device,
        )
        print(f"[radial-pdp-gate] prepared split={name}", flush=True)

    scores = {}
    for held_name in tune:
        training_folds = [
            folds[name] for name in tune if name != held_name
        ]
        model, mean, std = train_gate(
            training_folds,
            args.hidden,
            args.epochs,
            args.train_batch_size,
            args.learning_rate,
            args.weight_decay,
            args.seed + int(held_name),
            device,
            exclude_indices=set(
                folds[held_name].val_idx.tolist()
            ),
        )
        scores[held_name] = evaluate_gate(
            folds[held_name],
            model,
            mean,
            std,
            device,
            strengths,
        )
        print(
            f"[radial-pdp-gate] held={held_name} "
            "raw_gain="
            f"{scores[held_name]['strength_gains']['1']:+.6f}",
            flush=True,
        )
        del model

    final_model, final_mean, final_std = train_gate(
        [folds[name] for name in tune],
        args.hidden,
        args.epochs,
        args.train_batch_size,
        args.learning_rate,
        args.weight_decay,
        args.seed,
        device,
    )
    scores[args.audit_seed] = evaluate_gate(
        folds[args.audit_seed],
        final_model,
        final_mean,
        final_std,
        device,
        strengths,
    )
    tune_union = {
        int(index)
        for name in tune
        for index in folds[name].val_idx
    }
    strict_keep = np.asarray(
        [
            int(index) not in tune_union
            for index in folds[args.audit_seed].val_idx
        ],
        dtype=bool,
    )
    scores["audit_strict"] = evaluate_gate(
        folds[args.audit_seed],
        final_model,
        final_mean,
        final_std,
        device,
        strengths,
        query_keep=strict_keep,
    )
    strength_rows = [
        {
            "strength": strength,
            "tune_gain": {
                name: scores[name]["strength_gains"][
                    f"{strength:g}"
                ]
                for name in tune
            },
        }
        for strength in strengths
    ]
    for row in strength_rows:
        values = list(row["tune_gain"].values())
        row["tune_gain_mean"] = float(np.mean(values))
        row["tune_gain_worst"] = float(np.min(values))
        key = f"{row['strength']:g}"
        row["audit_gain"] = scores[args.audit_seed][
            "strength_gains"
        ][key]
        row["strict_audit_gain"] = scores["audit_strict"][
            "strength_gains"
        ][key]
    strength_rows.sort(
        key=lambda row: (
            row["tune_gain_worst"] >= 0.0,
            row["tune_gain_mean"],
            row["tune_gain_worst"],
        ),
        reverse=True,
    )
    selected = strength_rows[0]
    payload = {
        "selection_policy": (
            "each tune fold uses a gate trained on the other three; "
            "audit uses all four tune folds and no audit labels"
        ),
        "feature_names": FEATURE_NAMES,
        "prediction_template": args.prediction_template,
        "profile_lambda": args.profile_lambda,
        "k": args.k,
        "distance_power": args.distance_power,
        "anchor_strength": args.anchor_strength,
        "architecture": {
            "hidden": args.hidden,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
        "selected_strength": selected,
        "ranked_strengths": strength_rows,
        "scores": scores,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    checkpoint = out.with_suffix(".pt")
    torch.save(
        {
            "model_state": final_model.state_dict(),
            "feature_names": FEATURE_NAMES,
            "feature_mean": final_mean,
            "feature_std": final_std,
            "architecture": payload["architecture"],
            "selection": {
                "profile_lambda": args.profile_lambda,
                "k": args.k,
                "distance_power": args.distance_power,
                "anchor_strength": args.anchor_strength,
                "gate_strength": selected["strength"],
            },
        },
        checkpoint,
    )
    print(
        "RADIAL_PDP_GATE_DONE "
        f"strength={selected['strength']:g} "
        f"tune={selected['tune_gain_mean']:+.6f} "
        f"worst={selected['tune_gain_worst']:+.6f} "
        f"audit={selected['audit_gain']:+.6f} "
        f"strict={selected['strict_audit_gain']:+.6f} "
        f"out={out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
