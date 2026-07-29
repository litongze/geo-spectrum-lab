#!/usr/bin/env python3
"""Project clean neighbor attention onto local spatial moment constraints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree

from _bootstrap import ROOT  # noqa: F401
from score_holdout import reproduce_val_indices
from wireless_twin.data.map_loader import load_point_cloud
from wireless_twin.data.setup_config import load_setup
from wireless_twin.models.raytrace2 import build_heightmap
from wireless_twin.signal import (
    pas_spectrum,
    pas_spectrum_phv,
    pas_spectrum_pvh,
    pdp_spectrum,
)


class SliceAttention(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.idw_w = nn.Parameter(torch.tensor(2.0))

    def logits(
        self, features: torch.Tensor, log_distance: torch.Tensor
    ) -> torch.Tensor:
        return (
            self.mlp(features).squeeze(-1)
            - self.idw_w * log_distance
        )


def parse_floats(raw: str) -> list[float]:
    return [float(value) for value in raw.split(",") if value.strip()]


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


def moment_project(
    logits: torch.Tensor,
    delta: torch.Tensor,
    correction: float,
    axis_multiplier: torch.Tensor,
    iterations: int,
    residual_tolerance: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Exponentially tilt weights toward a requested spatial centroid.

    ``delta`` is shaped ``(B, K, 2)`` and ``logits`` is ``(B, K, Q)``.
    A correction of one requests zero first moment, while zero preserves the
    original attention centroid.
    """
    prior = torch.softmax(logits, dim=1)
    prior_mean = torch.einsum("bkq,bkd->bqd", prior, delta)
    if correction == 0:
        return prior, {
            "mean_residual": 0.0,
            "fallback_fraction": 0.0,
            "prior_centroid_norm": float(
                prior_mean.norm(dim=-1).mean()
            ),
            "projected_centroid_norm": float(
                prior_mean.norm(dim=-1).mean()
            ),
        }

    axis_correction = (
        correction * axis_multiplier
    ).clamp(min=0.0, max=1.25)
    target = (1.0 - axis_correction[None, None]) * prior_mean
    multiplier = torch.zeros_like(target)
    for _ in range(iterations):
        tilted = logits + torch.einsum(
            "bqd,bkd->bkq", multiplier, delta
        )
        weight = torch.softmax(tilted, dim=1)
        mean = torch.einsum("bkq,bkd->bqd", weight, delta)
        second = torch.einsum(
            "bkq,bki,bkj->bqij", weight, delta, delta
        )
        covariance = second - torch.einsum(
            "bqi,bqj->bqij", mean, mean
        )
        error = mean - target
        a = covariance[..., 0, 0] + 1e-5
        b = covariance[..., 0, 1]
        c = covariance[..., 1, 1] + 1e-5
        determinant = (a * c - b.square()).clamp_min(1e-8)
        step_x = (c * error[..., 0] - b * error[..., 1]) / determinant
        step_y = (a * error[..., 1] - b * error[..., 0]) / determinant
        step = torch.stack([step_x, step_y], dim=-1)
        step_norm = step.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        step = step * torch.clamp(5.0 / step_norm, max=1.0)
        multiplier = (multiplier - step).clamp(-60.0, 60.0)

    tilted = logits + torch.einsum(
        "bqd,bkd->bkq", multiplier, delta
    )
    projected = torch.softmax(tilted, dim=1)
    projected_mean = torch.einsum(
        "bkq,bkd->bqd", projected, delta
    )
    residual = (projected_mean - target).norm(dim=-1)
    invalid = (~torch.isfinite(residual)) | (
        residual > residual_tolerance
    )
    projected = torch.where(
        invalid[:, None, :], prior, projected
    )
    final_mean = torch.einsum(
        "bkq,bkd->bqd", projected, delta
    )
    final_target = torch.where(
        invalid[..., None], prior_mean, target
    )
    final_residual = (final_mean - final_target).norm(dim=-1)
    return projected, {
        "mean_residual": float(final_residual.mean()),
        "fallback_fraction": float(invalid.float().mean()),
        "prior_centroid_norm": float(prior_mean.norm(dim=-1).mean()),
        "projected_centroid_norm": float(
            final_mean.norm(dim=-1).mean()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", default="Round1_Map(2)")
    parser.add_argument("--cache-dir", default="cache/teammate_knn_hvp")
    parser.add_argument(
        "--baseline-dir",
        default="docs/clean_noeps_panel_phv_geom_k16",
    )
    parser.add_argument("--domains", default="pas,pdp")
    parser.add_argument(
        "--pas-layout",
        choices=("hvp", "pvh", "phv"),
        default="phv",
    )
    parser.add_argument(
        "--pas-checkpoint-template",
        default=None,
        help="optional format string containing {seed} for panel PAS arms",
    )
    parser.add_argument(
        "--pdp-checkpoint-template",
        default=None,
        help="optional format string containing {seed} for panel PDP arms",
    )
    parser.add_argument("--tune-seeds", default="1890,3716,962,1022")
    parser.add_argument("--audit-seed", type=int, default=2262)
    parser.add_argument(
        "--testmatched",
        default="handoff_to_teammate_20260727/splits/"
        "test_matched_seed2026_val.npy",
    )
    parser.add_argument(
        "--correction-grid", default="0,0.25,0.5,0.75,1"
    )
    parser.add_argument("--radial-multiplier", type=float, default=1.0)
    parser.add_argument("--tangent-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--blend-grid", default="0,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.75,1"
    )
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--residual-tolerance", type=float, default=0.03)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--save-experts-dir",
        default=None,
        help=(
            "optionally save expert spectra; intended for runs with one "
            "correction value"
        ),
    )
    parser.add_argument(
        "--out", default="docs/moment_attention/result.json"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    tune_seeds = [
        int(value) for value in args.tune_seeds.split(",") if value
    ]
    panel = tune_seeds + [args.audit_seed]
    corrections = parse_floats(args.correction_grid)
    blends = parse_floats(args.blend_grid)
    domains = [
        value.strip() for value in args.domains.split(",") if value.strip()
    ]
    if not set(domains) <= {"pas", "pdp"}:
        raise ValueError("--domains must contain only pas,pdp")
    datadir = Path(args.datadir)
    positions = np.load(
        datadir / "Round1_Train_Pos.npy"
    ).astype(np.float32)
    spec = load_setup(datadir / "Round1_Setup.json")

    points = load_point_cloud(datadir / "Round1_Map.ply")
    heightmap, x0, y0, resolution = build_heightmap(points)
    gx = np.clip(
        np.floor((positions[:, 0] - x0) / resolution).astype(np.int64),
        0,
        heightmap.shape[0] - 1,
    )
    gy = np.clip(
        np.floor((positions[:, 1] - y0) / resolution).astype(np.int64),
        0,
        heightmap.shape[1] - 1,
    )
    indoor = (heightmap[gx, gy] > 2.0).astype(np.float32)

    split_indices = {
        str(seed): np.asarray(
            sorted(reproduce_val_indices(len(positions), 0.1, seed)),
            dtype=np.int64,
        )
        for seed in panel
    }
    if args.testmatched:
        split_indices["testmatched"] = np.load(args.testmatched).astype(
            np.int64
        )
    all_indices = np.arange(len(positions), dtype=np.int64)

    def checkpoint_path(domain: str, split_name: str) -> Path:
        if split_name == "testmatched":
            return Path(
                "checkpoints/testmatched_phv/"
                + (
                    "pas16_pas_k16s0.pt"
                    if domain == "pas"
                    else "pdp32_pdp_k32s0.pt"
                )
            )
        seed = int(split_name)
        template = (
            args.pas_checkpoint_template
            if domain == "pas"
            else args.pdp_checkpoint_template
        )
        if template:
            return Path(template.format(seed=seed))
        if domain == "pas":
            return Path(
                "checkpoints/clean_panel_phv_geom_k16"
                f"/s{seed}/nbrattn_clean_k16_pas_k16s0.pt"
            )
        return Path(
            "checkpoints/clean_panel"
            f"/s{seed}/nbrattn_clean_k32_pdp_k32s0.pt"
        )

    def build_features(
        domain: str,
        spectra: torch.Tensor,
        query_idx: np.ndarray,
        neighbors: np.ndarray,
        distance: np.ndarray,
        feature_set: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, k = neighbors.shape
        candidate = spectra[
            torch.as_tensor(neighbors, dtype=torch.long, device=device)
        ]
        if domain == "pas":
            values = candidate.reshape(
                batch, k, -1, candidate.shape[-1]
            )
        else:
            values = candidate.reshape(
                batch, k, -1, candidate.shape[-1]
            )
        mean = values.mean(dim=1, keepdim=True)
        agreement = F.cosine_similarity(values, mean, dim=-1)
        distance_t = torch.as_tensor(
            distance, dtype=torch.float32, device=device
        )
        expanded_distance = distance_t[:, :, None].expand(
            -1, -1, values.shape[2]
        )
        neighbor_indoor = torch.as_tensor(
            indoor[neighbors], dtype=torch.float32, device=device
        )[:, :, None].expand(-1, -1, values.shape[2])
        target_indoor = torch.as_tensor(
            indoor[query_idx], dtype=torch.float32, device=device
        )[:, None, None].expand(-1, k, values.shape[2])
        columns = [
            expanded_distance / 3.0,
            (neighbor_indoor == target_indoor).float(),
            agreement,
            agreement.square(),
            torch.ones_like(expanded_distance),
            (expanded_distance < 2.5).float(),
        ]
        if feature_set == "geometry":
            target_xy = torch.as_tensor(
                positions[query_idx, :2],
                dtype=torch.float32,
                device=device,
            )
            neighbor_xy = torch.as_tensor(
                positions[neighbors, :2],
                dtype=torch.float32,
                device=device,
            )
            delta = neighbor_xy - target_xy[:, None]
            bs_xy = torch.as_tensor(
                spec.bs_position[:2],
                dtype=torch.float32,
                device=device,
            )
            radial = target_xy - bs_xy
            radius = radial.norm(dim=-1).clamp_min(1e-3)
            radial_unit = radial / radius[:, None]
            tangent_unit = torch.stack(
                [-radial_unit[:, 1], radial_unit[:, 0]], dim=-1
            )
            radial_delta = (delta * radial_unit[:, None]).sum(dim=-1)
            tangent_delta = (delta * tangent_unit[:, None]).sum(dim=-1)

            def expand(value: torch.Tensor) -> torch.Tensor:
                return value[:, :, None].expand(
                    -1, -1, values.shape[2]
                )

            angle = torch.atan2(radial[:, 1], radial[:, 0])
            columns.extend(
                [
                    expand(delta[..., 0] / 5.0),
                    expand(delta[..., 1] / 5.0),
                    expand(radial_delta / 5.0),
                    expand(tangent_delta / 5.0),
                    (radius / 200.0)[:, None, None].expand(
                        -1, k, values.shape[2]
                    ),
                    angle.sin()[:, None, None].expand(
                        -1, k, values.shape[2]
                    ),
                    angle.cos()[:, None, None].expand(
                        -1, k, values.shape[2]
                    ),
                    neighbor_indoor,
                    target_indoor,
                ]
            )
        elif feature_set != "basic":
            raise ValueError(f"unsupported feature_set={feature_set}")
        features = torch.stack(columns, dim=-1)
        log_distance = torch.log(
            distance_t.clamp_min(0.3)
        )[:, :, None]
        return features, values, log_distance

    payload: dict[str, object] = {
        "selection_policy": "four tune splits only",
        "pas_layout": args.pas_layout,
        "tune_seeds": tune_seeds,
        "audit_seed": args.audit_seed,
        "correction_grid": corrections,
        "blend_grid": blends,
        "iterations": args.iterations,
        "residual_tolerance": args.residual_tolerance,
        "radial_multiplier": args.radial_multiplier,
        "tangent_multiplier": args.tangent_multiplier,
        "domains": {},
    }
    cache_names = {
        "pas": f"train_pas_{args.pas_layout}.npy",
        "pdp": "train_pdp.npy",
    }
    pas_transforms = {
        "hvp": pas_spectrum,
        "pvh": pas_spectrum_pvh,
        "phv": pas_spectrum_phv,
    }
    transforms = {
        "pas": pas_transforms[args.pas_layout],
        "pdp": pdp_spectrum,
    }

    for domain in domains:
        print(f"[moment] loading {domain}", flush=True)
        cached = np.load(
            Path(args.cache_dir) / cache_names[domain], mmap_mode="r"
        )
        spectra = stable_unit(
            torch.as_tensor(
                np.array(cached, copy=True),
                dtype=torch.float32,
                device=device,
            )
        )
        del cached
        split_rows = {}
        for split_name, val_idx in split_indices.items():
            pool_idx = np.setdiff1d(all_indices, val_idx)
            checkpoint = checkpoint_path(domain, split_name)
            checkpoint_payload = torch.load(
                checkpoint, map_location=device, weights_only=False
            )
            meta = checkpoint_payload["meta"]
            k = int(meta["K"])
            feature_set = str(meta.get("feature_set", "basic"))
            tree = cKDTree(positions[pool_idx, :2])
            distance, local = tree.query(
                positions[val_idx, :2], k=k
            )
            distance = np.asarray(distance, dtype=np.float32)
            neighbors = pool_idx[np.asarray(local)]
            feature_dim = int(
                meta.get(
                    "feature_dim",
                    15 if meta.get("feature_set") == "geometry" else 6,
                )
            )
            model = SliceAttention(feature_dim).to(device)
            model.load_state_dict(checkpoint_payload["model_state"])
            model.eval()

            baseline = None
            if split_name != "testmatched":
                baseline_path = (
                    Path(args.baseline_dir)
                    / f"split_s{split_name}_prediction.npy"
                )
                channel = torch.as_tensor(
                    np.array(
                        np.load(baseline_path, mmap_mode="r"), copy=True
                    ),
                    dtype=torch.complex64,
                    device=device,
                )
                baseline = stable_unit(transforms[domain](channel, spec))
                del channel
            truth = spectra[
                torch.as_tensor(
                    val_idx, dtype=torch.long, device=device
                )
            ]
            totals = {
                correction: {
                    blend: 0.0 for blend in blends
                }
                for correction in corrections
            }
            expert_totals = {correction: 0.0 for correction in corrections}
            saved_experts = (
                {correction: [] for correction in corrections}
                if args.save_experts_dir
                else None
            )
            diagnostic_totals = {
                correction: {
                    "mean_residual": 0.0,
                    "fallback_fraction": 0.0,
                    "prior_centroid_norm": 0.0,
                    "projected_centroid_norm": 0.0,
                    "chunks": 0,
                }
                for correction in corrections
            }
            count = 0
            with torch.inference_mode():
                for start in range(0, len(val_idx), args.batch_size):
                    stop = min(
                        start + args.batch_size, len(val_idx)
                    )
                    query_idx = val_idx[start:stop]
                    features, values, log_distance = build_features(
                        domain,
                    spectra,
                    query_idx,
                    neighbors[start:stop],
                    distance[start:stop],
                    feature_set,
                    )
                    logits = model.logits(features, log_distance)
                    target = truth[start:stop].reshape(
                        stop - start, -1, truth.shape[-1]
                    )
                    base = (
                        baseline[start:stop].reshape(
                            stop - start,
                            -1,
                            baseline.shape[-1],
                        )
                        if baseline is not None
                        else None
                    )
                    delta = torch.as_tensor(
                        positions[neighbors[start:stop], :2]
                        - positions[query_idx, None, :2],
                        dtype=torch.float32,
                        device=device,
                    )
                    scale = torch.as_tensor(
                        np.median(distance[start:stop], axis=1),
                        dtype=torch.float32,
                        device=device,
                    ).clamp_min(0.3)
                    delta = delta / scale[:, None, None]
                    query_xy = torch.as_tensor(
                        positions[query_idx, :2],
                        dtype=torch.float32,
                        device=device,
                    )
                    bs_xy = torch.as_tensor(
                        spec.bs_position[:2],
                        dtype=torch.float32,
                        device=device,
                    )
                    radial = query_xy - bs_xy
                    radial /= radial.norm(
                        dim=-1, keepdim=True
                    ).clamp_min(1e-6)
                    tangent = torch.stack(
                        [-radial[:, 1], radial[:, 0]], dim=-1
                    )
                    delta = torch.stack(
                        [
                            (delta * radial[:, None]).sum(dim=-1),
                            (delta * tangent[:, None]).sum(dim=-1),
                        ],
                        dim=-1,
                    )
                    axis_multiplier = torch.tensor(
                        [
                            args.radial_multiplier,
                            args.tangent_multiplier,
                        ],
                        dtype=torch.float32,
                        device=device,
                    )
                    for correction in corrections:
                        weight, diagnostic = moment_project(
                            logits,
                            delta,
                            correction,
                            axis_multiplier,
                            args.iterations,
                            args.residual_tolerance,
                        )
                        expert = stable_unit(
                            torch.einsum(
                                "bkq,bkql->bql", weight, values
                            )
                        )
                        expert_totals[correction] += float(
                            (expert * target).sum()
                        )
                        if saved_experts is not None:
                            saved_experts[correction].append(
                                expert.detach().cpu()
                            )
                        if base is not None:
                            for blend in blends:
                                prediction = stable_unit(
                                    (1.0 - blend) * base
                                    + blend * expert
                                )
                                totals[correction][blend] += float(
                                    (prediction * target).sum()
                                )
                        for key, value in diagnostic.items():
                            diagnostic_totals[correction][key] += value
                        diagnostic_totals[correction]["chunks"] += 1
                    count += target.numel() // target.shape[-1]

            row = {
                "checkpoint": str(checkpoint),
                "k": k,
                "feature_set": feature_set,
                "expert_scores": {
                    str(correction): expert_totals[correction] / count
                    for correction in corrections
                },
                "blend_scores": (
                    {
                        str(correction): {
                            str(blend): totals[correction][blend] / count
                            for blend in blends
                        }
                        for correction in corrections
                    }
                    if baseline is not None
                    else None
                ),
                "diagnostics": {},
            }
            if baseline is not None:
                row["baseline_score"] = float(
                    (
                        baseline.reshape(
                            len(val_idx), -1, baseline.shape[-1]
                        )
                        * truth.reshape(
                            len(val_idx), -1, truth.shape[-1]
                        )
                    ).sum()
                    / count
                )
            for correction in corrections:
                values_d = diagnostic_totals[correction]
                chunks = values_d.pop("chunks")
                row["diagnostics"][str(correction)] = {
                    key: value / chunks
                    for key, value in values_d.items()
                }
            if saved_experts is not None:
                expert_dir = Path(args.save_experts_dir)
                expert_dir.mkdir(parents=True, exist_ok=True)
                for correction in corrections:
                    expert_array = (
                        torch.cat(saved_experts[correction])
                        .reshape(tuple(truth.shape))
                        .numpy()
                        .astype(np.float32)
                    )
                    expert_path = (
                        expert_dir
                        / f"{domain}_{split_name}_c{correction:g}.npy"
                    )
                    np.save(expert_path, expert_array)
                    row.setdefault("expert_artifacts", {})[
                        str(correction)
                    ] = str(expert_path)
            split_rows[split_name] = row
            print(
                f"[moment] {domain} split={split_name} "
                f"base={row.get('baseline_score', float('nan')):.6f} "
                f"raw={row['expert_scores'][str(corrections[0])]:.6f} "
                f"full={row['expert_scores'][str(corrections[-1])]:.6f}",
                flush=True,
            )
            del model, truth, baseline
            torch.cuda.empty_cache()

        ranked = []
        for correction in corrections:
            for blend in blends:
                tune = [
                    split_rows[str(seed)]["blend_scores"][
                        str(correction)
                    ][str(blend)]
                    for seed in tune_seeds
                ]
                ranked.append(
                    {
                        "correction": correction,
                        "blend": blend,
                        "tune_median": float(np.median(tune)),
                        "tune_mean": float(np.mean(tune)),
                        "tune_worst": float(np.min(tune)),
                        "audit": split_rows[str(args.audit_seed)][
                            "blend_scores"
                        ][str(correction)][str(blend)],
                        "testmatched_expert": (
                            split_rows["testmatched"]["expert_scores"][
                                str(correction)
                            ]
                            if "testmatched" in split_rows
                            else None
                        ),
                        "scores": {
                            split_name: row["blend_scores"][
                                str(correction)
                            ][str(blend)]
                            for split_name, row in split_rows.items()
                            if row["blend_scores"] is not None
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
        payload["domains"][domain] = {
            "ranked": ranked,
            "splits": split_rows,
        }
        print(
            f"[moment] {domain} TOP "
            + json.dumps(ranked[:5], ensure_ascii=False),
            flush=True,
        )
        del spectra
        torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"MOMENT_ATTENTION_DONE out={out}", flush=True)


if __name__ == "__main__":
    main()
