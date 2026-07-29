#!/usr/bin/env python3
"""Train a clean content-aware selector over neighboring PDP spectra."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

from _bootstrap import ROOT  # noqa: F401
from score_holdout import reproduce_val_indices
from wireless_twin.data.map_loader import load_point_cloud
from wireless_twin.data.setup_config import load_setup
from wireless_twin.models.raytrace2 import build_heightmap
from wireless_twin.models.spectrum_refiner import PdpContentAttention


def stable_unit(value: torch.Tensor) -> torch.Tensor:
    maximum = value.amax(dim=-1, keepdim=True)
    scaled = value / maximum.clamp_min(torch.finfo(value.dtype).tiny)
    norm = scaled.norm(dim=-1, keepdim=True)
    return torch.where(norm > 0, scaled / norm.clamp_min(1e-30), 0.0)


def content_features(spectra: torch.Tensor) -> torch.Tensor:
    blocks = spectra.reshape(*spectra.shape[:-1], 24, 8)
    mean = blocks.mean(dim=-1)
    maximum = blocks.amax(dim=-1)
    root_mean = blocks.clamp_min(0).sqrt().mean(dim=-1)
    mass = spectra.sum(dim=-1, keepdim=True)
    probability = spectra / mass.clamp_min(1e-30)
    axis = torch.linspace(
        -1.0, 1.0, spectra.shape[-1], device=spectra.device
    )
    centroid = (probability * axis).sum(dim=-1, keepdim=True)
    variance = (
        probability * (axis - centroid).square()
    ).sum(dim=-1, keepdim=True)
    entropy = -(
        probability * probability.clamp_min(1e-30).log()
    ).sum(dim=-1, keepdim=True) / np.log(spectra.shape[-1])
    peak_value, peak = spectra.max(dim=-1, keepdim=True)
    peak_position = (
        2.0 * peak.float() / max(spectra.shape[-1] - 1, 1) - 1.0
    )
    return torch.cat(
        [
            mean,
            maximum,
            root_mean,
            mass,
            centroid,
            variance.sqrt(),
            entropy,
            peak_value,
            peak_position,
        ],
        dim=-1,
    )


def slice_features(spec) -> np.ndarray:
    index = np.arange(spec.m * spec.n)
    antenna = index // spec.n
    ue = index % spec.n

    phv_mp = antenna // (spec.mh * spec.mv)
    phv_rem = antenna % (spec.mh * spec.mv)
    phv_mh = phv_rem // spec.mv
    phv_mv = phv_rem % spec.mv

    hvp_mh = antenna // (spec.mv * spec.mp)
    hvp_rem = antenna % (spec.mv * spec.mp)
    hvp_mv = hvp_rem // spec.mp
    hvp_mp = hvp_rem % spec.mp
    phase = 2.0 * np.pi * antenna / max(spec.m - 1, 1)
    values = np.column_stack(
        [
            2.0 * antenna / max(spec.m - 1, 1) - 1.0,
            2.0 * phv_mh / max(spec.mh - 1, 1) - 1.0,
            2.0 * phv_mv / max(spec.mv - 1, 1) - 1.0,
            2.0 * phv_mp - 1.0,
            2.0 * hvp_mh / max(spec.mh - 1, 1) - 1.0,
            2.0 * hvp_mv / max(spec.mv - 1, 1) - 1.0,
            2.0 * hvp_mp - 1.0,
            np.sin(phase),
            np.cos(phase),
            np.eye(spec.n, dtype=np.float32)[ue],
        ]
    )
    return values.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", default="Round1_Map(2)")
    parser.add_argument("--cache-dir", default="cache/teammate_knn_hvp")
    parser.add_argument("--split-seed", type=int, default=1890)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--query-batch", type=int, default=8)
    parser.add_argument("--slices-per-query", type=int, default=32)
    parser.add_argument("--eval-slice-batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--outdir", default="docs/pdp_content_attention_s1890"
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    datadir = Path(args.datadir)
    spec = load_setup(datadir / "Round1_Setup.json")
    positions = np.load(datadir / "Round1_Train_Pos.npy").astype(np.float32)
    val_idx = np.asarray(
        sorted(
            reproduce_val_indices(
                len(positions), 0.1, args.split_seed
            )
        ),
        dtype=np.int64,
    )
    train_idx = np.setdiff1d(np.arange(len(positions)), val_idx)

    def neighbors(
        query_idx: np.ndarray, exclude_self: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        extra = 1 if exclude_self else 0
        distance, local = cKDTree(positions[train_idx, :2]).query(
            positions[query_idx, :2], k=args.k + extra
        )
        selected = train_idx[local]
        if not exclude_self:
            return distance.astype(np.float32), selected
        result_d = np.empty((len(query_idx), args.k), dtype=np.float32)
        result_i = np.empty((len(query_idx), args.k), dtype=np.int64)
        for row, source in enumerate(query_idx):
            keep = selected[row] != source
            result_d[row] = distance[row][keep][: args.k]
            result_i[row] = selected[row][keep][: args.k]
        return result_d, result_i

    train_distance, train_neighbors = neighbors(train_idx, True)
    val_distance, val_neighbors = neighbors(val_idx, False)
    print("[content] loading and normalizing PDP cache", flush=True)
    spectra = stable_unit(
        torch.as_tensor(
            np.array(
                np.load(
                    Path(args.cache_dir) / "train_pdp.npy", mmap_mode="r"
                ),
                copy=True,
            ),
            dtype=torch.float32,
            device=device,
        )
    ).reshape(len(positions), spec.m * spec.n, spec.s)
    content = content_features(spectra)
    print(
        f"[content] spectra={tuple(spectra.shape)} "
        f"features={tuple(content.shape)}",
        flush=True,
    )

    points = load_point_cloud(datadir / "Round1_Map.ply")
    heightmap, x0, y0, resolution = build_heightmap(points)
    gx = np.clip(
        ((positions[:, 0] - x0) / resolution).astype(np.int64),
        0,
        heightmap.shape[0] - 1,
    )
    gy = np.clip(
        ((positions[:, 1] - y0) / resolution).astype(np.int64),
        0,
        heightmap.shape[1] - 1,
    )
    indoor = (heightmap[gx, gy] > positions[:, 2] + 0.5).astype(np.float32)
    del points, heightmap
    bs_xy = np.asarray(spec.bs_position[:2], dtype=np.float32)
    position_center = positions[train_idx, :2].mean(axis=0)
    position_scale = positions[train_idx, :2].std(axis=0).clip(1.0)
    base_slice = slice_features(spec)

    def batch_inputs(
        query_idx: np.ndarray,
        neighbor_idx: np.ndarray,
        distance: np.ndarray,
        selected_slices: np.ndarray,
    ):
        batch, sample_count = selected_slices.shape
        neighbor_t = torch.as_tensor(
            neighbor_idx, dtype=torch.long, device=device
        )
        slice_t = torch.as_tensor(
            selected_slices, dtype=torch.long, device=device
        )
        neighbor_content = content[
            neighbor_t[:, :, None], slice_t[:, None, :]
        ].permute(0, 2, 1, 3)
        neighbor_spectra = spectra[
            neighbor_t[:, :, None], slice_t[:, None, :]
        ].permute(0, 2, 1, 3)
        query_t = torch.as_tensor(
            query_idx, dtype=torch.long, device=device
        )
        truth = spectra[query_t[:, None], slice_t]

        delta = (
            positions[neighbor_idx, :2]
            - positions[query_idx, None, :2]
        )
        radial = positions[query_idx, :2] - bs_xy
        radial_norm = np.linalg.norm(radial, axis=1).clip(1e-3)
        radial_unit = radial / radial_norm[:, None]
        tangent_unit = np.column_stack(
            [-radial_unit[:, 1], radial_unit[:, 0]]
        )
        radial_delta = (delta * radial_unit[:, None]).sum(axis=-1)
        tangent_delta = (delta * tangent_unit[:, None]).sum(axis=-1)
        geometry_np = np.stack(
            [
                distance / 5.0,
                np.log(np.maximum(distance, 0.3)),
                delta[..., 0] / 5.0,
                delta[..., 1] / 5.0,
                radial_delta / 5.0,
                tangent_delta / 5.0,
                (
                    indoor[neighbor_idx]
                    == indoor[query_idx, None]
                ).astype(np.float32),
                (
                    np.linalg.norm(
                        positions[neighbor_idx, :2] - bs_xy, axis=-1
                    )
                    - radial_norm[:, None]
                )
                / 5.0,
            ],
            axis=-1,
        ).astype(np.float32)
        geometry = torch.as_tensor(geometry_np, device=device)
        geometry = geometry[:, None].expand(-1, sample_count, -1, -1)

        query_feature = np.column_stack(
            [
                (positions[query_idx, :2] - position_center)
                / position_scale,
                radial_norm / 200.0,
                radial_unit,
                indoor[query_idx],
            ]
        ).astype(np.float32)
        slice_np = base_slice[selected_slices]
        query_np = np.repeat(
            query_feature[:, None], sample_count, axis=1
        )
        condition = torch.as_tensor(
            np.concatenate([slice_np, query_np], axis=-1), device=device
        )
        log_distance = torch.as_tensor(
            np.log(np.maximum(distance, 0.3)), device=device
        )
        log_distance = log_distance[:, None].expand(
            -1, sample_count, -1
        )
        return (
            neighbor_content.reshape(
                batch * sample_count, args.k, -1
            ),
            geometry.reshape(batch * sample_count, args.k, -1),
            condition.reshape(batch * sample_count, -1),
            log_distance.reshape(batch * sample_count, args.k),
            neighbor_spectra.reshape(
                batch * sample_count, args.k, spec.s
            ),
            truth.reshape(batch * sample_count, spec.s),
        )

    content_dim = content.shape[-1]
    geometry_dim = 8
    condition_dim = base_slice.shape[-1] + 6
    model = PdpContentAttention(
        content_dim, geometry_dim, condition_dim
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2e-3, weight_decay=2e-3
    )

    def evaluate() -> tuple[float, float]:
        model.eval()
        total = 0.0
        baseline_total = 0.0
        count = 0
        with torch.inference_mode():
            for query_start in range(0, len(val_idx), 4):
                query_stop = min(query_start + 4, len(val_idx))
                rows = np.arange(query_start, query_stop)
                for slice_start in range(
                    0, spec.m * spec.n, args.eval_slice_batch
                ):
                    slice_stop = min(
                        slice_start + args.eval_slice_batch,
                        spec.m * spec.n,
                    )
                    selected = np.broadcast_to(
                        np.arange(slice_start, slice_stop),
                        (len(rows), slice_stop - slice_start),
                    )
                    inputs = batch_inputs(
                        val_idx[rows],
                        val_neighbors[rows],
                        val_distance[rows],
                        selected,
                    )
                    weights, _ = model(*inputs[:4])
                    prediction = (
                        weights[..., None] * inputs[4]
                    ).sum(dim=1)
                    cosine = F.cosine_similarity(
                        prediction, inputs[5], dim=-1
                    )
                    idw = inputs[3].mul(-2.0).softmax(dim=1)
                    baseline = (
                        idw[..., None] * inputs[4]
                    ).sum(dim=1)
                    baseline_cosine = F.cosine_similarity(
                        baseline, inputs[5], dim=-1
                    )
                    total += float(cosine.sum())
                    baseline_total += float(baseline_cosine.sum())
                    count += len(cosine)
        return total / count, baseline_total / count

    baseline_score = evaluate()[1]
    print(f"[content] strict IDW baseline={baseline_score:.6f}", flush=True)
    best_score = -float("inf")
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(train_idx))
        train_score = 0.0
        train_count = 0
        for start in range(0, len(train_idx), args.query_batch):
            rows = order[start : start + args.query_batch]
            selected = rng.integers(
                0,
                spec.m * spec.n,
                size=(len(rows), args.slices_per_query),
            )
            inputs = batch_inputs(
                train_idx[rows],
                train_neighbors[rows],
                train_distance[rows],
                selected,
            )
            weights, scores = model(*inputs[:4])
            prediction = (weights[..., None] * inputs[4]).sum(dim=1)
            cosine = F.cosine_similarity(
                prediction, inputs[5], dim=-1
            )
            valid = inputs[5].norm(dim=-1) > 0
            neighbor_cosine = (
                inputs[4] * inputs[5][:, None]
            ).sum(dim=-1)
            target_neighbor = neighbor_cosine.argmax(dim=1)
            loss = 1.0 - cosine[valid].mean()
            loss = loss + 0.02 * F.cross_entropy(
                scores[valid], target_neighbor[valid]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_score += float(cosine[valid].sum())
            train_count += int(valid.sum())
        val_score, _ = evaluate()
        row = {
            "epoch": epoch,
            "train_cosine": train_score / max(train_count, 1),
            "val_cosine": val_score,
            "idw_power": float(model.idw_power.detach()),
        }
        history.append(row)
        print(f"[content] {row}", flush=True)
        if val_score > best_score:
            best_score = val_score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("content attention produced no checkpoint")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoint = outdir / "model.pt"
    torch.save(
        {
            "model_state": best_state,
            "content_dim": int(content_dim),
            "geometry_dim": geometry_dim,
            "condition_dim": condition_dim,
            "k": args.k,
            "split_seed": args.split_seed,
            "seed": args.seed,
            "best_val_cosine": best_score,
            "position_center": position_center,
            "position_scale": position_scale,
        },
        checkpoint,
    )
    result = {
        "split_seed": args.split_seed,
        "k": args.k,
        "seed": args.seed,
        "baseline_idw": baseline_score,
        "best_val_cosine": best_score,
        "gain": best_score - baseline_score,
        "history": history,
        "checkpoint": str(checkpoint),
    }
    out = outdir / "result.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"PDP_CONTENT_ATTENTION_DONE out={out}", flush=True)


if __name__ == "__main__":
    main()
