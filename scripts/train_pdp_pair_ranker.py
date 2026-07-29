#!/usr/bin/env python3
"""Learn a spatially varying metric from clean PDP pair similarities."""
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
from sweep_mapaware_spectrum_knn import map_features, robust_standardize
from train_spectral_latent_mlp import position_features
from wireless_twin.data.map_loader import load_point_cloud
from wireless_twin.data.setup_config import load_setup
from wireless_twin.models.raytrace2 import build_heightmap
from wireless_twin.signal import pdp_spectrum


def stable_unit(value: torch.Tensor) -> torch.Tensor:
    maximum = value.amax(dim=-1, keepdim=True)
    scaled = value / maximum.clamp_min(torch.finfo(value.dtype).tiny)
    norm = scaled.norm(dim=-1, keepdim=True)
    return torch.where(norm > 0, scaled / norm.clamp_min(1e-30), 0.0)


class PairRanker(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value).squeeze(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", default="Round1_Map(2)")
    parser.add_argument("--cache-dir", default="cache/teammate_knn_hvp")
    parser.add_argument("--split-seed", type=int, default=1890)
    parser.add_argument("--baseline-val", required=True)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--outdir", default="docs/pdp_pair_ranker_s1890"
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

    def find_neighbors(
        query_idx: np.ndarray, exclude_self: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        extra = 1 if exclude_self else 0
        distance, local = cKDTree(positions[train_idx, :2]).query(
            positions[query_idx, :2], k=args.k + extra
        )
        selected = train_idx[local]
        if not exclude_self:
            return distance.astype(np.float32), selected
        output_d = np.empty((len(query_idx), args.k), dtype=np.float32)
        output_i = np.empty((len(query_idx), args.k), dtype=np.int64)
        for row, source in enumerate(query_idx):
            keep = selected[row] != source
            output_d[row] = distance[row][keep][: args.k]
            output_i[row] = selected[row][keep][: args.k]
        return output_d, output_i

    train_distance, train_neighbors = find_neighbors(train_idx, True)
    val_distance, val_neighbors = find_neighbors(val_idx, False)
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

    def pair_targets(
        query_idx: np.ndarray, neighbor_idx: np.ndarray
    ) -> np.ndarray:
        rows = []
        with torch.inference_mode():
            for start in range(0, len(query_idx), 4):
                stop = min(start + 4, len(query_idx))
                query_t = torch.as_tensor(
                    query_idx[start:stop],
                    dtype=torch.long,
                    device=device,
                )
                neighbor_t = torch.as_tensor(
                    neighbor_idx[start:stop],
                    dtype=torch.long,
                    device=device,
                )
                similarity = (
                    spectra[neighbor_t] * spectra[query_t, None]
                ).sum(dim=-1).mean(dim=2)
                rows.append(similarity.cpu())
        return torch.cat(rows).numpy().astype(np.float32)

    print("[ranker] computing clean pair labels", flush=True)
    train_target = pair_targets(train_idx, train_neighbors)
    val_target = pair_targets(val_idx, val_neighbors)

    points = load_point_cloud(datadir / "Round1_Map.ply")
    heightmap, x0, y0, resolution = build_heightmap(points)
    raw_map, _ = map_features(
        positions,
        np.asarray(spec.bs_position, dtype=np.float32),
        heightmap,
        x0,
        y0,
        resolution,
        ray_samples=128,
    )
    map_values, _, _ = robust_standardize(
        raw_map[train_idx], raw_map
    )
    encoded_position, _ = position_features(
        positions, map_values, train_idx
    )
    bs_xy = np.asarray(spec.bs_position[:2], dtype=np.float32)

    def pair_features(
        query_idx: np.ndarray,
        neighbor_idx: np.ndarray,
        distance: np.ndarray,
    ) -> np.ndarray:
        query_xy = positions[query_idx, :2]
        delta = positions[neighbor_idx, :2] - query_xy[:, None]
        radial = query_xy - bs_xy
        radial_norm = np.linalg.norm(radial, axis=1).clip(1e-3)
        radial_unit = radial / radial_norm[:, None]
        tangent_unit = np.column_stack(
            [-radial_unit[:, 1], radial_unit[:, 0]]
        )
        relative = np.stack(
            [
                distance / 5.0,
                np.log(np.maximum(distance, 0.3)),
                delta[..., 0] / 5.0,
                delta[..., 1] / 5.0,
                (delta * radial_unit[:, None]).sum(-1) / 5.0,
                (delta * tangent_unit[:, None]).sum(-1) / 5.0,
            ],
            axis=-1,
        )
        query_feature = np.broadcast_to(
            encoded_position[query_idx, None],
            (
                len(query_idx),
                args.k,
                encoded_position.shape[-1],
            ),
        )
        return np.concatenate(
            [
                query_feature,
                encoded_position[neighbor_idx],
                relative,
            ],
            axis=-1,
        ).astype(np.float32)

    train_feature = pair_features(
        train_idx, train_neighbors, train_distance
    )
    val_feature = pair_features(val_idx, val_neighbors, val_distance)
    center = train_target.mean()
    scale = train_target.std().clip(1e-3)
    train_target_z = (train_target - center) / scale
    model = PairRanker(train_feature.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-3
    )
    feature_t = torch.as_tensor(train_feature, device=device)
    target_t = torch.as_tensor(train_target_z, device=device)
    val_feature_t = torch.as_tensor(val_feature, device=device)
    best_loss = float("inf")
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(train_idx))
        for start in range(0, len(order), 64):
            rows = order[start : start + 64]
            prediction = model(feature_t[rows])
            regression = F.smooth_l1_loss(prediction, target_t[rows])
            ranking = F.cross_entropy(
                prediction, target_t[rows].argmax(dim=1)
            )
            loss = regression + 0.1 * ranking
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            predicted_val = model(val_feature_t)
            val_loss = float(
                F.mse_loss(
                    predicted_val,
                    torch.as_tensor(
                        (val_target - center) / scale, device=device
                    ),
                )
            )
        if epoch == 1 or epoch % 10 == 0:
            row = {"epoch": epoch, "val_pair_mse": val_loss}
            history.append(row)
            print(f"[ranker] {row}", flush=True)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }

    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        predicted_similarity = model(val_feature_t) * scale + center
    baseline_channel = torch.as_tensor(
        np.array(np.load(args.baseline_val, mmap_mode="r"), copy=True),
        dtype=torch.complex64,
        device=device,
    )
    baseline = stable_unit(pdp_spectrum(baseline_channel, spec))
    del baseline_channel
    truth = spectra[
        torch.as_tensor(val_idx, dtype=torch.long, device=device)
    ].reshape(len(val_idx), spec.m, spec.n, spec.s)
    neighbor_t = torch.as_tensor(
        val_neighbors, dtype=torch.long, device=device
    )
    neighbor_values = spectra[neighbor_t].reshape(
        len(val_idx), args.k, spec.m, spec.n, spec.s
    )
    log_distance = torch.as_tensor(
        np.log(np.maximum(val_distance, 0.3)), device=device
    )
    rows = []
    for temperature in (0.01, 0.02, 0.05, 0.1, 0.2):
        for distance_power in (0.0, 0.5, 1.0, 2.0):
            weight = torch.softmax(
                predicted_similarity / temperature
                - distance_power * log_distance,
                dim=1,
            )
            selected = stable_unit(
                (weight[:, :, None, None, None] * neighbor_values).sum(1)
            )
            for beta in (0.0, 0.25, 0.5, 0.75, 1.0):
                prediction = stable_unit(
                    (1.0 - beta) * baseline + beta * selected
                )
                rows.append(
                    {
                        "temperature": temperature,
                        "distance_power": distance_power,
                        "blend_beta": beta,
                        "score": float(
                            (prediction * truth).sum(-1).mean()
                        ),
                    }
                )
    rows.sort(key=lambda row: row["score"], reverse=True)
    result = {
        "split_seed": args.split_seed,
        "k": args.k,
        "best_pair_mse": best_loss,
        "pair_target_mean": float(train_target.mean()),
        "pair_target_std": float(train_target.std()),
        "baseline_score": float((baseline * truth).sum(-1).mean()),
        "best": rows[0],
        "rows": rows,
        "history": history,
    }
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": {
                key: value.detach().cpu()
                for key, value in best_state.items()
            },
            "input_dim": train_feature.shape[-1],
            "target_center": float(center),
            "target_scale": float(scale),
            "k": args.k,
        },
        outdir / "model.pt",
    )
    (outdir / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)
    print(f"PDP_PAIR_RANKER_DONE out={outdir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
