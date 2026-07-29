#!/usr/bin/env python3
"""Measure whether label-free features can gate a coordinate spectrum expert."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from _bootstrap import ROOT  # noqa: F401
from score_holdout import reproduce_val_indices
from sweep_mapaware_spectrum_knn import map_features, robust_standardize
from train_coordinate_spectrum_mlp import (
    CoordinateSpectrumMLP,
    slice_features,
    stable_unit,
)
from train_spectral_latent_mlp import position_features
from wireless_twin.data.map_loader import load_point_cloud
from wireless_twin.data.setup_config import load_setup
from wireless_twin.models.raytrace2 import build_heightmap
from wireless_twin.signal import pas_spectrum_phv, pdp_spectrum


class CoordinateGate(torch.nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 24),
            torch.nn.SiLU(),
            torch.nn.Linear(24, 12),
            torch.nn.SiLU(),
            torch.nn.Linear(12, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1).sigmoid()


def fit_gate(
    features: torch.Tensor,
    baseline: torch.Tensor,
    coordinate: torch.Tensor,
    truth: torch.Tensor,
    fit_rows: np.ndarray,
    stop_rows: np.ndarray,
    seed: int,
) -> tuple[CoordinateGate, torch.Tensor, torch.Tensor, float]:
    torch.manual_seed(seed)
    fit_features = features[fit_rows]
    center = fit_features.mean(dim=(0, 1))
    scale = fit_features.std(dim=(0, 1)).clamp_min(1e-3)
    normalized = (features - center) / scale
    gate = CoordinateGate(features.shape[-1]).to(features.device)
    optimizer = torch.optim.AdamW(
        gate.parameters(), lr=3e-3, weight_decay=3e-2
    )
    best_score = -float("inf")
    best_state = None
    stale = 0
    fit_rows_t = torch.as_tensor(fit_rows, device=features.device)
    stop_rows_t = torch.as_tensor(stop_rows, device=features.device)
    for _ in range(600):
        gate.train()
        beta = gate(normalized[fit_rows_t])
        prediction = stable_unit(
            (1.0 - beta[..., None]) * baseline[fit_rows_t]
            + beta[..., None] * coordinate[fit_rows_t]
        )
        loss = -(prediction * truth[fit_rows_t]).sum(-1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        gate.eval()
        with torch.inference_mode():
            beta = gate(normalized[stop_rows_t])
            prediction = stable_unit(
                (1.0 - beta[..., None]) * baseline[stop_rows_t]
                + beta[..., None] * coordinate[stop_rows_t]
            )
            score = float(
                (prediction * truth[stop_rows_t]).sum(-1).mean()
            )
        if score > best_score + 1e-7:
            best_score = score
            best_state = {
                key: value.detach().clone()
                for key, value in gate.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= 60:
                break
    if best_state is None:
        raise RuntimeError("coordinate gate training produced no checkpoint")
    gate.load_state_dict(best_state)
    gate.eval()
    return gate, center, scale, best_score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", default="Round1_Map(2)")
    parser.add_argument("--cache-dir", default="cache/teammate_knn_hvp")
    parser.add_argument("--domain", choices=("pas", "pdp"), required=True)
    parser.add_argument("--split-seed", type=int, default=1890)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--baseline-val", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

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
    val_idx_t = torch.from_numpy(val_idx).to(device)
    payload = torch.load(
        args.checkpoint, map_location=device, weights_only=False
    )
    model = CoordinateSpectrumMLP(
        int(payload["input_dim"]),
        int(payload["output_dim"]),
        int(payload["hidden"]),
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    cache_name = (
        "train_pas_phv.npy" if args.domain == "pas" else "train_pdp.npy"
    )
    spectra = stable_unit(
        torch.as_tensor(
            np.array(
                np.load(Path(args.cache_dir) / cache_name, mmap_mode="r"),
                copy=True,
            ),
            dtype=torch.float32,
            device=device,
        )
    )
    if args.domain == "pas":
        spectra = spectra.reshape(
            len(positions), spec.n * spec.s, spec.mh * spec.mv
        )
    else:
        spectra = spectra.reshape(
            len(positions), spec.m * spec.n, spec.s
        )
    slice_count, output_dim = spectra.shape[1:]

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
    map_values, _, _ = robust_standardize(raw_map[train_idx], raw_map)
    position_np, _ = position_features(positions, map_values, train_idx)
    position = torch.from_numpy(position_np).to(device)
    slice_np = slice_features(spec, args.domain)
    slice_coordinate = torch.from_numpy(slice_np).to(device)

    def feature_batch(query: torch.Tensor) -> torch.Tensor:
        slices = torch.arange(slice_count, device=device)[None].expand(
            len(query), -1
        )
        position_part = position[query][:, None].expand(
            -1, slice_count, -1
        )
        slice_part = slice_coordinate[slices]
        return torch.cat([position_part, slice_part], dim=-1).reshape(
            -1, position_part.shape[-1] + slice_part.shape[-1]
        )

    chunks = []
    with torch.inference_mode():
        for start in range(0, len(val_idx), 4):
            query = val_idx_t[start : start + 4]
            chunks.append(
                model(feature_batch(query)).reshape(
                    len(query), slice_count, output_dim
                )
            )
    coordinate = torch.cat(chunks)
    baseline_h = torch.as_tensor(
        np.array(np.load(args.baseline_val, mmap_mode="r"), copy=True),
        dtype=torch.complex64,
        device=device,
    )
    if args.domain == "pas":
        baseline = pas_spectrum_phv(baseline_h, spec)
    else:
        baseline = pdp_spectrum(baseline_h, spec)
    baseline = stable_unit(
        baseline.reshape(len(val_idx), slice_count, output_dim)
    )
    truth = spectra[val_idx_t]

    def entropy(value: torch.Tensor) -> torch.Tensor:
        probability = value / value.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(value.dtype).tiny
        )
        return -(probability * probability.clamp_min(1e-30).log()).sum(-1)

    agreement = (baseline * coordinate).sum(-1)
    nearest = cKDTree(positions[train_idx, :2]).query(
        positions[val_idx, :2], k=1
    )[0].astype(np.float32)
    query_features = np.concatenate(
        [
            position_np[val_idx, :2],
            map_values[val_idx],
            nearest[:, None] / 5.0,
        ],
        axis=1,
    )
    query_feature_t = torch.from_numpy(query_features).to(device)
    query_feature_t = query_feature_t[:, None].expand(-1, slice_count, -1)
    slice_feature_t = slice_coordinate[None].expand(len(val_idx), -1, -1)
    gate_features = torch.cat(
        [
            agreement[..., None],
            baseline.amax(-1, keepdim=True),
            coordinate.amax(-1, keepdim=True),
            entropy(baseline)[..., None],
            entropy(coordinate)[..., None],
            query_feature_t,
            slice_feature_t,
        ],
        dim=-1,
    )

    betas = torch.tensor(
        [0.0, 0.1, 0.2, 0.3, 0.5, 1.0], device=device
    )
    blended = stable_unit(
        (1.0 - betas[:, None, None, None]) * baseline[None]
        + betas[:, None, None, None] * coordinate[None]
    )
    score_grid = (blended * truth[None]).sum(-1)
    rng = np.random.default_rng(20260729)
    order = rng.permutation(len(val_idx))
    train_rows, test_rows = order[:100], order[100:]
    fit_rows, stop_rows = train_rows[:80], train_rows[80:]
    predicted_betas = []
    stop_scores = []
    with torch.enable_grad():
        for seed in (7, 19, 43):
            gate, center, scale, stop_score = fit_gate(
                gate_features,
                baseline,
                coordinate,
                truth,
                fit_rows,
                stop_rows,
                seed,
            )
            with torch.inference_mode():
                predicted_betas.append(
                    gate(
                        (gate_features[test_rows] - center) / scale
                    )
                )
            stop_scores.append(stop_score)
    predicted_beta = torch.stack(predicted_betas).mean(dim=0)

    baseline_test = baseline[test_rows]
    coordinate_test = coordinate[test_rows]
    truth_test = truth[test_rows]
    rows = []
    for shrink in (0.25, 0.5, 0.75, 1.0):
        beta = predicted_beta * shrink
        prediction = stable_unit(
            (1.0 - beta[..., None]) * baseline_test
            + beta[..., None] * coordinate_test
        )
        rows.append(
            {
                "gate_shrink": shrink,
                "score": float((prediction * truth_test).sum(-1).mean()),
                "mean_beta": float(beta.mean()),
            }
        )
    fixed_rows = []
    for beta in betas:
        prediction = stable_unit(
            (1.0 - beta) * baseline_test + beta * coordinate_test
        )
        fixed_rows.append(
            {
                "beta": float(beta),
                "score": float((prediction * truth_test).sum(-1).mean()),
            }
        )
    oracle = float(score_grid[:, test_rows].amax(dim=0).mean())
    result = {
        "domain": args.domain,
        "split_seed": args.split_seed,
        "gate_train_queries": train_rows.tolist(),
        "gate_fit_queries": fit_rows.tolist(),
        "gate_stop_queries": stop_rows.tolist(),
        "gate_test_queries": test_rows.tolist(),
        "gate_stop_scores": stop_scores,
        "fixed_rows": fixed_rows,
        "gate_rows": rows,
        "oracle_slice_grid": oracle,
        "best_fixed": max(fixed_rows, key=lambda row: row["score"]),
        "best_gate": max(rows, key=lambda row: row["score"]),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"COORDINATE_GATE_DONE out={out}", flush=True)


if __name__ == "__main__":
    main()
