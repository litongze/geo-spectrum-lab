#!/usr/bin/env python3
"""Train a shared coordinate-conditioned PAS/PDP spectrum predictor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from _bootstrap import ROOT  # noqa: F401
from score_holdout import reproduce_val_indices
from sweep_mapaware_spectrum_knn import map_features, robust_standardize
from train_spectral_latent_mlp import position_features
from wireless_twin.data.map_loader import load_point_cloud
from wireless_twin.data.setup_config import load_setup
from wireless_twin.models.raytrace2 import build_heightmap
from wireless_twin.signal import pas_spectrum_phv, pdp_spectrum


class CoordinateSpectrumMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: int) -> None:
        super().__init__()
        self.input = nn.Linear(input_dim, hidden)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden),
                    nn.Linear(hidden, hidden * 2),
                    nn.SiLU(),
                    nn.Linear(hidden * 2, hidden),
                )
                for _ in range(3)
            ]
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.input(value))
        for block in self.blocks:
            hidden = hidden + block(hidden)
        prediction = F.softplus(self.output(hidden))
        return prediction / prediction.norm(
            dim=-1, keepdim=True
        ).clamp_min(torch.finfo(prediction.dtype).tiny)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", default="Round1_Map(2)")
    parser.add_argument("--cache-dir", default="cache/teammate_knn_hvp")
    parser.add_argument("--domain", choices=("pas", "pdp"), default="pas")
    parser.add_argument("--split-seed", type=int, default=1890)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--query-batch", type=int, default=32)
    parser.add_argument("--slices-per-query", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--baseline-val", required=True)
    parser.add_argument(
        "--beta-grid", default="0,0.025,0.05,0.1,0.2,0.3,0.5,1"
    )
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def stable_unit(value: torch.Tensor) -> torch.Tensor:
    scale = value.amax(dim=-1, keepdim=True)
    scaled = torch.where(
        scale > 0,
        value / scale.clamp_min(torch.finfo(value.dtype).tiny),
        torch.zeros_like(value),
    )
    return scaled / scaled.norm(dim=-1, keepdim=True).clamp_min(
        torch.finfo(value.dtype).tiny
    )


def fourier_scalar(value: np.ndarray) -> np.ndarray:
    columns = [value[:, None]]
    for frequency in (1.0, 2.0, 4.0, 8.0, 16.0):
        angle = np.pi * frequency * value
        columns.extend([np.sin(angle)[:, None], np.cos(angle)[:, None]])
    return np.concatenate(columns, axis=1).astype(np.float32)


def slice_features(spec, domain: str) -> np.ndarray:
    if domain == "pas":
        ue, subcarrier = np.meshgrid(
            np.arange(spec.n), np.arange(spec.s), indexing="ij"
        )
        ue_one_hot = np.eye(spec.n, dtype=np.float32)[ue.reshape(-1)]
        frequency = (
            2.0 * subcarrier.reshape(-1) / max(spec.s - 1, 1) - 1.0
        )
        return np.concatenate(
            [ue_one_hot, fourier_scalar(frequency)], axis=1
        ).astype(np.float32)

    antenna, ue = np.meshgrid(
        np.arange(spec.m), np.arange(spec.n), indexing="ij"
    )
    antenna = antenna.reshape(-1)
    polarization = antenna // (spec.mh * spec.mv)
    spatial = antenna % (spec.mh * spec.mv)
    horizontal = spatial // spec.mv
    vertical = spatial % spec.mv

    def normalized(value: np.ndarray, size: int) -> np.ndarray:
        return 2.0 * value / max(size - 1, 1) - 1.0

    return np.concatenate(
        [
            np.eye(spec.mp, dtype=np.float32)[polarization],
            np.eye(spec.n, dtype=np.float32)[ue.reshape(-1)],
            fourier_scalar(normalized(horizontal, spec.mh)),
            fourier_scalar(normalized(vertical, spec.mv)),
        ],
        axis=1,
    ).astype(np.float32)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    datadir = Path(args.datadir)
    spec = load_setup(datadir / "Round1_Setup.json")
    positions = np.load(
        datadir / "Round1_Train_Pos.npy"
    ).astype(np.float32)
    val_idx = np.asarray(
        sorted(
            reproduce_val_indices(
                len(positions), 0.1, args.split_seed
            )
        ),
        dtype=np.int64,
    )
    train_idx = np.setdiff1d(
        np.arange(len(positions), dtype=np.int64), val_idx
    )
    train_idx_t = torch.from_numpy(train_idx).to(device)
    val_idx_t = torch.from_numpy(val_idx).to(device)

    cache_name = (
        "train_pas_phv.npy" if args.domain == "pas" else "train_pdp.npy"
    )
    raw = np.load(Path(args.cache_dir) / cache_name, mmap_mode="r")
    spectra = stable_unit(
        torch.as_tensor(
            np.array(raw, copy=True), dtype=torch.float32, device=device
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
    raw_map, map_names = map_features(
        positions,
        np.asarray(spec.bs_position, dtype=np.float32),
        heightmap,
        x0,
        y0,
        resolution,
        ray_samples=128,
    )
    map_values, map_center, map_scale = robust_standardize(
        raw_map[train_idx], raw_map
    )
    position_np, position_meta = position_features(
        positions, map_values, train_idx
    )
    position = torch.from_numpy(position_np).to(device)
    slice_np = slice_features(spec, args.domain)
    slice_coordinate = torch.from_numpy(slice_np).to(device)

    baseline_channel = np.load(args.baseline_val, mmap_mode="r")
    baseline_h = torch.as_tensor(
        np.array(baseline_channel, copy=True),
        dtype=torch.complex64,
        device=device,
    )
    if args.domain == "pas":
        baseline = pas_spectrum_phv(baseline_h, spec).reshape(
            len(val_idx), slice_count, output_dim
        )
    else:
        baseline = pdp_spectrum(baseline_h, spec).reshape(
            len(val_idx), slice_count, output_dim
        )
    baseline = stable_unit(baseline)
    truth_val = spectra[val_idx_t]
    baseline_score = float((baseline * truth_val).sum(dim=-1).mean())
    print(
        f"[coord-spectrum] domain={args.domain} slices={slice_count} "
        f"baseline={baseline_score:.6f}",
        flush=True,
    )

    model = CoordinateSpectrumMLP(
        position.shape[1] + slice_coordinate.shape[1],
        output_dim,
        args.hidden,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05
    )
    betas = [
        float(value) for value in args.beta_grid.split(",") if value.strip()
    ]
    best_score = -1.0
    best_payload = None
    sample_count = min(args.slices_per_query, slice_count)

    def feature_batch(
        query_indices: torch.Tensor, slices: torch.Tensor
    ) -> torch.Tensor:
        position_part = position[query_indices][:, None, :].expand(
            -1, slices.shape[1], -1
        )
        slice_part = slice_coordinate[slices]
        return torch.cat([position_part, slice_part], dim=-1).reshape(
            -1, position_part.shape[-1] + slice_part.shape[-1]
        )

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = train_idx_t[
            torch.randperm(len(train_idx_t), device=device)
        ]
        losses = []
        for start in range(0, len(order), args.query_batch):
            query = order[start : start + args.query_batch]
            slices = torch.randint(
                slice_count,
                (len(query), sample_count),
                device=device,
            )
            prediction = model(feature_batch(query, slices)).reshape(
                len(query), sample_count, output_dim
            )
            target = spectra[query[:, None], slices]
            loss = 1.0 - F.cosine_similarity(
                prediction, target, dim=-1
            ).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss))
        scheduler.step()

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            model.eval()
            prediction_chunks = []
            with torch.inference_mode():
                all_slices = torch.arange(slice_count, device=device)
                for start in range(0, len(val_idx), 4):
                    query = val_idx_t[start : start + 4]
                    slices = all_slices[None].expand(len(query), -1)
                    prediction_chunks.append(
                        model(feature_batch(query, slices)).reshape(
                            len(query), slice_count, output_dim
                        )
                    )
                coordinate_prediction = torch.cat(prediction_chunks)
                coordinate_score = float(
                    (coordinate_prediction * truth_val).sum(dim=-1).mean()
                )
                rows = []
                for beta in betas:
                    blended = stable_unit(
                        (1.0 - beta) * baseline
                        + beta * coordinate_prediction
                    )
                    score = float(
                        (blended * truth_val).sum(dim=-1).mean()
                    )
                    rows.append({"beta": beta, "score": score})
                selected = max(rows, key=lambda row: row["score"])
            if selected["score"] > best_score:
                best_score = selected["score"]
                best_payload = {
                    "model_state": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "epoch": epoch,
                    "coordinate_score": coordinate_score,
                    "blend_rows": rows,
                    "selected": selected,
                }
            print(
                f"[coord-spectrum] epoch={epoch} "
                f"loss={np.mean(losses):.6f} "
                f"raw={coordinate_score:.6f} "
                f"blend={selected['score']:.6f}@{selected['beta']}",
                flush=True,
            )

    if best_payload is None:
        raise RuntimeError("no model selected")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = out.with_suffix(".pt")
    torch.save(
        {
            **best_payload,
            "domain": args.domain,
            "split_seed": args.split_seed,
            "hidden": args.hidden,
            "input_dim": position.shape[1] + slice_coordinate.shape[1],
            "output_dim": output_dim,
            "position_meta": position_meta,
            "map_names": map_names,
            "map_center": map_center.tolist(),
            "map_scale": map_scale.tolist(),
        },
        checkpoint,
    )
    result = {
        "domain": args.domain,
        "split_seed": args.split_seed,
        "hidden": args.hidden,
        "baseline_score": baseline_score,
        "best_epoch": best_payload["epoch"],
        "coordinate_score": best_payload["coordinate_score"],
        "blend_rows": best_payload["blend_rows"],
        "selected": best_payload["selected"],
        "blend_gain": best_payload["selected"]["score"] - baseline_score,
        "checkpoint": str(checkpoint),
    }
    out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"COORDINATE_SPECTRUM_DONE gain={result['blend_gain']:.6f} "
        f"out={out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
