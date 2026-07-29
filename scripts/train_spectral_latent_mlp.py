#!/usr/bin/env python3
"""Train a clean position-to-spectrum latent model and test complementarity."""
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
from wireless_twin.data.map_loader import load_point_cloud
from wireless_twin.data.setup_config import load_setup
from wireless_twin.models.raytrace2 import build_heightmap
from wireless_twin.signal import pas_spectrum_phv, pdp_spectrum


class LatentMLP(nn.Module):
    def __init__(self, input_dim: int, rank: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, rank),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", default="Round1_Map(2)")
    parser.add_argument("--cache-dir", default="cache/teammate_knn_hvp")
    parser.add_argument("--domain", choices=("pas", "pdp"), default="pas")
    parser.add_argument("--split-seed", type=int, default=1890)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument(
        "--schedule-epochs",
        type=int,
        default=None,
        help="cosine schedule horizon; defaults to --epochs",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--pca-iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--baseline-val", required=True)
    parser.add_argument("--fixed-beta", type=float, default=None)
    parser.add_argument(
        "--select-final-only",
        action="store_true",
        help="freeze model selection to the requested final epoch",
    )
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def position_features(
    positions: np.ndarray,
    map_values: np.ndarray,
    train_idx: np.ndarray,
) -> tuple[np.ndarray, dict]:
    center = positions[train_idx, :2].mean(axis=0)
    scale = positions[train_idx, :2].std(axis=0).clip(1e-3)
    xy = (positions[:, :2] - center) / scale
    columns = [xy]
    for frequency in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        angle = np.pi * frequency * xy
        columns.extend([np.sin(angle), np.cos(angle)])
    columns.append(map_values)
    value = np.concatenate(columns, axis=1).astype(np.float32)
    return value, {
        "position_center": center.tolist(),
        "position_scale": scale.tolist(),
        "input_dim": value.shape[1],
    }


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
        sorted(reproduce_val_indices(len(positions), 0.1, args.split_seed)),
        dtype=np.int64,
    )
    train_idx = np.setdiff1d(
        np.arange(len(positions), dtype=np.int64), val_idx
    )

    cache_dir = Path(args.cache_dir)
    cache_name = (
        "train_pas_phv.npy" if args.domain == "pas" else "train_pdp.npy"
    )
    spectra_np = np.load(cache_dir / cache_name, mmap_mode="r")
    spectra = torch.as_tensor(
        np.array(spectra_np, copy=True),
        dtype=torch.float32,
        device=device,
    )
    spectra = spectra / spectra.norm(
        dim=-1, keepdim=True
    ).clamp_min(torch.finfo(torch.float32).tiny)
    original_shape = tuple(spectra.shape[1:])
    flat = spectra.reshape(len(spectra), -1)
    train_tensor = torch.as_tensor(
        train_idx, dtype=torch.long, device=device
    )
    val_tensor = torch.as_tensor(
        val_idx, dtype=torch.long, device=device
    )

    print(
        f"[latent] PCA domain={args.domain} rank={args.rank} "
        f"matrix={len(train_idx)}x{flat.shape[1]}",
        flush=True,
    )
    train_flat = flat[train_tensor]
    spectrum_mean = train_flat.mean(dim=0)
    centered = train_flat - spectrum_mean
    _, singular, components = torch.pca_lowrank(
        centered,
        q=args.rank,
        center=False,
        niter=args.pca_iterations,
    )
    train_coefficients = centered @ components
    coefficient_scale = train_coefficients.std(
        dim=0
    ).clamp_min(1e-6)
    target_coefficients = train_coefficients / coefficient_scale
    val_projection = spectrum_mean + (
        (flat[val_tensor] - spectrum_mean) @ components
    ) @ components.T
    val_projection = val_projection.clamp_min(0).reshape(
        len(val_idx), *original_shape
    )
    val_projection = val_projection / val_projection.norm(
        dim=-1, keepdim=True
    ).clamp_min(torch.finfo(torch.float32).tiny)
    pca_upper = float(
        (val_projection * spectra[val_tensor]).sum(dim=-1).mean()
    )
    print(f"[latent] PCA holdout upper={pca_upper:.6f}", flush=True)

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
    feature_np, feature_meta = position_features(
        positions, map_values, train_idx
    )
    feature = torch.as_tensor(
        feature_np, dtype=torch.float32, device=device
    )

    baseline_channel = np.load(args.baseline_val, mmap_mode="r")
    if len(baseline_channel) != len(val_idx):
        raise ValueError("baseline validation prediction length mismatch")
    baseline_h = torch.as_tensor(
        np.array(baseline_channel, copy=True),
        dtype=torch.complex64,
        device=device,
    )
    if args.domain == "pas":
        baseline = pas_spectrum_phv(baseline_h, spec)
    else:
        baseline = pdp_spectrum(baseline_h, spec)
    baseline = baseline / baseline.norm(
        dim=-1, keepdim=True
    ).clamp_min(torch.finfo(torch.float32).tiny)
    baseline_score = float(
        (baseline * spectra[val_tensor]).sum(dim=-1).mean()
    )
    print(f"[latent] baseline={baseline_score:.6f}", flush=True)

    model = LatentMLP(
        feature.shape[1], args.rank, args.hidden
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.schedule_epochs or args.epochs,
        eta_min=args.lr * 0.05,
    )
    best_blend = -1.0
    best_payload = None
    beta_grid = (
        (args.fixed_beta,)
        if args.fixed_beta is not None
        else (0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = train_tensor[
            torch.randperm(len(train_tensor), device=device)
        ]
        losses = []
        for start in range(0, len(order), args.batch_size):
            index = order[start : start + args.batch_size]
            rows = torch.searchsorted(train_tensor, index)
            prediction = model(feature[index])
            loss = F.smooth_l1_loss(
                prediction, target_coefficients[rows], beta=0.5
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss))
        scheduler.step()

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            model.eval()
            with torch.inference_mode():
                predicted_coefficients = (
                    model(feature[val_tensor]) * coefficient_scale
                )
                prediction_flat = (
                    spectrum_mean
                    + predicted_coefficients @ components.T
                ).clamp_min(0)
                prediction = prediction_flat.reshape(
                    len(val_idx), *original_shape
                )
                prediction = prediction / prediction.norm(
                    dim=-1, keepdim=True
                ).clamp_min(torch.finfo(torch.float32).tiny)
                latent_score = float(
                    (prediction * spectra[val_tensor]).sum(dim=-1).mean()
                )
                rows = []
                for beta in beta_grid:
                    blended = (
                        (1.0 - beta) * baseline + beta * prediction
                    )
                    blended = blended / blended.norm(
                        dim=-1, keepdim=True
                    ).clamp_min(torch.finfo(torch.float32).tiny)
                    score = float(
                        (blended * spectra[val_tensor])
                        .sum(dim=-1)
                        .mean()
                    )
                    rows.append({"beta": beta, "score": score})
                selected = max(rows, key=lambda row: row["score"])
            eligible = not args.select_final_only or epoch == args.epochs
            if eligible and selected["score"] > best_blend:
                best_blend = selected["score"]
                best_payload = {
                    "model_state": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "epoch": epoch,
                    "latent_score": latent_score,
                    "blend_rows": rows,
                    "selected": selected,
                }
            print(
                f"[latent] epoch={epoch} loss={np.mean(losses):.6f} "
                f"latent={latent_score:.6f} "
                f"blend={selected['score']:.6f}@{selected['beta']}",
                flush=True,
            )

    if best_payload is None:
        raise RuntimeError("no model checkpoint selected")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = out.with_suffix(".pt")
    torch.save(
        {
            **best_payload,
            "domain": args.domain,
            "split_seed": args.split_seed,
            "rank": args.rank,
            "hidden": args.hidden,
            "components": components.cpu(),
            "spectrum_mean": spectrum_mean.cpu(),
            "coefficient_scale": coefficient_scale.cpu(),
            "feature_meta": feature_meta,
            "map_names": map_names,
            "map_center": map_center.tolist(),
            "map_scale": map_scale.tolist(),
        },
        checkpoint,
    )
    result = {
        "domain": args.domain,
        "split_seed": args.split_seed,
        "rank": args.rank,
        "pca_upper": pca_upper,
        "baseline_score": baseline_score,
        "best_epoch": best_payload["epoch"],
        "latent_score": best_payload["latent_score"],
        "blend_rows": best_payload["blend_rows"],
        "selected": best_payload["selected"],
        "blend_gain": best_payload["selected"]["score"] - baseline_score,
        "checkpoint": str(checkpoint),
        "singular_values": singular.cpu().tolist(),
    }
    out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"LATENT_MLP_DONE gain={result['blend_gain']:.6f} out={out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
