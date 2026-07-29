#!/usr/bin/env python3
"""Learn a clean global PDP anchor and use it to align neighbor spectra."""
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
    scaled = torch.where(
        maximum > 0,
        value / maximum.clamp_min(torch.finfo(value.dtype).tiny),
        torch.zeros_like(value),
    )
    return scaled / scaled.norm(dim=-1, keepdim=True).clamp_min(
        torch.finfo(value.dtype).tiny
    )


class AnchorRefiner(nn.Module):
    def __init__(self, input_dim: int, hidden: int, bins: int) -> None:
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
                for _ in range(4)
            ]
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, bins),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(
        self, feature: torch.Tensor, baseline: torch.Tensor
    ) -> torch.Tensor:
        hidden = F.silu(self.input(feature))
        for block in self.blocks:
            hidden = hidden + block(hidden)
        delta = self.output(hidden)
        prediction = F.relu(baseline + 0.1 * delta)
        return stable_unit(prediction)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", default="Round1_Map(2)")
    parser.add_argument(
        "--cache", default="cache/teammate_knn_hvp/train_pdp.npy"
    )
    parser.add_argument("--split-seed", type=int, default=1890)
    parser.add_argument("--baseline-val", required=True)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--distance-power", type=float, default=2.0)
    parser.add_argument("--max-shift", type=int, default=12)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--beta-grid", default="0,0.1,0.25,0.5,0.75,1"
    )
    parser.add_argument(
        "--out", default="docs/global_pdp_anchor_s1890/result.json"
    )
    return parser.parse_args()


def neighbor_table(
    positions: np.ndarray,
    query_idx: np.ndarray,
    pool_idx: np.ndarray,
    k: int,
    exclude_self: bool,
) -> tuple[np.ndarray, np.ndarray]:
    extra = 1 if exclude_self else 0
    distance, local = cKDTree(positions[pool_idx, :2]).query(
        positions[query_idx, :2], k=k + extra
    )
    selected = pool_idx[local]
    if not exclude_self:
        return distance.astype(np.float32), selected.astype(np.int64)
    output_distance = np.empty((len(query_idx), k), dtype=np.float32)
    output_index = np.empty((len(query_idx), k), dtype=np.int64)
    for row, source in enumerate(query_idx):
        keep = selected[row] != source
        output_distance[row] = distance[row][keep][:k]
        output_index[row] = selected[row][keep][:k]
    return output_distance, output_index


def aggregate_features(
    profile: torch.Tensor,
    neighbors: np.ndarray,
    distance: np.ndarray,
    power: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    index = torch.as_tensor(neighbors, dtype=torch.long, device=device)
    candidate = profile[index]
    distance_t = torch.as_tensor(
        distance, dtype=torch.float32, device=device
    ).clamp_min(0.3)
    weight = distance_t.pow(-power)
    weight = weight / weight.sum(dim=1, keepdim=True)
    anchor = stable_unit((weight[..., None] * candidate).sum(dim=1))
    nearest = candidate[:, 0]
    spread = candidate.std(dim=1)
    feature = torch.cat(
        [anchor, nearest, nearest - anchor, spread], dim=-1
    )
    return anchor, feature


def aligned_prediction(
    raw_pdp: np.ndarray,
    truth_idx: np.ndarray,
    neighbors: np.ndarray,
    distance: np.ndarray,
    anchor: torch.Tensor,
    baseline: torch.Tensor,
    k: int,
    power: float,
    max_shift: int,
    betas: list[float],
    device: torch.device,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    totals = torch.zeros(len(betas), device=device)
    baseline_total = 0.0
    aligned_total = 0.0
    count = 0
    shifts = torch.arange(-max_shift, max_shift + 1, device=device)
    delay = torch.arange(raw_pdp.shape[-1], device=device)
    for start in range(0, len(truth_idx), 2):
        stop = min(start + 2, len(truth_idx))
        truth = stable_unit(
            torch.as_tensor(
                np.array(raw_pdp[truth_idx[start:stop]], copy=True),
                dtype=torch.float32,
                device=device,
            )
        )
        candidate_raw = torch.as_tensor(
            np.array(raw_pdp[neighbors[start:stop, :k]], copy=True),
            dtype=torch.float32,
            device=device,
        )
        candidate = stable_unit(candidate_raw)
        candidate_anchor = stable_unit(candidate_raw.sum(dim=(2, 3)))
        target_anchor = anchor[start:stop]
        shift_score = torch.stack(
            [
                (
                    torch.roll(candidate_anchor, int(shift), dims=-1)
                    * target_anchor[:, None]
                ).sum(dim=-1)
                for shift in shifts
            ],
            dim=-1,
        )
        selected_shift = shifts[shift_score.argmax(dim=-1)]
        source = torch.remainder(
            delay[None, None, None, None, :]
            - selected_shift[..., None, None, None],
            raw_pdp.shape[-1],
        )
        aligned_candidate = torch.gather(
            candidate,
            -1,
            source.expand(
                -1,
                -1,
                candidate.shape[2],
                candidate.shape[3],
                -1,
            ),
        )
        weight = torch.as_tensor(
            distance[start:stop, :k],
            dtype=torch.float32,
            device=device,
        ).clamp_min(0.3).pow(-power)
        weight = weight / weight.sum(dim=1, keepdim=True)
        aligned = stable_unit(
            (weight[..., None, None, None] * aligned_candidate).sum(dim=1)
        )
        base = baseline[start:stop]
        baseline_total += float((base * truth).sum())
        aligned_total += float((aligned * truth).sum())
        for beta_index, beta in enumerate(betas):
            prediction = stable_unit(
                (1.0 - beta) * base + beta * aligned
            )
            totals[beta_index] += (prediction * truth).sum()
        count += truth.numel() // truth.shape[-1]
    rows = [
        {"beta": beta, "score": float(total / count)}
        for beta, total in zip(betas, totals)
    ]
    diagnostic = {
        "baseline": baseline_total / count,
        "aligned": aligned_total / count,
    }
    return rows, diagnostic


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    datadir = Path(args.datadir)
    positions = np.load(
        datadir / "Round1_Train_Pos.npy"
    ).astype(np.float32)
    spec = load_setup(datadir / "Round1_Setup.json")
    val_idx = np.asarray(
        sorted(reproduce_val_indices(len(positions), 0.1, args.split_seed)),
        dtype=np.int64,
    )
    train_idx = np.setdiff1d(
        np.arange(len(positions), dtype=np.int64), val_idx
    )
    train_distance, train_neighbors = neighbor_table(
        positions, train_idx, train_idx, args.k, True
    )
    val_distance, val_neighbors = neighbor_table(
        positions, val_idx, train_idx, args.k, False
    )

    raw_pdp = np.load(args.cache, mmap_mode="r")
    aggregate_chunks = []
    for start in range(0, len(positions), 100):
        raw = torch.as_tensor(
            np.array(raw_pdp[start : start + 100], copy=True),
            dtype=torch.float32,
            device=device,
        )
        aggregate_chunks.append(stable_unit(raw.sum(dim=(1, 2))).cpu())
    aggregate = torch.cat(aggregate_chunks).to(device)
    train_anchor, train_local_feature = aggregate_features(
        aggregate,
        train_neighbors,
        train_distance,
        args.distance_power,
        device,
    )
    val_anchor, val_local_feature = aggregate_features(
        aggregate,
        val_neighbors,
        val_distance,
        args.distance_power,
        device,
    )
    train_tensor = torch.as_tensor(
        train_idx, dtype=torch.long, device=device
    )
    val_tensor = torch.as_tensor(val_idx, dtype=torch.long, device=device)
    target_train = aggregate[train_tensor]
    target_val = aggregate[val_tensor]
    anchor_score = float((val_anchor * target_val).sum(dim=-1).mean())

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
    map_value, map_center, map_scale = robust_standardize(
        raw_map[train_idx], raw_map
    )
    position_value, position_meta = position_features(
        positions, map_value, train_idx
    )
    position_t = torch.as_tensor(
        position_value, dtype=torch.float32, device=device
    )
    train_feature = torch.cat(
        [position_t[train_tensor], train_local_feature], dim=-1
    )
    val_feature = torch.cat(
        [position_t[val_tensor], val_local_feature], dim=-1
    )

    baseline_channel = torch.as_tensor(
        np.array(np.load(args.baseline_val, mmap_mode="r"), copy=True),
        dtype=torch.complex64,
        device=device,
    )
    baseline = stable_unit(pdp_spectrum(baseline_channel, spec))
    del baseline_channel
    betas = [
        float(value) for value in args.beta_grid.split(",") if value.strip()
    ]

    model = AnchorRefiner(
        train_feature.shape[-1], args.hidden, spec.s
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05
    )
    best_score = -1.0
    best_payload = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train_idx), device=device)
        losses = []
        for start in range(0, len(order), args.batch_size):
            rows = order[start : start + args.batch_size]
            prediction = model(
                train_feature[rows], train_anchor[rows]
            )
            target = target_train[rows]
            cosine_loss = 1.0 - (prediction * target).sum(dim=-1).mean()
            prediction_l1 = prediction / prediction.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-12)
            target_l1 = target / target.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-12)
            distribution_loss = -(
                target_l1 * prediction_l1.clamp_min(1e-8).log()
            ).sum(dim=-1).mean()
            loss = cosine_loss + 0.02 * distribution_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss))
        scheduler.step()

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            model.eval()
            with torch.inference_mode():
                predicted_anchor = model(val_feature, val_anchor)
                profile_score = float(
                    (predicted_anchor * target_val).sum(dim=-1).mean()
                )
                rows, diagnostic = aligned_prediction(
                    raw_pdp,
                    val_idx,
                    val_neighbors,
                    val_distance,
                    predicted_anchor,
                    baseline,
                    args.k,
                    args.distance_power,
                    args.max_shift,
                    betas,
                    device,
                )
                selected = max(rows, key=lambda row: row["score"])
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(np.mean(losses)),
                    "profile_score": profile_score,
                    "aligned_score": diagnostic["aligned"],
                    "selected": selected,
                }
            )
            if selected["score"] > best_score:
                best_score = selected["score"]
                best_payload = {
                    "model_state": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "epoch": epoch,
                    "profile_score": profile_score,
                    "rows": rows,
                    "diagnostic": diagnostic,
                    "selected": selected,
                }
            print(
                f"[pdp-anchor] epoch={epoch} "
                f"profile={profile_score:.6f} "
                f"aligned={diagnostic['aligned']:.6f} "
                f"blend={selected['score']:.6f}@{selected['beta']}",
                flush=True,
            )

    if best_payload is None:
        raise RuntimeError("no PDP anchor checkpoint selected")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = out.with_suffix(".pt")
    torch.save(
        {
            **best_payload,
            "input_dim": train_feature.shape[-1],
            "hidden": args.hidden,
            "k": args.k,
            "distance_power": args.distance_power,
            "max_shift": args.max_shift,
            "position_meta": position_meta,
            "map_names": map_names,
            "map_center": map_center.tolist(),
            "map_scale": map_scale.tolist(),
        },
        checkpoint,
    )
    result = {
        "split_seed": args.split_seed,
        "anchor_baseline_score": anchor_score,
        "best_epoch": best_payload["epoch"],
        "profile_score": best_payload["profile_score"],
        "diagnostic": best_payload["diagnostic"],
        "selected": best_payload["selected"],
        "rows": best_payload["rows"],
        "history": history,
        "checkpoint": str(checkpoint),
    }
    out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)
    print(f"GLOBAL_PDP_ANCHOR_DONE out={out}", flush=True)


if __name__ == "__main__":
    main()
