#!/usr/bin/env python3
"""Train a clean PHV-array PDP residual refiner on one representative split."""
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
from wireless_twin.data.setup_config import load_setup
from wireless_twin.models.spectrum_refiner import PdpSpectrumRefiner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", default="Round1_Map(2)")
    parser.add_argument(
        "--cache", default="cache/teammate_knn_hvp/train_pdp.npy"
    )
    parser.add_argument("--split-seed", type=int, default=1890)
    parser.add_argument("--neighbors", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--query-batch", type=int, default=8)
    parser.add_argument("--components-per-query", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=24)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def stable_unit_delay(value: torch.Tensor) -> torch.Tensor:
    scale = value.amax(dim=-1, keepdim=True)
    scaled = torch.where(
        scale > 0,
        value / scale.clamp_min(torch.finfo(value.dtype).tiny),
        torch.zeros_like(value),
    )
    return scaled / scaled.norm(dim=-1, keepdim=True).clamp_min(
        torch.finfo(value.dtype).tiny
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
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
    train_idx = np.setdiff1d(
        np.arange(len(positions), dtype=np.int64), val_idx
    )

    print("[pdp-refiner] loading and normalizing PDP cache", flush=True)
    raw = np.load(args.cache, mmap_mode="r")
    pdp = torch.from_numpy(np.array(raw, copy=True)).to(device)
    pdp = stable_unit_delay(pdp).reshape(
        len(positions), spec.mp, spec.mh, spec.mv, spec.n, spec.s
    )
    pdp = pdp.permute(0, 1, 4, 2, 3, 5).reshape(
        len(positions),
        spec.mp * spec.n,
        spec.mh,
        spec.mv,
        spec.s,
    )

    tree = cKDTree(positions[train_idx, :2])
    train_distance, train_local = tree.query(
        positions[train_idx, :2], k=args.neighbors + 1
    )
    train_neighbors = train_idx[train_local[:, 1:]]
    train_distance = train_distance[:, 1:].astype(np.float32)
    val_distance, val_local = tree.query(
        positions[val_idx, :2], k=args.neighbors
    )
    val_neighbors = train_idx[val_local]
    val_distance = val_distance.astype(np.float32)
    bs_xy = np.asarray(spec.bs_position[:2], dtype=np.float32)

    def geometry(
        target_positions: np.ndarray,
        neighbor_indices: np.ndarray,
        distance: np.ndarray,
    ) -> torch.Tensor:
        target_xy = target_positions[:, :2]
        delta = positions[neighbor_indices, :2] - target_xy[:, None, :]
        radial = target_xy - bs_xy
        radius = np.linalg.norm(radial, axis=1).clip(1e-3)
        angle = np.arctan2(radial[:, 1], radial[:, 0])
        features = np.concatenate(
            [
                delta[..., 0] / 5.0,
                delta[..., 1] / 5.0,
                distance / 5.0,
                (radius / 200.0)[:, None],
                np.sin(angle)[:, None],
                np.cos(angle)[:, None],
            ],
            axis=1,
        ).astype(np.float32)
        return torch.from_numpy(features).to(device)

    model = PdpSpectrumRefiner(args.neighbors, args.hidden).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-5
    )
    train_distance_t = torch.from_numpy(train_distance).to(device)
    val_distance_t = torch.from_numpy(val_distance).to(device)
    train_geometry = geometry(
        positions[train_idx], train_neighbors, train_distance
    )
    val_geometry = geometry(
        positions[val_idx], val_neighbors, val_distance
    )
    train_neighbors_t = torch.from_numpy(train_neighbors).to(device)
    val_neighbors_t = torch.from_numpy(val_neighbors).to(device)
    train_idx_t = torch.from_numpy(train_idx).to(device)
    val_idx_t = torch.from_numpy(val_idx).to(device)
    component_count = spec.mp * spec.n
    sample_count = min(args.components_per_query, component_count)

    def component_batch(
        target_indices: torch.Tensor,
        neighbor_indices: torch.Tensor,
        components: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_rows = torch.arange(
            len(target_indices), device=device
        ).repeat_interleave(components.shape[1])
        flat_components = components.reshape(-1)
        selected_neighbors = neighbor_indices[query_rows]
        neighbor = pdp[
            selected_neighbors,
            flat_components[:, None],
        ]
        target = pdp[
            target_indices[query_rows],
            flat_components,
        ]
        return neighbor, target

    best = -1.0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(train_idx), device=device)
        for start in range(0, len(train_idx), args.query_batch):
            rows = order[start : start + args.query_batch]
            batch = len(rows)
            components = torch.stack(
                [
                    torch.randperm(component_count, device=device)[
                        :sample_count
                    ]
                    for _ in range(batch)
                ]
            )
            neighbor, target = component_batch(
                train_idx_t[rows],
                train_neighbors_t[rows],
                components,
            )
            repeated_rows = torch.arange(
                batch, device=device
            ).repeat_interleave(sample_count)
            prediction = model(
                neighbor,
                train_distance_t[rows][repeated_rows],
                train_geometry[rows][repeated_rows],
            )
            loss = 1.0 - F.cosine_similarity(
                prediction, target, dim=-1
            ).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        total = 0.0
        count = 0
        with torch.inference_mode():
            for start in range(0, len(val_idx), 2):
                stop = min(start + 2, len(val_idx))
                rows = torch.arange(start, stop, device=device)
                components = torch.arange(
                    component_count, device=device
                )[None].expand(len(rows), -1)
                neighbor, target = component_batch(
                    val_idx_t[rows],
                    val_neighbors_t[rows],
                    components,
                )
                repeated_rows = torch.arange(
                    len(rows), device=device
                ).repeat_interleave(component_count)
                prediction = model(
                    neighbor,
                    val_distance_t[rows][repeated_rows],
                    val_geometry[rows][repeated_rows],
                )
                total += float(
                    F.cosine_similarity(
                        prediction, target, dim=-1
                    ).sum()
                )
                count += target.numel() // spec.s
        score = total / count
        if score > best:
            best = score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "meta": {
                        "class": "PdpSpectrumRefiner",
                        "array_layout": "phv",
                        "split_seed": args.split_seed,
                        "clean_holdout": True,
                        "neighbors": args.neighbors,
                        "hidden": args.hidden,
                        "best_pdp": best,
                        "epoch": epoch + 1,
                    },
                },
                out,
            )
        print(
            f"[pdp-refiner] epoch={epoch + 1} val_pdp={score:.6f} "
            f"best={best:.6f}",
            flush=True,
        )

    out.with_suffix(".json").write_text(
        json.dumps(
            {
                "split_seed": args.split_seed,
                "neighbors": args.neighbors,
                "hidden": args.hidden,
                "best_pdp": best,
                "checkpoint": str(out),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"PDP_REFINER_DONE best={best:.6f} out={out}", flush=True)


if __name__ == "__main__":
    main()
