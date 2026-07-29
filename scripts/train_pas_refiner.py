#!/usr/bin/env python3
"""Train a clean PHV PAS residual refiner on one representative split."""
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
from wireless_twin.models.spectrum_refiner import PasSpectrumRefiner
from wireless_twin.signal import pas_spectrum_phv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", default="Round1_Map(2)")
    parser.add_argument("--split-seed", type=int, default=1890)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--query-batch", type=int, default=32)
    parser.add_argument("--slices-per-query", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    datadir = Path(args.datadir)
    spec = load_setup(datadir / "Round1_Setup.json")
    positions = np.load(datadir / "Round1_Train_Pos.npy").astype(np.float32)
    channels = np.load(
        datadir / "Round1_Train_Channel.npy", mmap_mode="r"
    )
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
    print("[refiner] caching PHV PAS", flush=True)
    pas = torch.empty(
        len(positions),
        spec.n * spec.s,
        spec.mh * spec.mv,
        dtype=torch.float32,
        device=device,
    )
    for start in range(0, len(positions), 50):
        stop = min(start + 50, len(positions))
        value = torch.from_numpy(
            np.array(channels[start:stop], copy=True)
        ).to(device)
        spectrum = pas_spectrum_phv(value, spec)
        spectrum = spectrum / spectrum.norm(
            dim=-1, keepdim=True
        ).clamp_min(torch.finfo(torch.float32).tiny)
        pas[start:stop] = spectrum.reshape(
            stop - start, spec.n * spec.s, spec.mh * spec.mv
        )
        del value, spectrum

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

    model = PasSpectrumRefiner(args.neighbors).to(device)
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
    slice_count = spec.n * spec.s
    best = -1.0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(train_idx), device=device)
        for start in range(0, len(train_idx), args.query_batch):
            rows = order[start : start + args.query_batch]
            batch = len(rows)
            slices = torch.randint(
                slice_count,
                (batch, args.slices_per_query),
                device=device,
            )
            neighbor = pas[
                train_neighbors_t[rows, :, None],
                slices[:, None, :],
            ]
            target = pas[train_idx_t[rows, None], slices]
            neighbor = neighbor.permute(0, 2, 1, 3).reshape(
                -1, args.neighbors, spec.mh, spec.mv
            )
            target = target.reshape(-1, spec.mh * spec.mv)
            distance = train_distance_t[rows, None, :].expand(
                -1, args.slices_per_query, -1
            ).reshape(-1, args.neighbors)
            geo = train_geometry[rows, None, :].expand(
                -1, args.slices_per_query, -1
            ).reshape(
                -1, train_geometry.shape[-1]
            )
            prediction = model(neighbor, distance, geo).reshape(
                -1, spec.mh * spec.mv
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
            for query_start in range(0, len(val_idx), 4):
                query_stop = min(query_start + 4, len(val_idx))
                rows = torch.arange(
                    query_start, query_stop, device=device
                )
                for slice_start in range(0, slice_count, 96):
                    slice_stop = min(slice_start + 96, slice_count)
                    slices = torch.arange(
                        slice_start, slice_stop, device=device
                    )
                    neighbor = pas[
                        val_neighbors_t[rows, :, None],
                        slices[None, None, :],
                    ]
                    target = pas[
                        val_idx_t[rows, None], slices[None, :]
                    ]
                    neighbor = neighbor.permute(0, 2, 1, 3).reshape(
                        -1, args.neighbors, spec.mh, spec.mv
                    )
                    target = target.reshape(-1, spec.mh * spec.mv)
                    distance = val_distance_t[rows, None, :].expand(
                        -1, len(slices), -1
                    ).reshape(-1, args.neighbors)
                    geo = val_geometry[rows, None, :].expand(
                        -1, len(slices), -1
                    ).reshape(-1, val_geometry.shape[-1])
                    prediction = model(
                        neighbor, distance, geo
                    ).reshape(-1, spec.mh * spec.mv)
                    total += float(
                        F.cosine_similarity(
                            prediction, target, dim=-1
                        ).sum()
                    )
                    count += len(target)
        score = total / count
        if score > best:
            best = score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "meta": {
                        "class": "PasSpectrumRefiner",
                        "pas_layout": "phv",
                        "split_seed": args.split_seed,
                        "clean_holdout": True,
                        "neighbors": args.neighbors,
                        "hidden": 32,
                        "best_pas": best,
                        "epoch": epoch + 1,
                    },
                },
                out,
            )
        print(
            f"[refiner] epoch={epoch + 1} val_pas={score:.6f} "
            f"best={best:.6f}",
            flush=True,
        )
    (out.with_suffix(".json")).write_text(
        json.dumps(
            {
                "split_seed": args.split_seed,
                "neighbors": args.neighbors,
                "best_pas": best,
                "checkpoint": str(out),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"PAS_REFINER_DONE best={best:.6f} out={out}", flush=True)


if __name__ == "__main__":
    main()
