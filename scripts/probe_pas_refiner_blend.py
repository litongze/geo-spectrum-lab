#!/usr/bin/env python3
"""Check whether a PHV 2D refiner complements the clean attention PAS arm."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

from _bootstrap import ROOT  # noqa: F401
from probe_clean_panel_noeps import AttnWide
from score_holdout import reproduce_val_indices
from wireless_twin.data.map_loader import load_point_cloud
from wireless_twin.data.setup_config import load_setup
from wireless_twin.models.raytrace2 import build_heightmap
from wireless_twin.models.spectrum_refiner import PasSpectrumRefiner
from wireless_twin.signal import pas_spectrum_phv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", default="Round1_Map(2)")
    parser.add_argument("--split-seed", type=int, default=1890)
    parser.add_argument("--attention", required=True)
    parser.add_argument("--refiner", required=True)
    parser.add_argument("--beta-grid", default="0,0.05,0.1,0.2,0.3,0.5,1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    datadir = Path(args.datadir)
    spec = load_setup(datadir / "Round1_Setup.json")
    positions = np.load(datadir / "Round1_Train_Pos.npy").astype(np.float32)
    channels = np.load(
        datadir / "Round1_Train_Channel.npy", mmap_mode="r"
    )
    val_idx = np.asarray(
        sorted(reproduce_val_indices(len(positions), 0.1, args.split_seed)),
        dtype=np.int64,
    )
    train_idx = np.setdiff1d(
        np.arange(len(positions), dtype=np.int64), val_idx
    )

    pas = torch.empty(
        len(positions),
        spec.n * spec.s,
        spec.mh * spec.mv,
        dtype=torch.float32,
        device=device,
    )
    for start in range(0, len(positions), 50):
        stop = min(start + 50, len(positions))
        value = torch.as_tensor(
            np.array(channels[start:stop], copy=True),
            dtype=torch.complex64,
            device=device,
        )
        spectrum = pas_spectrum_phv(value, spec).reshape(
            stop - start, spec.n * spec.s, spec.mh * spec.mv
        )
        pas[start:stop] = spectrum / spectrum.norm(
            dim=-1, keepdim=True
        ).clamp_min(torch.finfo(torch.float32).tiny)
        del value, spectrum

    points = load_point_cloud(datadir / "Round1_Map.ply")
    heightmap, x0, y0, resolution = build_heightmap(points)

    def indoor_flags(target: np.ndarray) -> np.ndarray:
        gx = np.clip(
            np.floor((target[:, 0] - x0) / resolution).astype(int),
            0,
            heightmap.shape[0] - 1,
        )
        gy = np.clip(
            np.floor((target[:, 1] - y0) / resolution).astype(int),
            0,
            heightmap.shape[1] - 1,
        )
        return (heightmap[gx, gy] > 2.0).astype(np.float32)

    indoor = indoor_flags(positions)
    attention_payload = torch.load(
        args.attention, map_location=device, weights_only=False
    )
    attention_meta = attention_payload["meta"]
    attention_k = int(attention_meta["K"])
    attention = AttnWide(int(attention_meta["feature_dim"])).to(device)
    attention.load_state_dict(attention_payload["model_state"])
    attention.eval()

    refiner_payload = torch.load(
        args.refiner, map_location=device, weights_only=False
    )
    refiner_meta = refiner_payload["meta"]
    refiner_k = int(refiner_meta["neighbors"])
    refiner = PasSpectrumRefiner(refiner_k).to(device)
    refiner.load_state_dict(refiner_payload["model_state"])
    refiner.eval()

    max_k = max(attention_k, refiner_k)
    tree = cKDTree(positions[train_idx, :2])
    distance, local_idx = tree.query(positions[val_idx, :2], k=max_k)
    distance = distance.astype(np.float32)
    neighbors = train_idx[local_idx]
    bs_xy = np.asarray(spec.bs_position[:2], dtype=np.float32)

    def geometry(k: int) -> torch.Tensor:
        target_xy = positions[val_idx, :2]
        neighbor_xy = positions[neighbors[:, :k], :2]
        delta = neighbor_xy - target_xy[:, None, :]
        radial = target_xy - bs_xy
        radius = np.linalg.norm(radial, axis=1).clip(1e-3)
        angle = np.arctan2(radial[:, 1], radial[:, 0])
        return torch.from_numpy(
            np.concatenate(
                [
                    delta[..., 0] / 5.0,
                    delta[..., 1] / 5.0,
                    distance[:, :k] / 5.0,
                    (radius / 200.0)[:, None],
                    np.sin(angle)[:, None],
                    np.cos(angle)[:, None],
                ],
                axis=1,
            ).astype(np.float32)
        ).to(device)

    refiner_geometry = geometry(refiner_k)
    truth = pas[torch.as_tensor(val_idx, device=device)]
    attention_chunks = []
    refiner_chunks = []
    with torch.inference_mode():
        for start in range(0, len(val_idx), 10):
            stop = min(start + 10, len(val_idx))
            batch = stop - start
            target_xy = torch.as_tensor(
                positions[val_idx[start:stop], :2],
                dtype=torch.float32,
                device=device,
            )
            neighbor_idx = neighbors[start:stop, :attention_k]
            y = pas[
                torch.as_tensor(neighbor_idx, device=device)
            ]
            mean = y.mean(dim=1, keepdim=True)
            agree = F.cosine_similarity(y, mean, dim=-1)
            d = torch.as_tensor(
                distance[start:stop, :attention_k], device=device
            )[:, :, None].expand(-1, -1, y.shape[2])
            neighbor_indoor = torch.as_tensor(
                indoor[neighbor_idx], device=device
            )[:, :, None].expand(-1, -1, y.shape[2])
            target_indoor = torch.as_tensor(
                indoor[val_idx[start:stop]], device=device
            )[:, None, None].expand(-1, attention_k, y.shape[2])
            delta = torch.as_tensor(
                positions[neighbor_idx, :2], device=device
            ) - target_xy[:, None, :]
            radial = target_xy - torch.as_tensor(bs_xy, device=device)
            radius = radial.norm(dim=-1).clamp_min(1e-3)
            radial_unit = radial / radius[:, None]
            tangent_unit = torch.stack(
                [-radial_unit[:, 1], radial_unit[:, 0]], dim=-1
            )

            def expand(value: torch.Tensor) -> torch.Tensor:
                return value[:, :, None].expand(-1, -1, y.shape[2])

            angle = torch.atan2(radial[:, 1], radial[:, 0])
            features = torch.stack(
                [
                    d / 3.0,
                    (neighbor_indoor == target_indoor).float(),
                    agree,
                    agree.square(),
                    torch.ones_like(d),
                    (d < 2.5).float(),
                    expand(delta[..., 0] / 5.0),
                    expand(delta[..., 1] / 5.0),
                    expand((delta * radial_unit[:, None]).sum(-1) / 5.0),
                    expand((delta * tangent_unit[:, None]).sum(-1) / 5.0),
                    (radius / 200.0)[:, None, None].expand_as(d),
                    angle.sin()[:, None, None].expand_as(d),
                    angle.cos()[:, None, None].expand_as(d),
                    neighbor_indoor,
                    target_indoor,
                ],
                dim=-1,
            )
            log_distance = torch.log(
                torch.as_tensor(
                    distance[start:stop, :attention_k], device=device
                ).clamp_min(0.3)
            )[:, :, None]
            weight = attention(features, log_distance)
            attention_chunks.append((weight[..., None] * y).sum(dim=1))

            refiner_neighbors = pas[
                torch.as_tensor(
                    neighbors[start:stop, :refiner_k], device=device
                )
            ].permute(0, 2, 1, 3).reshape(
                batch * spec.n * spec.s,
                refiner_k,
                spec.mh,
                spec.mv,
            )
            refiner_distance = torch.as_tensor(
                distance[start:stop, :refiner_k], device=device
            )[:, None, :].expand(-1, spec.n * spec.s, -1).reshape(
                -1, refiner_k
            )
            refiner_geo = refiner_geometry[start:stop, None, :].expand(
                -1, spec.n * spec.s, -1
            ).reshape(-1, refiner_geometry.shape[-1])
            refiner_chunks.append(
                refiner(
                    refiner_neighbors, refiner_distance, refiner_geo
                ).reshape(batch, spec.n * spec.s, -1)
            )

    attention_prediction = torch.cat(attention_chunks)
    refiner_prediction = torch.cat(refiner_chunks)
    betas = [float(value) for value in args.beta_grid.split(",")]
    rows = []
    for beta in betas:
        prediction = (
            (1.0 - beta) * attention_prediction
            + beta * refiner_prediction
        )
        prediction = prediction / prediction.norm(
            dim=-1, keepdim=True
        ).clamp_min(torch.finfo(torch.float32).tiny)
        score = float(
            F.cosine_similarity(prediction, truth, dim=-1).mean()
        )
        rows.append({"beta": beta, "pas": score})
        print(f"[blend] beta={beta:.3f} PAS={score:.6f}", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "split_seed": args.split_seed,
                "attention": args.attention,
                "refiner": args.refiner,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"REFINER_BLEND_DONE output={output}", flush=True)


if __name__ == "__main__":
    main()
