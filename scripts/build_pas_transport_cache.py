#!/usr/bin/env python3
"""Build leakage-free physically transported PAS artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from _bootstrap import ROOT  # noqa: F401
from probe_full_array_steering_phase import array_steering_phase
from score_holdout import reproduce_val_indices
from sweep_moment_attention import stable_unit
from wireless_twin.data.setup_config import load_setup
from wireless_twin.signal import pas_spectrum_phv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", default="Round1_Map(2)")
    parser.add_argument(
        "--config", default="configs/gs40_transport_pas.json"
    )
    parser.add_argument("--panel", default="1890,3716,962,1022,2262")
    parser.add_argument("--external-indices")
    parser.add_argument("--external-name", default="testmatched")
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--distance-power", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--outdir", default="cache/pas_transport_k16_p2"
    )
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    steering = config["steering"]
    datadir = Path(args.datadir)
    spec = load_setup(datadir / "Round1_Setup.json")
    train_pos = np.load(
        datadir / "Round1_Train_Pos.npy"
    ).astype(np.float64)
    train_channels = np.load(
        datadir / "Round1_Train_Channel.npy", mmap_mode="r"
    )
    bs = np.asarray(spec.bs_position, dtype=np.float64)
    train_radius = np.linalg.norm(train_pos - bs[None], axis=1)
    train_unit = (train_pos - bs[None]) / np.maximum(
        train_radius[:, None], 1e-12
    )
    all_idx = np.arange(len(train_pos), dtype=np.int64)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    artifacts = {}

    def build(
        name: str,
        query_pos: np.ndarray,
        pool_idx: np.ndarray,
    ) -> Path:
        distance, local = cKDTree(train_pos[pool_idx, :2]).query(
            query_pos[:, :2], k=args.k
        )
        distance = np.asarray(distance, dtype=np.float64)
        neighbors = pool_idx[np.asarray(local)]
        query_radius = np.linalg.norm(query_pos - bs[None], axis=1)
        query_unit = (query_pos - bs[None]) / np.maximum(
            query_radius[:, None], 1e-12
        )
        delta_unit = query_unit[:, None] - train_unit[neighbors]
        result = np.empty(
            (
                len(query_pos),
                spec.n,
                spec.s,
                spec.mh * spec.mv,
            ),
            dtype=np.float32,
        )
        with torch.inference_mode():
            for start in range(0, len(query_pos), args.batch_size):
                stop = min(start + args.batch_size, len(query_pos))
                batch = stop - start
                source = torch.as_tensor(
                    np.array(
                        train_channels[
                            neighbors[start:stop].reshape(-1)
                        ],
                        copy=True,
                    ).reshape(
                        batch,
                        args.k,
                        spec.m,
                        spec.n,
                        spec.s,
                    ),
                    dtype=torch.complex64,
                    device=device,
                )
                steering_phase = torch.as_tensor(
                    array_steering_phase(
                        delta_unit[start:stop], spec, steering
                    ),
                    dtype=torch.float32,
                    device=device,
                )
                source *= torch.exp(1j * steering_phase[..., None])
                neighbor_pas = stable_unit(
                    pas_spectrum_phv(
                        source.reshape(
                            batch * args.k,
                            spec.m,
                            spec.n,
                            spec.s,
                        ),
                        spec,
                    )
                ).reshape(
                    batch,
                    args.k,
                    spec.n,
                    spec.s,
                    spec.mh * spec.mv,
                )
                weight = (
                    1.0
                    / np.maximum(distance[start:stop], 0.05)
                    ** args.distance_power
                )
                weight /= np.maximum(
                    weight.sum(axis=1, keepdims=True), 1e-30
                )
                prediction = stable_unit(
                    torch.einsum(
                        "bk,bknsa->bnsa",
                        torch.as_tensor(
                            weight,
                            dtype=torch.float32,
                            device=device,
                        ),
                        neighbor_pas,
                    )
                )
                result[start:stop] = prediction.cpu().numpy()
                del source, neighbor_pas, prediction
        path = outdir / f"pas_{name}.npy"
        np.save(path, result)
        print(
            f"[pas-transport-cache] name={name} "
            f"shape={result.shape} nearest={distance[:, 0].mean():.4f}",
            flush=True,
        )
        return path

    for seed in [
        int(value) for value in args.panel.split(",") if value
    ]:
        val_idx = np.asarray(
            sorted(
                reproduce_val_indices(
                    len(train_pos), 0.1, seed
                )
            ),
            dtype=np.int64,
        )
        pool_idx = np.setdiff1d(all_idx, val_idx)
        path = build(str(seed), train_pos[val_idx], pool_idx)
        artifacts[str(seed)] = str(path)

    if args.external_indices:
        val_idx = np.load(args.external_indices).astype(np.int64)
        pool_idx = np.setdiff1d(all_idx, val_idx)
        path = build(
            args.external_name, train_pos[val_idx], pool_idx
        )
        artifacts[args.external_name] = str(path)

    if args.include_test:
        test_pos = np.load(
            datadir / "Round1_Test_Pos.npy"
        ).astype(np.float64)
        path = build("test", test_pos, all_idx)
        artifacts["test"] = str(path)

    metadata = {
        "selection_policy": (
            "source-only physical transport; each validation pool excludes "
            "its complete holdout"
        ),
        "k": args.k,
        "distance_power": args.distance_power,
        "steering": steering,
        "artifacts": artifacts,
    }
    (outdir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"PAS_TRANSPORT_CACHE_DONE outdir={outdir}", flush=True)


if __name__ == "__main__":
    main()
