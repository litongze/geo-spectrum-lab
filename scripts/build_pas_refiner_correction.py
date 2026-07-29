#!/usr/bin/env python3
"""Apply an ensemble PHV PAS refiner correction to a frozen backbone."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from _bootstrap import ROOT  # noqa: F401
from wireless_twin.data.setup_config import load_setup
from wireless_twin.models.spectrum_refiner import PasSpectrumRefiner
from wireless_twin.signal import pas_spectrum_phv, pdp_spectrum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", default="Round1_Map(2)")
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--checkpoint-pattern", required=True)
    parser.add_argument("--seeds", default="1890,3716,962,1022,2262")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--outdir", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(value: torch.Tensor, dim: int) -> torch.Tensor:
    return value / value.norm(dim=dim, keepdim=True).clamp_min(
        torch.finfo(torch.float32).tiny
    )


def main() -> None:
    args = parse_args()
    if not 0 <= args.beta <= 1:
        raise ValueError("beta must be in [0, 1]")
    device = torch.device(args.device)
    datadir = Path(args.datadir)
    spec = load_setup(datadir / "Round1_Setup.json")
    train_positions = np.load(
        datadir / "Round1_Train_Pos.npy"
    ).astype(np.float32)
    test_positions = np.load(
        datadir / "Round1_Test_Pos.npy"
    ).astype(np.float32)
    train_channels = np.load(
        datadir / "Round1_Train_Channel.npy", mmap_mode="r"
    )
    backbone_path = Path(args.backbone)
    backbone = np.load(backbone_path, mmap_mode="r")
    seeds = [int(value) for value in args.seeds.split(",") if value]

    print("[refiner-build] caching labeled PHV PAS", flush=True)
    train_pas = torch.empty(
        len(train_positions),
        spec.n * spec.s,
        spec.mh * spec.mv,
        dtype=torch.float32,
        device=device,
    )
    for start in range(0, len(train_positions), 50):
        stop = min(start + 50, len(train_positions))
        channels = torch.as_tensor(
            np.array(train_channels[start:stop], copy=True),
            dtype=torch.complex64,
            device=device,
        )
        spectrum = pas_spectrum_phv(channels, spec).reshape(
            stop - start, spec.n * spec.s, spec.mh * spec.mv
        )
        train_pas[start:stop] = normalize(spectrum, -1)
        del channels, spectrum

    tree = cKDTree(train_positions[:, :2])
    bs_xy = np.asarray(spec.bs_position[:2], dtype=np.float32)
    refiner_sum = torch.zeros(
        len(test_positions),
        spec.n * spec.s,
        spec.mh * spec.mv,
        dtype=torch.float32,
        device=device,
    )
    checkpoint_manifest = {}
    for seed in seeds:
        path = Path(args.checkpoint_pattern.format(seed=seed))
        payload = torch.load(path, map_location=device, weights_only=False)
        meta = payload["meta"]
        if int(meta.get("split_seed", -1)) != seed:
            raise ValueError(f"{path}: split seed mismatch")
        if meta.get("pas_layout") != "phv":
            raise ValueError(f"{path}: not a PHV checkpoint")
        k = int(meta["neighbors"])
        model = PasSpectrumRefiner(k).to(device)
        model.load_state_dict(payload["model_state"])
        model.eval()
        distance, neighbors = tree.query(test_positions[:, :2], k=k)
        distance = distance.astype(np.float32)
        target_xy = test_positions[:, :2]
        delta = train_positions[neighbors, :2] - target_xy[:, None, :]
        radial = target_xy - bs_xy
        radius = np.linalg.norm(radial, axis=1).clip(1e-3)
        angle = np.arctan2(radial[:, 1], radial[:, 0])
        geometry = torch.from_numpy(
            np.concatenate(
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
        ).to(device)
        with torch.inference_mode():
            for start in range(0, len(test_positions), 10):
                stop = min(start + 10, len(test_positions))
                batch = stop - start
                neighbor_spectra = train_pas[
                    torch.as_tensor(
                        neighbors[start:stop],
                        dtype=torch.long,
                        device=device,
                    )
                ].permute(0, 2, 1, 3).reshape(
                    batch * spec.n * spec.s,
                    k,
                    spec.mh,
                    spec.mv,
                )
                batch_distance = torch.as_tensor(
                    distance[start:stop], device=device
                )[:, None, :].expand(-1, spec.n * spec.s, -1).reshape(
                    -1, k
                )
                batch_geometry = geometry[
                    start:stop, None, :
                ].expand(-1, spec.n * spec.s, -1).reshape(
                    -1, geometry.shape[-1]
                )
                prediction = model(
                    neighbor_spectra, batch_distance, batch_geometry
                ).reshape(batch, spec.n * spec.s, spec.mh * spec.mv)
                refiner_sum[start:stop] += prediction
        checkpoint_manifest[str(seed)] = {
            "path": str(path),
            "sha256": sha256(path),
            "best_pas": float(meta["best_pas"]),
            "epoch": int(meta["epoch"]),
        }
        del model
        print(f"[refiner-build] checkpoint seed={seed} done", flush=True)
    refiner_pas = normalize(refiner_sum / len(seeds), -1)

    def enforce_pas(
        value: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        angular = torch.fft.fft2(
            value.reshape(
                -1, spec.mp, spec.mh, spec.mv, spec.n, spec.s
            ),
            dim=(2, 3),
            norm="ortho",
        )
        current = angular.abs().square().sum(dim=1)
        current_vector = current.reshape(
            -1, spec.mh * spec.mv, spec.n, spec.s
        ).permute(0, 2, 3, 1)
        desired = target * current_vector.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-30)
        desired = desired.permute(0, 3, 1, 2).reshape_as(current)
        gain = torch.sqrt(
            desired.clamp_min(0) / current.clamp_min(1e-38)
        )[:, None]
        return torch.fft.ifft2(
            angular * gain, dim=(2, 3), norm="ortho"
        ).reshape(value.shape)

    def enforce_pdp(
        value: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        delay = torch.fft.ifft(value, dim=-1, norm="ortho")
        current = delay.abs().square()
        desired = target * current.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-30)
        adjusted = torch.sqrt(desired.clamp_min(0)) * (
            delay / delay.abs().clamp_min(1e-30)
        )
        return torch.fft.fft(adjusted, dim=-1, norm="ortho")

    output = np.empty(backbone.shape, dtype=np.complex64)
    for start in range(0, len(backbone), args.batch_size):
        stop = min(start + args.batch_size, len(backbone))
        value = torch.as_tensor(
            np.array(backbone[start:stop], copy=True),
            dtype=torch.complex64,
            device=device,
        )
        current_pas = normalize(
            pas_spectrum_phv(value, spec).reshape(
                stop - start, spec.n * spec.s, spec.mh * spec.mv
            ),
            -1,
        )
        target_pas = normalize(
            (1.0 - args.beta) * current_pas
            + args.beta * refiner_pas[start:stop],
            -1,
        ).reshape(stop - start, spec.n, spec.s, spec.mh * spec.mv)
        target_pdp = normalize(pdp_spectrum(value, spec), -1)
        adjusted = enforce_pas(value, target_pas)
        for _ in range(args.iterations):
            adjusted = enforce_pdp(adjusted, target_pdp)
            adjusted = enforce_pas(adjusted, target_pas)
        adjusted = (
            adjusted
            / adjusted.abs().square().mean().sqrt().clamp_min(1e-30)
            * value.abs().square().mean().sqrt()
        )
        output[start:stop] = adjusted.cpu().numpy()
        print(f"[refiner-build] projection {stop}/{len(backbone)}", flush=True)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "Round1_Test_Channel.npy"
    np.save(out, output)
    manifest = {
        "formula": (
            "(1-beta)*backbone PHV PAS + beta*refiner ensemble PHV PAS; "
            "backbone PDP preserved by alternating projection"
        ),
        "beta": args.beta,
        "iterations": args.iterations,
        "backbone": f"{backbone_path}:{sha256(backbone_path)}",
        "checkpoints": checkpoint_manifest,
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": bool(np.isfinite(output).all()),
        "rms": float(np.sqrt(np.mean(np.abs(output) ** 2))),
        "sha256": sha256(out),
    }
    (outdir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"PAS_REFINER_BUILD_DONE out={out} sha256={manifest['sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
