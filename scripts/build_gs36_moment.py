#!/usr/bin/env python3
"""Build the full-bank anisotropic moment-attention submission candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

from _bootstrap import ROOT  # noqa: F401
from sweep_moment_attention import (
    SliceAttention,
    moment_project,
    stable_unit,
)
from validate_moment_projection import enforce_pas, enforce_pdp
from wireless_twin.data.map_loader import load_point_cloud
from wireless_twin.data.setup_config import load_setup
from wireless_twin.models.raytrace2 import build_heightmap
from wireless_twin.signal import pas_spectrum_phv, pdp_spectrum


SEEDS = [1890, 3716, 962, 1022, 2262]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", default="Round1_Map(2)")
    parser.add_argument("--cache-dir", default="cache/teammate_knn_hvp")
    parser.add_argument(
        "--baseline",
        default=(
            "best_submit/BLEND_GS34_PHV_FULLBANK_REF10_TF05/"
            "Round1_Test_Channel.npy"
        ),
    )
    parser.add_argument(
        "--outdir", default="best_submit/BLEND_GS36_MOMENT_RT"
    )
    parser.add_argument("--pas-correction", type=float, default=1.0)
    parser.add_argument("--pas-radial", type=float, default=0.45)
    parser.add_argument("--pas-tangent", type=float, default=1.0)
    parser.add_argument("--pas-blend", type=float, default=0.8)
    parser.add_argument("--pdp-correction", type=float, default=0.8)
    parser.add_argument("--pdp-radial", type=float, default=1.0)
    parser.add_argument("--pdp-tangent", type=float, default=1.0)
    parser.add_argument("--pdp-blend", type=float, default=0.3)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    datadir = Path(args.datadir)
    spec = load_setup(datadir / "Round1_Setup.json")
    train_pos = np.load(
        datadir / "Round1_Train_Pos.npy"
    ).astype(np.float32)
    test_pos = np.load(
        datadir / "Round1_Test_Pos.npy"
    ).astype(np.float32)
    baseline_path = Path(args.baseline)
    baseline_np = np.load(baseline_path, mmap_mode="r")
    baseline_rms = float(
        np.sqrt(np.mean(np.square(np.abs(baseline_np), dtype=np.float64)))
    )
    baseline = torch.as_tensor(
        np.array(baseline_np, copy=True),
        dtype=torch.complex64,
        device=device,
    )
    del baseline_np

    points = load_point_cloud(datadir / "Round1_Map.ply")
    heightmap, x0, y0, resolution = build_heightmap(points)
    gx = np.clip(
        np.floor((train_pos[:, 0] - x0) / resolution).astype(np.int64),
        0,
        heightmap.shape[0] - 1,
    )
    gy = np.clip(
        np.floor((train_pos[:, 1] - y0) / resolution).astype(np.int64),
        0,
        heightmap.shape[1] - 1,
    )
    indoor = (heightmap[gx, gy] > 2.0).astype(np.float32)
    pool_idx = np.arange(len(train_pos), dtype=np.int64)

    checkpoint_templates = {
        "pas": (
            "checkpoints/clean_panel_phv_geom_k16/s{seed}/"
            "nbrattn_clean_k16_pas_k16s0.pt"
        ),
        "pdp": (
            "checkpoints/clean_panel/s{seed}/"
            "nbrattn_clean_k32_pdp_k32s0.pt"
        ),
    }
    cache_names = {
        "pas": "train_pas_phv.npy",
        "pdp": "train_pdp.npy",
    }
    manifest = {
        "baseline": f"{baseline_path}:{sha256(baseline_path)}",
        "arms": {},
    }

    def build_features(
        domain: str,
        spectra: torch.Tensor,
        query_pos: np.ndarray,
        neighbors: np.ndarray,
        distance: np.ndarray,
        feature_set: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, k = neighbors.shape
        candidate = spectra[
            torch.as_tensor(neighbors, dtype=torch.long, device=device)
        ]
        values = candidate.reshape(
            batch, k, -1, candidate.shape[-1]
        )
        mean = values.mean(dim=1, keepdim=True)
        agreement = F.cosine_similarity(values, mean, dim=-1)
        distance_t = torch.as_tensor(
            distance, dtype=torch.float32, device=device
        )
        expanded_distance = distance_t[:, :, None].expand(
            -1, -1, values.shape[2]
        )
        neighbor_indoor = torch.as_tensor(
            indoor[neighbors], dtype=torch.float32, device=device
        )[:, :, None].expand(-1, -1, values.shape[2])
        # Test points have the same fixed UE height; use map occupancy directly.
        query_gx = np.clip(
            np.floor((query_pos[:, 0] - x0) / resolution).astype(np.int64),
            0,
            heightmap.shape[0] - 1,
        )
        query_gy = np.clip(
            np.floor((query_pos[:, 1] - y0) / resolution).astype(np.int64),
            0,
            heightmap.shape[1] - 1,
        )
        query_indoor_np = (
            heightmap[query_gx, query_gy] > 2.0
        ).astype(np.float32)
        query_indoor = torch.as_tensor(
            query_indoor_np, dtype=torch.float32, device=device
        )[:, None, None].expand(-1, k, values.shape[2])
        columns = [
            expanded_distance / 3.0,
            (neighbor_indoor == query_indoor).float(),
            agreement,
            agreement.square(),
            torch.ones_like(expanded_distance),
            (expanded_distance < 2.5).float(),
        ]
        if feature_set == "geometry":
            target_xy = torch.as_tensor(
                query_pos[:, :2],
                dtype=torch.float32,
                device=device,
            )
            neighbor_xy = torch.as_tensor(
                train_pos[neighbors, :2],
                dtype=torch.float32,
                device=device,
            )
            delta = neighbor_xy - target_xy[:, None]
            bs_xy = torch.as_tensor(
                spec.bs_position[:2],
                dtype=torch.float32,
                device=device,
            )
            radial = target_xy - bs_xy
            radius = radial.norm(dim=-1).clamp_min(1e-3)
            radial_unit = radial / radius[:, None]
            tangent_unit = torch.stack(
                [-radial_unit[:, 1], radial_unit[:, 0]], dim=-1
            )
            radial_delta = (delta * radial_unit[:, None]).sum(dim=-1)
            tangent_delta = (delta * tangent_unit[:, None]).sum(dim=-1)

            def expand(value: torch.Tensor) -> torch.Tensor:
                return value[:, :, None].expand(
                    -1, -1, values.shape[2]
                )

            angle = torch.atan2(radial[:, 1], radial[:, 0])
            columns.extend(
                [
                    expand(delta[..., 0] / 5.0),
                    expand(delta[..., 1] / 5.0),
                    expand(radial_delta / 5.0),
                    expand(tangent_delta / 5.0),
                    (radius / 200.0)[:, None, None].expand(
                        -1, k, values.shape[2]
                    ),
                    angle.sin()[:, None, None].expand(
                        -1, k, values.shape[2]
                    ),
                    angle.cos()[:, None, None].expand(
                        -1, k, values.shape[2]
                    ),
                    neighbor_indoor,
                    query_indoor,
                ]
            )
        elif feature_set != "basic":
            raise ValueError(f"unsupported feature_set={feature_set}")
        features = torch.stack(columns, dim=-1)
        log_distance = torch.log(
            distance_t.clamp_min(0.3)
        )[:, :, None]
        return features, values, log_distance

    def spatial_delta(
        query_pos: np.ndarray,
        neighbors: np.ndarray,
        distance: np.ndarray,
    ) -> torch.Tensor:
        delta = torch.as_tensor(
            train_pos[neighbors, :2] - query_pos[:, None, :2],
            dtype=torch.float32,
            device=device,
        )
        scale = torch.as_tensor(
            np.median(distance, axis=1),
            dtype=torch.float32,
            device=device,
        ).clamp_min(0.3)
        delta /= scale[:, None, None]
        target_xy = torch.as_tensor(
            query_pos[:, :2],
            dtype=torch.float32,
            device=device,
        )
        bs_xy = torch.as_tensor(
            spec.bs_position[:2],
            dtype=torch.float32,
            device=device,
        )
        radial = target_xy - bs_xy
        radial /= radial.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        tangent = torch.stack(
            [-radial[:, 1], radial[:, 0]], dim=-1
        )
        return torch.stack(
            [
                (delta * radial[:, None]).sum(dim=-1),
                (delta * tangent[:, None]).sum(dim=-1),
            ],
            dim=-1,
        )

    experts = {}
    for domain in ("pas", "pdp"):
        print(f"[GS36] loading {domain} spectra", flush=True)
        cached = np.load(
            Path(args.cache_dir) / cache_names[domain], mmap_mode="r"
        )
        spectra = stable_unit(
            torch.as_tensor(
                np.array(cached, copy=True),
                dtype=torch.float32,
                device=device,
            )
        )
        del cached
        k = 16 if domain == "pas" else 32
        distance, local = cKDTree(train_pos[:, :2]).query(
            test_pos[:, :2], k=k
        )
        distance = np.asarray(distance, dtype=np.float32)
        neighbors = pool_idx[np.asarray(local)]
        expert_sum = torch.zeros(
            (len(test_pos),) + tuple(spectra.shape[1:]),
            dtype=torch.float32,
            device=device,
        )
        correction = (
            args.pas_correction
            if domain == "pas"
            else args.pdp_correction
        )
        axis_multiplier = torch.tensor(
            [
                args.pas_radial if domain == "pas" else args.pdp_radial,
                args.pas_tangent if domain == "pas" else args.pdp_tangent,
            ],
            dtype=torch.float32,
            device=device,
        )
        for seed in SEEDS:
            checkpoint = Path(
                checkpoint_templates[domain].format(seed=seed)
            )
            payload = torch.load(
                checkpoint, map_location=device, weights_only=False
            )
            meta = payload["meta"]
            feature_set = str(meta.get("feature_set", "basic"))
            feature_dim = int(
                meta.get(
                    "feature_dim",
                    15 if feature_set == "geometry" else 6,
                )
            )
            model = SliceAttention(feature_dim).to(device)
            model.load_state_dict(payload["model_state"])
            model.eval()
            chunks = []
            with torch.inference_mode():
                for start in range(0, len(test_pos), args.batch_size):
                    stop = min(start + args.batch_size, len(test_pos))
                    features, values, log_distance = build_features(
                        domain,
                        spectra,
                        test_pos[start:stop],
                        neighbors[start:stop],
                        distance[start:stop],
                        feature_set,
                    )
                    logits = model.logits(features, log_distance)
                    weight, _ = moment_project(
                        logits,
                        spatial_delta(
                            test_pos[start:stop],
                            neighbors[start:stop],
                            distance[start:stop],
                        ),
                        correction,
                        axis_multiplier,
                        12,
                        0.03,
                    )
                    chunks.append(
                        stable_unit(
                            torch.einsum(
                                "bkq,bkql->bql", weight, values
                            )
                        )
                    )
            expert_sum += torch.cat(chunks).reshape(expert_sum.shape)
            manifest["arms"][f"{domain}_{seed}"] = (
                f"{checkpoint}:{sha256(checkpoint)}"
            )
            del model, chunks
            torch.cuda.empty_cache()
            print(f"[GS36] {domain} seed={seed} done", flush=True)
        experts[domain] = stable_unit(expert_sum / len(SEEDS))
        del spectra, expert_sum
        torch.cuda.empty_cache()

    baseline_pas = stable_unit(pas_spectrum_phv(baseline, spec))
    baseline_pdp = stable_unit(pdp_spectrum(baseline, spec))
    target_pas = stable_unit(
        (1.0 - args.pas_blend) * baseline_pas
        + args.pas_blend * experts["pas"]
    )
    target_pdp = stable_unit(
        (1.0 - args.pdp_blend) * baseline_pdp
        + args.pdp_blend * experts["pdp"]
    )
    prediction = baseline.clone()
    for _ in range(args.iterations):
        prediction = enforce_pdp(prediction, target_pdp)
        prediction = enforce_pas(prediction, target_pas, spec)
    prediction *= (
        baseline.abs().square().mean().sqrt()
        / prediction.abs().square().mean().sqrt().clamp_min(1e-30)
    )
    output = prediction.detach().cpu().numpy().astype(np.complex64)
    if output.shape != baseline.shape:
        raise ValueError(f"unexpected output shape {output.shape}")
    if not np.isfinite(output.real).all() or not np.isfinite(
        output.imag
    ).all():
        raise ValueError("output contains non-finite values")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    output_path = outdir / "Round1_Test_Channel.npy"
    np.save(output_path, output)
    output_rms = float(
        np.sqrt(np.mean(np.square(np.abs(output), dtype=np.float64)))
    )
    pas_cosine = float(
        (
            stable_unit(pas_spectrum_phv(prediction, spec))
            * baseline_pas
        )
        .sum(dim=-1)
        .mean()
    )
    pdp_cosine = float(
        (
            stable_unit(pdp_spectrum(prediction, spec))
            * baseline_pdp
        )
        .sum(dim=-1)
        .mean()
    )
    channel_cosine = float(
        (
            prediction.conj() * baseline
        ).sum().abs()
        / (
            prediction.norm()
            * baseline.norm()
        ).clamp_min(1e-30)
    )
    result = {
        "name": "BLEND_GS36_MOMENT_RT",
        "source_validation": "docs/moment_projection/result.json",
        "baseline": manifest["baseline"],
        "full_neighbor_bank_size": len(train_pos),
        "seeds": SEEDS,
        "pas": {
            "k": 16,
            "correction": args.pas_correction,
            "radial_multiplier": args.pas_radial,
            "tangent_multiplier": args.pas_tangent,
            "blend": args.pas_blend,
        },
        "pdp": {
            "k": 32,
            "correction": args.pdp_correction,
            "radial_multiplier": args.pdp_radial,
            "tangent_multiplier": args.pdp_tangent,
            "blend": args.pdp_blend,
        },
        "projection_order": "pdp_pas",
        "iterations": args.iterations,
        "baseline_rms": baseline_rms,
        "output_rms": output_rms,
        "relative_to_baseline": {
            "pas_cosine": pas_cosine,
            "pdp_cosine": pdp_cosine,
            "channel_cosine": channel_cosine,
        },
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "sha256": sha256(output_path),
        "manifest": manifest,
    }
    (outdir / "manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (outdir / "说明.txt").write_text(
        "BLEND_GS36_MOMENT_RT\n\n"
        "基于 GS34 的全邻居库矩约束注意力候选。\n"
        "PAS: K16，径向/切向校正=0.45/1.0，专家融合权重=0.8。\n"
        "PDP: K32，各向同性校正=0.8，专家融合权重=0.3。\n"
        "重建: PDP->PAS 交替投影 8 轮，保持 GS34 全局 RMS。\n\n"
        "clean panel 完整复信道验证:\n"
        "- tune median C: 0.74307 -> 0.74548\n"
        "- audit C: 0.76796 -> 0.76953\n"
        "- 五个 split 全部提升。\n",
        encoding="utf-8",
    )
    print(
        f"GS36_MOMENT_DONE path={output_path} "
        f"rms={output_rms:.9e} "
        f"pas_cos={pas_cosine:.6f} pdp_cos={pdp_cosine:.6f} "
        f"channel_cos={channel_cosine:.6f} "
        f"sha256={result['sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
