#!/usr/bin/env python3
"""Sweep label-clean bilateral KNN interpolation in PAS and PDP space.

Each validation query is removed from the labeled neighbor bank. Candidate
training spectra are reweighted by spatial distance and their agreement with
an anchor derived only from those candidates. This probes whether ordinary
KNN mixes samples across local propagation boundaries.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from _bootstrap import ROOT  # noqa: F401
from score_holdout import reproduce_val_indices
from wireless_twin.data.setup_config import load_setup
from wireless_twin.signal import pas_spectrum_phv, pdp_spectrum


def parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def parse_names(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def stable_unit(value: torch.Tensor) -> torch.Tensor:
    maximum = value.amax(dim=-1, keepdim=True)
    scaled = value / maximum.clamp_min(torch.finfo(value.dtype).tiny)
    norm = scaled.norm(dim=-1, keepdim=True)
    return torch.where(norm > 0, scaled / norm.clamp_min(1e-30), 0.0)


def anchor_agreements(
    neighbors: torch.Tensor,
    distance: torch.Tensor,
    modes: list[str],
) -> dict[str, torch.Tensor]:
    """Return candidate-anchor cosine features with shape ``(B, K, Q)``."""
    nearest = (neighbors * neighbors[:, :1]).sum(dim=-1)
    anchor_weight = distance.clamp_min(0.3).pow(-2)
    anchor_weight /= anchor_weight.sum(dim=1, keepdim=True)
    consensus = stable_unit(
        torch.einsum("bk,bkql->bql", anchor_weight, neighbors)
    )
    consensus_agreement = (
        neighbors * consensus[:, None]
    ).sum(dim=-1)

    batch, _, slices, length = neighbors.shape
    slice_medoid = consensus_agreement.argmax(dim=1)
    slice_anchor = neighbors.gather(
        1,
        slice_medoid[:, None, :, None].expand(
            batch, 1, slices, length
        ),
    ).squeeze(1)
    medoid_slice = (
        neighbors * slice_anchor[:, None]
    ).sum(dim=-1)

    global_medoid = consensus_agreement.mean(dim=-1).argmax(dim=1)
    global_anchor = neighbors[
        torch.arange(batch, device=neighbors.device), global_medoid
    ]
    medoid_global = (
        neighbors * global_anchor[:, None]
    ).sum(dim=-1).mean(dim=-1, keepdim=True)

    available = {
        "nearest_slice": nearest,
        "nearest_global": nearest.mean(dim=-1, keepdim=True),
        "consensus_slice": consensus_agreement,
        "consensus_global": consensus_agreement.mean(
            dim=-1, keepdim=True
        ),
        "medoid_slice": medoid_slice,
        "medoid_global": medoid_global,
        "nearest_consensus_slice": 0.5 * (
            nearest + consensus_agreement
        ),
        "nearest_consensus_global": 0.5 * (
            nearest + consensus_agreement
        ).mean(dim=-1, keepdim=True),
    }
    unknown = sorted(set(modes) - set(available))
    if unknown:
        raise ValueError(f"unknown anchor modes: {unknown}")
    return {name: available[name] for name in modes}


def load_domain(
    domain: str,
    cache_dir: Path,
    baseline_path: Path,
    spec,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache_name = (
        "train_pas_phv.npy" if domain == "pas" else "train_pdp.npy"
    )
    cached = np.load(cache_dir / cache_name, mmap_mode="r")
    spectra = stable_unit(
        torch.as_tensor(
            np.array(cached, copy=True),
            dtype=torch.float32,
            device=device,
        )
    )
    del cached

    channel_mmap = np.load(baseline_path, mmap_mode="r")
    channel = torch.as_tensor(
        np.array(channel_mmap, copy=True),
        dtype=torch.complex64,
        device=device,
    )
    del channel_mmap
    transform = pas_spectrum_phv if domain == "pas" else pdp_spectrum
    baseline = stable_unit(transform(channel, spec))
    del channel
    return spectra, baseline


def sweep_domain(
    *,
    domain: str,
    cache_dir: Path,
    baseline_path: Path,
    spec,
    val_idx: np.ndarray,
    neighbors: np.ndarray,
    distance: np.ndarray,
    powers: list[float],
    lambdas: list[float],
    blends: list[float],
    modes: list[str],
    batch_size: int,
    device: torch.device,
) -> dict:
    print(f"[bilateral] loading {domain} spectra", flush=True)
    spectra, baseline = load_domain(
        domain, cache_dir, baseline_path, spec, device
    )
    truth = spectra[
        torch.as_tensor(val_idx, dtype=torch.long, device=device)
    ]
    truth = truth.reshape(len(val_idx), -1, truth.shape[-1])
    baseline = baseline.reshape(
        len(val_idx), -1, baseline.shape[-1]
    )
    configs = [
        (power, strength)
        for power in powers
        for strength in lambdas
    ]
    config_power = torch.tensor(
        [item[0] for item in configs],
        dtype=torch.float32,
        device=device,
    )
    config_lambda = torch.tensor(
        [item[1] for item in configs],
        dtype=torch.float32,
        device=device,
    )
    totals = {
        mode: torch.zeros(
            len(configs), len(blends), device=device
        )
        for mode in modes
    }
    count = 0
    for start in range(0, len(val_idx), batch_size):
        stop = min(start + batch_size, len(val_idx))
        neighbor_index = torch.as_tensor(
            neighbors[start:stop], dtype=torch.long, device=device
        )
        values = spectra[neighbor_index]
        values = values.reshape(
            stop - start, values.shape[1], -1, values.shape[-1]
        )
        distance_t = torch.as_tensor(
            distance[start:stop],
            dtype=torch.float32,
            device=device,
        ).clamp_min(0.3)
        agreements = anchor_agreements(values, distance_t, modes)
        log_distance = distance_t.log()[None, :, :, None]
        target = truth[start:stop]
        base = baseline[start:stop]
        for mode, agreement in agreements.items():
            logits = (
                -config_power[:, None, None, None] * log_distance
                + config_lambda[:, None, None, None]
                * agreement[None]
            )
            weight = torch.softmax(logits, dim=2)
            selected = stable_unit(
                torch.einsum("gbkq,bkql->gbql", weight, values)
            )
            for blend_index, beta in enumerate(blends):
                prediction = stable_unit(
                    (1.0 - beta) * base[None] + beta * selected
                )
                totals[mode][:, blend_index] += (
                    prediction * target[None]
                ).sum(dim=(1, 2, 3))
            del logits, weight, selected
        count += target.shape[0] * target.shape[1]
        if stop == len(val_idx) or stop % 40 == 0:
            print(
                f"[bilateral] {domain} {stop}/{len(val_idx)}",
                flush=True,
            )

    rows = []
    for mode in modes:
        scores = totals[mode] / count
        for config_index, (power, strength) in enumerate(configs):
            for blend_index, beta in enumerate(blends):
                rows.append(
                    {
                        "mode": mode,
                        "distance_power": power,
                        "anchor_strength": strength,
                        "blend_beta": beta,
                        "score": float(
                            scores[config_index, blend_index]
                        ),
                    }
                )
    rows.sort(key=lambda row: row["score"], reverse=True)
    baseline_score = float((baseline * truth).sum() / count)
    best_by_mode = {
        mode: next(row for row in rows if row["mode"] == mode)
        for mode in modes
    }
    expert_rows = [
        row for row in rows if row["blend_beta"] == 1.0
    ]
    result = {
        "domain": domain,
        "baseline_score": baseline_score,
        "best": rows[0],
        "gain": rows[0]["score"] - baseline_score,
        "best_expert": expert_rows[0] if expert_rows else None,
        "best_by_mode": best_by_mode,
        "rows": rows,
        "invalid_truth_fraction": float(
            (truth.norm(dim=-1) == 0).float().mean()
        ),
    }
    del spectra, baseline, truth
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", default="Round1_Map(2)")
    parser.add_argument(
        "--cache-dir", default="cache/teammate_knn_hvp"
    )
    parser.add_argument("--split-seed", type=int, default=1890)
    parser.add_argument("--baseline-val", required=True)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--domains", default="pas,pdp")
    parser.add_argument(
        "--modes",
        default=(
            "nearest_slice,nearest_global,consensus_slice,"
            "consensus_global,medoid_slice,medoid_global,"
            "nearest_consensus_slice,nearest_consensus_global"
        ),
    )
    parser.add_argument(
        "--distance-powers", default="0.5,1,2,3"
    )
    parser.add_argument(
        "--anchor-strengths", default="0,1,2,4,8,16"
    )
    parser.add_argument(
        "--blend-grid", default="0,0.05,0.1,0.2,0.3,0.5,0.75,1"
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out",
        default="docs/bilateral_spectrum_knn_s1890/result.json",
    )
    args = parser.parse_args()

    domains = parse_names(args.domains)
    if not set(domains) <= {"pas", "pdp"}:
        raise ValueError("--domains must contain only pas,pdp")
    modes = parse_names(args.modes)
    powers = parse_floats(args.distance_powers)
    strengths = parse_floats(args.anchor_strengths)
    blends = parse_floats(args.blend_grid)
    device = torch.device(args.device)
    datadir = Path(args.datadir)
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
    pool_idx = np.setdiff1d(np.arange(len(positions)), val_idx)
    distance, local = cKDTree(positions[pool_idx, :2]).query(
        positions[val_idx, :2], k=args.k
    )
    distance = np.asarray(distance, dtype=np.float32)
    neighbors = pool_idx[np.asarray(local)]
    spec = load_setup(datadir / "Round1_Setup.json")

    result = {
        "split_seed": args.split_seed,
        "clean_neighbor_bank": True,
        "validation_size": len(val_idx),
        "neighbor_bank_size": len(pool_idx),
        "k": args.k,
        "distance_powers": powers,
        "anchor_strengths": strengths,
        "blend_grid": blends,
        "modes": modes,
        "domains": {},
    }
    for domain in domains:
        result["domains"][domain] = sweep_domain(
            domain=domain,
            cache_dir=Path(args.cache_dir),
            baseline_path=Path(args.baseline_val),
            spec=spec,
            val_idx=val_idx,
            neighbors=neighbors,
            distance=distance,
            powers=powers,
            lambdas=strengths,
            blends=blends,
            modes=modes,
            batch_size=args.batch_size,
            device=device,
        )
        summary = result["domains"][domain]
        print(
            f"[bilateral] {domain} baseline="
            f"{summary['baseline_score']:.6f} best="
            f"{summary['best']['score']:.6f} "
            f"gain={summary['gain']:+.6f}",
            flush=True,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    summary = {
        "split_seed": result["split_seed"],
        "clean_neighbor_bank": result["clean_neighbor_bank"],
        "domains": {
            domain: {
                "baseline_score": values["baseline_score"],
                "best": values["best"],
                "gain": values["gain"],
                "best_expert": values["best_expert"],
            }
            for domain, values in result["domains"].items()
        },
    }
    print(json.dumps(summary, indent=2), flush=True)
    print(f"BILATERAL_SPECTRUM_KNN_DONE out={out}", flush=True)


if __name__ == "__main__":
    main()
