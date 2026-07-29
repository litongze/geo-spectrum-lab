#!/usr/bin/env python3
"""Train and strictly cross-validate a low-capacity PAS transport gate."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from _bootstrap import ROOT  # noqa: F401
from pas_transport_gate import (
    FEATURE_NAMES,
    GATE_CENTER,
    PasTransportGate,
    build_gate_features,
    mixed_cosine_from_dots,
)
from score_holdout import reproduce_val_indices
from wireless_twin.data.setup_config import load_setup


@dataclass
class SplitData:
    name: str
    query_id: np.ndarray
    features: np.ndarray
    raw_truth: np.ndarray
    transport_truth: np.ndarray
    raw_transport: np.ndarray


def unit(value: torch.Tensor) -> torch.Tensor:
    return value / value.norm(dim=-1, keepdim=True).clamp_min(
        torch.finfo(value.dtype).tiny
    )


def split_indices(
    size: int,
    split: str,
    external_name: str,
    external_indices: str | None,
) -> np.ndarray:
    if split == external_name:
        if external_indices is None:
            raise ValueError("external split requires --external-indices")
        return np.load(external_indices).astype(np.int64)
    return np.asarray(
        sorted(reproduce_val_indices(size, 0.1, int(split))),
        dtype=np.int64,
    )


def load_split(
    name: str,
    val_idx: np.ndarray,
    positions: np.ndarray,
    truth_cache: np.ndarray,
    raw_path: Path,
    transport_path: Path,
    k: int,
    spec: object,
    device: torch.device,
    query_batch: int,
) -> SplitData:
    all_idx = np.arange(len(positions), dtype=np.int64)
    pool_idx = np.setdiff1d(all_idx, val_idx)
    distance, _ = cKDTree(positions[pool_idx, :2]).query(
        positions[val_idx, :2], k=k
    )
    distance = np.asarray(distance, dtype=np.float32)
    raw_cache = np.load(raw_path, mmap_mode="r")
    transport_cache = np.load(transport_path, mmap_mode="r")
    expected = (
        len(val_idx),
        spec.n,
        spec.s,
        spec.mh * spec.mv,
    )
    if raw_cache.shape != expected or transport_cache.shape != expected:
        raise ValueError(
            f"{name}: expected {expected}, got "
            f"{raw_cache.shape} and {transport_cache.shape}"
        )

    feature_chunks = []
    raw_truth_chunks = []
    transport_truth_chunks = []
    raw_transport_chunks = []
    bs = torch.as_tensor(
        spec.bs_position, dtype=torch.float32, device=device
    )
    with torch.inference_mode():
        for start in range(0, len(val_idx), query_batch):
            stop = min(start + query_batch, len(val_idx))
            raw = unit(
                torch.as_tensor(
                    np.array(raw_cache[start:stop], copy=True),
                    dtype=torch.float32,
                    device=device,
                )
            )
            transported = unit(
                torch.as_tensor(
                    np.array(transport_cache[start:stop], copy=True),
                    dtype=torch.float32,
                    device=device,
                )
            )
            truth = unit(
                torch.as_tensor(
                    np.array(
                        truth_cache[val_idx[start:stop]], copy=True
                    ),
                    dtype=torch.float32,
                    device=device,
                )
            )
            features = build_gate_features(
                raw,
                transported,
                torch.as_tensor(
                    positions[val_idx[start:stop]],
                    dtype=torch.float32,
                    device=device,
                ),
                torch.as_tensor(
                    distance[start:stop],
                    dtype=torch.float32,
                    device=device,
                ),
                bs,
                spec.mh,
                spec.mv,
            )
            feature_chunks.append(features.cpu().numpy())
            raw_truth_chunks.append(
                (raw * truth).sum(dim=-1).cpu().numpy()
            )
            transport_truth_chunks.append(
                (transported * truth).sum(dim=-1).cpu().numpy()
            )
            raw_transport_chunks.append(
                (raw * transported).sum(dim=-1).cpu().numpy()
            )
    result = SplitData(
        name=name,
        query_id=val_idx,
        features=np.concatenate(feature_chunks).astype(
            np.float32, copy=False
        ),
        raw_truth=np.concatenate(raw_truth_chunks).astype(
            np.float32, copy=False
        ),
        transport_truth=np.concatenate(
            transport_truth_chunks
        ).astype(np.float32, copy=False),
        raw_transport=np.concatenate(
            raw_transport_chunks
        ).astype(np.float32, copy=False),
    )
    fixed = score_beta(result, GATE_CENTER)
    oracle = oracle_score(result)
    print(
        f"[gate-data] split={name} raw={result.raw_truth.mean():.6f} "
        f"transport={result.transport_truth.mean():.6f} "
        f"fixed={fixed:.6f} oracle_005_045={oracle:.6f}",
        flush=True,
    )
    return result


def score_beta(data: SplitData, beta: float) -> float:
    left = 1.0 - beta
    numerator = (
        left * data.raw_truth + beta * data.transport_truth
    )
    denominator = np.sqrt(
        left * left
        + beta * beta
        + 2.0 * left * beta * data.raw_transport
    )
    return float(
        np.mean(
            numerator
            / np.maximum(denominator, np.finfo(np.float32).tiny)
        )
    )


def oracle_score(data: SplitData) -> float:
    c = data.raw_transport.astype(np.float64)
    u = data.raw_truth.astype(np.float64)
    v = data.transport_truth.astype(np.float64)
    denominator = (1.0 - c) * (u + v)
    beta = np.full_like(u, GATE_CENTER)
    valid = np.abs(denominator) > 1e-12
    beta[valid] = (v[valid] - c[valid] * u[valid]) / denominator[valid]
    beta = np.clip(beta, 0.05, 0.45)
    left = 1.0 - beta
    score = (left * u + beta * v) / np.sqrt(
        np.maximum(
            left * left + beta * beta + 2.0 * left * beta * c,
            np.finfo(np.float64).tiny,
        )
    )
    return float(score.mean())


def combine_training(
    splits: list[SplitData],
    excluded_query_ids: set[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Combine complete query blocks while deduplicating global positions."""
    seen = set(excluded_query_ids)
    features = []
    raw_truth = []
    transport_truth = []
    raw_transport = []
    used_queries = 0
    for data in splits:
        keep = []
        for local, query_id in enumerate(data.query_id):
            key = int(query_id)
            if key not in seen:
                seen.add(key)
                keep.append(local)
        if not keep:
            continue
        features.append(data.features[keep])
        raw_truth.append(data.raw_truth[keep])
        transport_truth.append(data.transport_truth[keep])
        raw_transport.append(data.raw_transport[keep])
        used_queries += len(keep)
    return (
        np.concatenate(features).reshape(-1, len(FEATURE_NAMES)),
        np.concatenate(raw_truth).reshape(-1),
        np.concatenate(transport_truth).reshape(-1),
        np.concatenate(raw_transport).reshape(-1),
        used_queries,
    )


def fit_gate(
    architecture: str,
    training: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int],
    device: torch.device,
    epochs: int,
    batch_size: int,
    regularization: float,
    seed: int,
) -> tuple[PasTransportGate, np.ndarray, np.ndarray, dict[str, float]]:
    x_np, u_np, v_np, c_np, query_count = training
    mean = x_np.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x_np.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-5)
    x = torch.as_tensor(
        np.clip((x_np - mean) / std, -8.0, 8.0),
        dtype=torch.float32,
        device=device,
    )
    u = torch.as_tensor(u_np, dtype=torch.float32, device=device)
    v = torch.as_tensor(v_np, dtype=torch.float32, device=device)
    c = torch.as_tensor(c_np, dtype=torch.float32, device=device)
    torch.manual_seed(seed)
    model = PasTransportGate(len(FEATURE_NAMES), architecture).to(device)
    learning_rate = 1e-2 if architecture == "linear" else 3e-3
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-3,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 101)
    sample_count = len(x)
    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(
            sample_count, generator=generator, device=device
        )
        epoch_score = 0.0
        epoch_beta = 0.0
        for start in range(0, sample_count, batch_size):
            index = permutation[start : start + batch_size]
            beta = model(x[index])
            score = mixed_cosine_from_dots(
                beta, u[index], v[index], c[index]
            )
            loss = -score.mean() + regularization * (
                beta - GATE_CENTER
            ).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            epoch_score += float(score.detach().sum())
            epoch_beta += float(beta.detach().sum())
        if epoch in {0, epochs - 1}:
            print(
                f"[gate-train] arch={architecture} epoch={epoch + 1}/"
                f"{epochs} score={epoch_score / sample_count:.6f} "
                f"beta={epoch_beta / sample_count:.4f}",
                flush=True,
            )
    model.eval()
    with torch.inference_mode():
        beta = model(x)
        train_score = float(
            mixed_cosine_from_dots(beta, u, v, c).mean()
        )
    stats = {
        "query_count": query_count,
        "slice_count": sample_count,
        "score": train_score,
        "beta_mean": float(beta.mean()),
        "beta_std": float(beta.std(correction=0)),
        "beta_min": float(beta.min()),
        "beta_max": float(beta.max()),
    }
    del x, u, v, c, beta
    return model, mean, std, stats


def evaluate_gate(
    model: PasTransportGate,
    mean: np.ndarray,
    std: np.ndarray,
    data: SplitData,
    device: torch.device,
    batch_size: int,
    beta_out: Path | None = None,
) -> dict[str, float]:
    x_np = data.features.reshape(-1, len(FEATURE_NAMES))
    u_np = data.raw_truth.reshape(-1)
    v_np = data.transport_truth.reshape(-1)
    c_np = data.raw_transport.reshape(-1)
    score_sum = 0.0
    beta_values = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x_np), batch_size):
            stop = min(start + batch_size, len(x_np))
            x = torch.as_tensor(
                np.clip(
                    (x_np[start:stop] - mean) / std, -8.0, 8.0
                ),
                dtype=torch.float32,
                device=device,
            )
            beta = model(x)
            score = mixed_cosine_from_dots(
                beta,
                torch.as_tensor(
                    u_np[start:stop],
                    dtype=torch.float32,
                    device=device,
                ),
                torch.as_tensor(
                    v_np[start:stop],
                    dtype=torch.float32,
                    device=device,
                ),
                torch.as_tensor(
                    c_np[start:stop],
                    dtype=torch.float32,
                    device=device,
                ),
            )
            score_sum += float(score.sum())
            beta_values.append(beta.cpu().numpy())
    beta_np = np.concatenate(beta_values)
    if beta_out is not None:
        beta_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(beta_out, beta_np.reshape(data.raw_truth.shape))
    fixed = score_beta(data, GATE_CENTER)
    score = score_sum / len(x_np)
    return {
        "raw": float(data.raw_truth.mean()),
        "transport": float(data.transport_truth.mean()),
        "fixed_beta_025": fixed,
        "score": score,
        "delta_vs_fixed": score - fixed,
        "beta_mean": float(beta_np.mean()),
        "beta_std": float(beta_np.std()),
        "beta_min": float(beta_np.min()),
        "beta_max": float(beta_np.max()),
        "oracle_005_045": oracle_score(data),
    }


def save_checkpoint(
    path: Path,
    model: PasTransportGate,
    mean: np.ndarray,
    std: np.ndarray,
    architecture: str,
    metadata: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "architecture": architecture,
            "feature_names": list(FEATURE_NAMES),
            "feature_mean": mean,
            "feature_std": std,
            "gate_center": GATE_CENTER,
            "gate_radius": 0.20,
            "metadata": metadata,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", default="Round1_Map(2)")
    parser.add_argument(
        "--truth-cache",
        default="cache/teammate_knn_hvp/train_pas_phv.npy",
    )
    parser.add_argument(
        "--raw-dir", default="cache/moment_attention_selected"
    )
    parser.add_argument(
        "--transport-dir", default="cache/pas_transport_k16_p2"
    )
    parser.add_argument("--tune-seeds", default="1890,3716,962,1022")
    parser.add_argument("--audit-seed", default="2262")
    parser.add_argument("--external-indices")
    parser.add_argument("--external-name", default="testmatched")
    parser.add_argument("--architectures", default="linear,mlp8")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--feature-query-batch", type=int, default=20)
    parser.add_argument("--regularization", type=float, default=0.005)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/pas_transport_gate/selected.pt",
    )
    parser.add_argument(
        "--beta-outdir", default="cache/pas_transport_gate"
    )
    parser.add_argument(
        "--out", default="docs/pas_transport_gate/result.json"
    )
    args = parser.parse_args()

    tune = [value for value in args.tune_seeds.split(",") if value]
    split_names = tune + [args.audit_seed]
    if args.external_indices:
        split_names.append(args.external_name)
    architectures = [
        value for value in args.architectures.split(",") if value
    ]
    datadir = Path(args.datadir)
    spec = load_setup(datadir / "Round1_Setup.json")
    positions = np.load(
        datadir / "Round1_Train_Pos.npy"
    ).astype(np.float32)
    truth_cache = np.load(args.truth_cache, mmap_mode="r")
    device = torch.device(args.device)

    data: dict[str, SplitData] = {}
    for name in split_names:
        val_idx = split_indices(
            len(positions),
            name,
            args.external_name,
            args.external_indices,
        )
        data[name] = load_split(
            name,
            val_idx,
            positions,
            truth_cache,
            Path(args.raw_dir) / f"pas_{name}_c1.npy",
            Path(args.transport_dir) / f"pas_{name}.npy",
            args.k,
            spec,
            device,
            args.feature_query_batch,
        )

    cross_validation = {}
    for architecture in architectures:
        fold_rows = {}
        for fold_number, heldout in enumerate(tune):
            train_splits = [
                data[name] for name in tune if name != heldout
            ]
            training = combine_training(
                train_splits,
                {int(value) for value in data[heldout].query_id},
            )
            model, mean, std, train_stats = fit_gate(
                architecture,
                training,
                device,
                args.epochs,
                args.batch_size,
                args.regularization,
                1701 + fold_number,
            )
            evaluation = evaluate_gate(
                model,
                mean,
                std,
                data[heldout],
                device,
                args.batch_size,
                Path(args.beta_outdir) / f"beta_{heldout}.npy",
            )
            fold_rows[heldout] = {
                "training": train_stats,
                "evaluation": evaluation,
            }
            print(
                f"[gate-cv] arch={architecture} heldout={heldout} "
                f"score={evaluation['score']:.6f} "
                f"fixed={evaluation['fixed_beta_025']:.6f} "
                f"delta={evaluation['delta_vs_fixed']:+.6f}",
                flush=True,
            )
            del model
            torch.cuda.empty_cache()
        deltas = [
            row["evaluation"]["delta_vs_fixed"]
            for row in fold_rows.values()
        ]
        scores = [
            row["evaluation"]["score"] for row in fold_rows.values()
        ]
        cross_validation[architecture] = {
            "median_delta": float(np.median(deltas)),
            "mean_delta": float(np.mean(deltas)),
            "worst_delta": float(np.min(deltas)),
            "median_score": float(np.median(scores)),
            "folds": fold_rows,
        }

    ranked = sorted(
        cross_validation,
        key=lambda architecture: (
            cross_validation[architecture]["median_delta"],
            cross_validation[architecture]["mean_delta"],
            cross_validation[architecture]["worst_delta"],
        ),
        reverse=True,
    )
    selected = ranked[0]
    diagnostics = {}
    for offset, name in enumerate(
        [args.audit_seed]
        + ([args.external_name] if args.external_indices else [])
    ):
        training = combine_training(
            [data[key] for key in tune],
            {int(value) for value in data[name].query_id},
        )
        model, mean, std, train_stats = fit_gate(
            selected,
            training,
            device,
            args.epochs,
            args.batch_size,
            args.regularization,
            2901 + offset,
        )
        diagnostics[name] = {
            "training": train_stats,
            "evaluation": evaluate_gate(
                model,
                mean,
                std,
                data[name],
                device,
                args.batch_size,
                Path(args.beta_outdir) / f"beta_{name}.npy",
            ),
        }
        evaluation = diagnostics[name]["evaluation"]
        print(
            f"[gate-diagnostic] split={name} score="
            f"{evaluation['score']:.6f} fixed="
            f"{evaluation['fixed_beta_025']:.6f} delta="
            f"{evaluation['delta_vs_fixed']:+.6f}",
            flush=True,
        )
        del model
        torch.cuda.empty_cache()

    final_training = combine_training(
        [data[name] for name in tune + [args.audit_seed]], set()
    )
    final_model, final_mean, final_std, final_stats = fit_gate(
        selected,
        final_training,
        device,
        args.epochs,
        args.batch_size,
        args.regularization,
        3901,
    )
    checkpoint = Path(args.checkpoint)
    checkpoint_metadata = {
        "selection_policy": (
            "architecture selected only by four-fold query-disjoint OOF "
            "cross-validation; audit and external splits are diagnostics"
        ),
        "tune_seeds": tune,
        "audit_seed": args.audit_seed,
        "epochs": args.epochs,
        "regularization": args.regularization,
        "final_training": final_stats,
    }
    save_checkpoint(
        checkpoint,
        final_model,
        final_mean,
        final_std,
        selected,
        checkpoint_metadata,
    )
    payload = {
        "selection_policy": checkpoint_metadata["selection_policy"],
        "feature_names": list(FEATURE_NAMES),
        "gate_range": [0.05, 0.45],
        "tune_seeds": tune,
        "audit_seed": args.audit_seed,
        "external_name": (
            args.external_name if args.external_indices else None
        ),
        "epochs": args.epochs,
        "regularization": args.regularization,
        "ranked_architectures": ranked,
        "cross_validation": cross_validation,
        "selected_architecture": selected,
        "diagnostics": diagnostics,
        "final_training": final_stats,
        "checkpoint": str(checkpoint),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"PAS_TRANSPORT_GATE_DONE selected={selected} out={out} "
        f"checkpoint={checkpoint}",
        flush=True,
    )


if __name__ == "__main__":
    main()
