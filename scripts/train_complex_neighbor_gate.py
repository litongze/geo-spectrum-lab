#!/usr/bin/env python3
"""Train strictly clean complex-neighbor gates on representative splits."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from _bootstrap import ROOT  # noqa: F401
from complex_neighbor_gate import (
    ARCHITECTURES,
    FEATURE_NAMES,
    ComplexNeighborGate,
    baseline_coefficients,
    build_complex_features,
)
from probe_full_array_steering_phase import array_steering_phase
from score_holdout import reproduce_val_indices
from wireless_twin.data.setup_config import load_setup


@dataclass
class GateData:
    features: np.ndarray
    amplitude: np.ndarray
    base_weight: np.ndarray
    gram: np.ndarray
    cross: np.ndarray
    truth_energy: np.ndarray


def split_indices(
    size: int,
    name: str,
    external_name: str,
    external_indices: str | None,
) -> np.ndarray:
    if name == external_name:
        if not external_indices:
            raise ValueError("external split requires indices")
        return np.load(external_indices).astype(np.int64)
    return np.asarray(
        sorted(reproduce_val_indices(size, 0.1, int(name))),
        dtype=np.int64,
    )


def query_neighbors(
    positions: np.ndarray,
    query_idx: np.ndarray,
    pool_idx: np.ndarray,
    k: int,
    leave_self_out: bool,
) -> tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(positions[pool_idx, :2])
    query_k = k + 1 if leave_self_out else k
    distance, local = tree.query(
        positions[query_idx, :2], k=query_k
    )
    distance = np.asarray(distance, dtype=np.float64)
    neighbors = pool_idx[np.asarray(local)]
    if leave_self_out:
        output_distance = np.empty((len(query_idx), k), dtype=np.float64)
        output_neighbors = np.empty((len(query_idx), k), dtype=np.int64)
        for row, query in enumerate(query_idx):
            keep = neighbors[row] != query
            output_neighbors[row] = neighbors[row, keep][:k]
            output_distance[row] = distance[row, keep][:k]
        return output_distance, output_neighbors
    return distance.reshape(len(query_idx), k), neighbors.reshape(
        len(query_idx), k
    )


def build_data(
    name: str,
    query_idx: np.ndarray,
    pool_idx: np.ndarray,
    leave_self_out: bool,
    positions: np.ndarray,
    channels: np.ndarray,
    spec: object,
    config: dict,
    k: int,
    batch_size: int,
    device: torch.device,
) -> GateData:
    distance, neighbors = query_neighbors(
        positions, query_idx, pool_idx, k, leave_self_out
    )
    bs = np.asarray(spec.bs_position, dtype=np.float64)
    radius = np.linalg.norm(positions - bs[None], axis=1)
    unit = (positions - bs[None]) / np.maximum(
        radius[:, None], 1e-12
    )
    delta_radius = (
        radius[query_idx, None] - radius[neighbors]
    )
    delta_unit = unit[query_idx, None] - unit[neighbors]
    delta_xy = (
        positions[neighbors, :2]
        - positions[query_idx, None, :2]
    )
    radial = positions[query_idx, :2] - bs[None, :2]
    radial /= np.maximum(
        np.linalg.norm(radial, axis=1, keepdims=True), 1e-12
    )
    tangent = np.column_stack([-radial[:, 1], radial[:, 0]])
    radial_delta = np.einsum("bkd,bd->bk", delta_xy, radial)
    tangent_delta = np.einsum("bkd,bd->bk", delta_xy, tangent)
    source_config = config["source"]
    effective = np.sqrt(
        np.square(
            radial_delta * float(source_config["radial_ratio"])
        )
        + np.square(tangent_delta)
    )
    base_weight = (
        1.0
        / np.maximum(effective, 0.05)
        ** float(source_config["distance_power"])
    )
    base_weight /= np.maximum(
        base_weight.sum(axis=1, keepdims=True), 1e-30
    )
    subcarrier = np.arange(spec.s, dtype=np.float64)
    subcarrier -= subcarrier.mean()
    phase_config = config["phase"]
    k0 = float(phase_config["k0_rad_per_meter"])
    k1 = float(
        phase_config["k1_rad_per_meter_per_subcarrier"]
    )
    steering = config["steering"]

    gram_chunks = []
    cross_chunks = []
    truth_energy_chunks = []
    feature_chunks = []
    amplitude_chunks = []
    bs_tensor = torch.as_tensor(
        bs, dtype=torch.float32, device=device
    )
    with torch.inference_mode():
        for start in range(0, len(query_idx), batch_size):
            stop = min(start + batch_size, len(query_idx))
            batch = stop - start
            source = torch.as_tensor(
                np.array(
                    channels[neighbors[start:stop].reshape(-1)],
                    copy=True,
                ).reshape(
                    batch, k, spec.m, spec.n, spec.s
                ),
                dtype=torch.complex64,
                device=device,
            )
            truth = torch.as_tensor(
                np.array(channels[query_idx[start:stop]], copy=True),
                dtype=torch.complex64,
                device=device,
            )
            phase = (
                k0 + k1 * subcarrier[None, None]
            ) * delta_radius[start:stop, :, None]
            source *= torch.as_tensor(
                np.exp(1j * phase),
                dtype=torch.complex64,
                device=device,
            )[:, :, None, None, :]
            steering_phase = array_steering_phase(
                delta_unit[start:stop], spec, steering
            )
            source *= torch.as_tensor(
                np.exp(1j * steering_phase),
                dtype=torch.complex64,
                device=device,
            )[..., None]
            flat = source.reshape(batch, k, -1)
            truth_flat = truth.reshape(batch, -1)
            gram = torch.einsum(
                "bil,bjl->bij", flat.conj(), flat
            )
            cross = torch.einsum(
                "bil,bl->bi", flat.conj(), truth_flat
            )
            truth_energy = truth_flat.abs().square().sum(dim=1)
            features, amplitude = build_complex_features(
                gram,
                torch.as_tensor(
                    base_weight[start:stop],
                    dtype=torch.float32,
                    device=device,
                ),
                torch.as_tensor(
                    distance[start:stop],
                    dtype=torch.float32,
                    device=device,
                ),
                torch.as_tensor(
                    effective[start:stop],
                    dtype=torch.float32,
                    device=device,
                ),
                torch.as_tensor(
                    radial_delta[start:stop],
                    dtype=torch.float32,
                    device=device,
                ),
                torch.as_tensor(
                    tangent_delta[start:stop],
                    dtype=torch.float32,
                    device=device,
                ),
                torch.as_tensor(
                    positions[query_idx[start:stop]],
                    dtype=torch.float32,
                    device=device,
                ),
                bs_tensor,
            )
            gram_chunks.append(gram.cpu().numpy())
            cross_chunks.append(cross.cpu().numpy())
            truth_energy_chunks.append(truth_energy.cpu().numpy())
            feature_chunks.append(features.cpu().numpy())
            amplitude_chunks.append(amplitude.cpu().numpy())
    result = GateData(
        features=np.concatenate(feature_chunks).astype(np.float32),
        amplitude=np.concatenate(amplitude_chunks).astype(np.float32),
        base_weight=base_weight.astype(np.float32),
        gram=np.concatenate(gram_chunks).astype(np.complex64),
        cross=np.concatenate(cross_chunks).astype(np.complex64),
        truth_energy=np.concatenate(truth_energy_chunks).astype(
            np.float32
        ),
    )
    print(
        f"[complex-gate-data] {name} queries={len(query_idx)} "
        f"nearest={distance[:, 0].mean():.4f}",
        flush=True,
    )
    return result


def save_data(path: Path, data: GateData) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        features=data.features,
        amplitude=data.amplitude,
        base_weight=data.base_weight,
        gram=data.gram,
        cross=data.cross,
        truth_energy=data.truth_energy,
    )


def load_data(path: Path) -> GateData:
    payload = np.load(path)
    return GateData(
        **{name: payload[name] for name in GateData.__annotations__}
    )


def select_features(
    data: GateData, feature_names: tuple[str, ...]
) -> GateData:
    indices = [FEATURE_NAMES.index(name) for name in feature_names]
    return GateData(
        features=data.features[..., indices],
        amplitude=data.amplitude,
        base_weight=data.base_weight,
        gram=data.gram,
        cross=data.cross,
        truth_energy=data.truth_energy,
    )


def tensors(
    data: GateData, device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        "features": torch.as_tensor(
            data.features, dtype=torch.float32, device=device
        ),
        "amplitude": torch.as_tensor(
            data.amplitude, dtype=torch.float32, device=device
        ),
        "base_weight": torch.as_tensor(
            data.base_weight, dtype=torch.float32, device=device
        ),
        "gram": torch.as_tensor(
            data.gram, dtype=torch.complex64, device=device
        ),
        "cross": torch.as_tensor(
            data.cross, dtype=torch.complex64, device=device
        ),
        "truth_energy": torch.as_tensor(
            data.truth_energy, dtype=torch.float32, device=device
        ),
    }


def sufficient(
    coefficient: torch.Tensor,
    data: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    prediction_energy = torch.einsum(
        "bi,bij,bj->b",
        coefficient.conj(),
        data["gram"],
        coefficient,
    ).real.clamp_min(0.0)
    cross = torch.einsum(
        "bi,bi->b", coefficient.conj(), data["cross"]
    )
    return cross, prediction_energy


def optimal_calibration(
    cross: torch.Tensor, prediction_energy: torch.Tensor
) -> tuple[float, float]:
    total_cross = complex(cross.sum().detach().cpu())
    phase = float(np.angle(total_cross))
    scale = max(
        float(
            np.real(total_cross * np.exp(-1j * phase))
            / max(float(prediction_energy.sum()), 1e-30)
        ),
        0.0,
    )
    return phase, scale


def evaluate(
    coefficient: torch.Tensor,
    data: dict[str, torch.Tensor],
    phase: float,
    scale: float,
) -> dict[str, float]:
    cross, energy = sufficient(coefficient, data)
    rotated = torch.real(
        cross
        * torch.tensor(
            np.exp(-1j * phase),
            dtype=torch.complex64,
            device=cross.device,
        )
    )
    truth_energy = data["truth_energy"]
    nmse = float(
        (
            truth_energy.sum()
            + scale**2 * energy.sum()
            - 2.0 * scale * rotated.sum()
        )
        / truth_energy.sum().clamp_min(1e-30)
    )
    coherence = float(
        rotated.sum()
        / torch.sqrt(
            energy.sum().clamp_min(1e-20)
            * truth_energy.sum().clamp_min(1e-20)
        )
    )
    return {
        "NMSE": nmse,
        "real_coherence": coherence,
        "phase": phase,
        "scale": scale,
    }


def fit_model(
    train_data: GateData,
    validation_data: GateData | None,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    regularization: float,
    seed: int,
    feature_names: tuple[str, ...] = FEATURE_NAMES,
    architecture: str = "mlp16",
) -> tuple[dict, dict]:
    train = tensors(train_data, device)
    feature_mean = train["features"].mean(dim=(0, 1))
    feature_std = train["features"].std(
        dim=(0, 1), correction=0
    ).clamp_min(1e-5)
    normalized = (
        (train["features"] - feature_mean) / feature_std
    ).clamp(-8.0, 8.0)
    torch.manual_seed(seed)
    model = ComplexNeighborGate(
        len(feature_names), architecture
    ).to(device)
    initial_coefficient = baseline_coefficients(
        train["base_weight"], train["amplitude"]
    )
    initial_cross, initial_energy = sufficient(
        initial_coefficient, train
    )
    initial_phase, initial_scale = optimal_calibration(
        initial_cross, initial_energy
    )
    log_scale = torch.nn.Parameter(
        torch.tensor(
            np.log(max(initial_scale, 1e-4)),
            dtype=torch.float32,
            device=device,
        )
    )
    phase = torch.nn.Parameter(
        torch.tensor(
            initial_phase, dtype=torch.float32, device=device
        )
    )
    optimizer = torch.optim.AdamW(
        [*model.parameters(), log_scale, phase],
        lr=learning_rate,
        weight_decay=1e-4,
    )
    model.train()
    for epoch in range(epochs):
        coefficient, diagnostic = model.coefficients(
            normalized, train["base_weight"], train["amplitude"]
        )
        cross, energy = sufficient(coefficient, train)
        scale = log_scale.exp()
        rotated = torch.real(
            cross * torch.exp(-1j * phase).to(torch.complex64)
        )
        truth_energy = train["truth_energy"]
        nmse = (
            truth_energy.sum()
            + scale.square() * energy.sum()
            - 2.0 * scale * rotated.sum()
        ) / truth_energy.sum().clamp_min(1e-30)
        coherence = rotated.sum() / torch.sqrt(
            energy.sum().clamp_min(1e-20)
            * truth_energy.sum().clamp_min(1e-20)
        )
        penalty = (
            diagnostic["logit_delta"].square().mean()
            + diagnostic["phase_delta"].square().mean()
            + diagnostic["amplitude_factor"].log().square().mean()
        )
        loss = nmse + 0.03 * (1.0 - coherence) + regularization * penalty
        if not torch.isfinite(loss):
            raise FloatingPointError(
                "non-finite complex-neighbor gate objective"
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [*model.parameters(), log_scale, phase], 2.0
        )
        optimizer.step()
        with torch.no_grad():
            log_scale.clamp_(-2.0, 2.0)
            phase.clamp_(-np.pi, np.pi)
        if epoch in {0, epochs - 1}:
            print(
                f"[complex-gate-train] epoch={epoch + 1}/{epochs} "
                f"nmse={float(nmse):.6f} coh={float(coherence):.6f} "
                f"scale={float(scale):.4f} phase={float(phase):.4f}",
                flush=True,
            )

    model.eval()
    with torch.inference_mode():
        learned_coefficient, diagnostic = model.coefficients(
            normalized, train["base_weight"], train["amplitude"]
        )
        baseline_coefficient = baseline_coefficients(
            train["base_weight"], train["amplitude"]
        )
        baseline_cross, baseline_energy = sufficient(
            baseline_coefficient, train
        )
        baseline_phase, baseline_scale = optimal_calibration(
            baseline_cross, baseline_energy
        )
        learned_train = evaluate(
            learned_coefficient,
            train,
            float(phase),
            float(log_scale.exp()),
        )
        baseline_train = evaluate(
            baseline_coefficient,
            train,
            baseline_phase,
            baseline_scale,
        )
        validation_rows = None
        if validation_data is not None:
            validation = tensors(validation_data, device)
            validation_normalized = (
                (validation["features"] - feature_mean) / feature_std
            ).clamp(-8.0, 8.0)
            learned_validation_coefficient, validation_diagnostic = (
                model.coefficients(
                    validation_normalized,
                    validation["base_weight"],
                    validation["amplitude"],
                )
            )
            baseline_validation_coefficient = baseline_coefficients(
                validation["base_weight"], validation["amplitude"]
            )
            validation_rows = {
                "baseline": evaluate(
                    baseline_validation_coefficient,
                    validation,
                    baseline_phase,
                    baseline_scale,
                ),
                "learned": evaluate(
                    learned_validation_coefficient,
                    validation,
                    float(phase),
                    float(log_scale.exp()),
                ),
                "amplitude_factor_mean": float(
                    validation_diagnostic[
                        "amplitude_factor"
                    ].mean()
                ),
                "amplitude_factor_std": float(
                    validation_diagnostic[
                        "amplitude_factor"
                    ].std(correction=0)
                ),
                "phase_delta_std": float(
                    validation_diagnostic[
                        "phase_delta"
                    ].std(correction=0)
                ),
            }
    checkpoint = {
        "model_state": model.state_dict(),
        "architecture": architecture,
        "feature_names": list(feature_names),
        "feature_mean": feature_mean.detach().cpu().numpy(),
        "feature_std": feature_std.detach().cpu().numpy(),
        "global_phase": float(phase),
        "global_scale": float(log_scale.exp()),
    }
    result = {
        "training": {
            "baseline": baseline_train,
            "learned": learned_train,
            "amplitude_factor_mean": float(
                diagnostic["amplitude_factor"].mean()
            ),
            "amplitude_factor_std": float(
                diagnostic["amplitude_factor"].std(correction=0)
            ),
        },
        "validation": validation_rows,
    }
    return checkpoint, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", default="Round1_Map(2)")
    parser.add_argument(
        "--config", default="configs/gs43_dual_mix10.json"
    )
    parser.add_argument("--tune-seeds", default="1890,3716,962,1022")
    parser.add_argument("--audit-seed", default="2262")
    parser.add_argument("--external-indices")
    parser.add_argument("--external-name", default="testmatched")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--channel-batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--regularization", type=float, default=0.01)
    parser.add_argument(
        "--feature-set",
        choices=("full", "invariant"),
        default="full",
        help=(
            "invariant removes absolute query radius/angle while keeping "
            "relative geometry and channel-agreement features"
        ),
    )
    parser.add_argument(
        "--architecture",
        choices=ARCHITECTURES,
        default="mlp16",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--cache-dir", default="cache/complex_neighbor_gate"
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints/complex_neighbor_gate",
    )
    parser.add_argument(
        "--out", default="docs/complex_neighbor_gate/result.json"
    )
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    datadir = Path(args.datadir)
    spec = load_setup(datadir / "Round1_Setup.json")
    positions = np.load(
        datadir / "Round1_Train_Pos.npy"
    ).astype(np.float64)
    channels = np.load(
        datadir / "Round1_Train_Channel.npy", mmap_mode="r"
    )
    all_idx = np.arange(len(positions), dtype=np.int64)
    tune = [value for value in args.tune_seeds.split(",") if value]
    names = tune + [args.audit_seed]
    if args.external_indices:
        names.append(args.external_name)
    cache_dir = Path(args.cache_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    device = torch.device(args.device)
    feature_names = (
        FEATURE_NAMES
        if args.feature_set == "full"
        else tuple(
            name
            for name in FEATURE_NAMES
            if name
            not in {
                "query_radius",
                "query_angle_sin",
                "query_angle_cos",
            }
        )
    )

    def cached_data(
        cache_name: str,
        query_idx: np.ndarray,
        pool_idx: np.ndarray,
        leave_self_out: bool,
    ) -> GateData:
        path = cache_dir / f"{cache_name}.npz"
        if path.exists() and not args.rebuild_cache:
            return load_data(path)
        data = build_data(
            cache_name,
            query_idx,
            pool_idx,
            leave_self_out,
            positions,
            channels,
            spec,
            config,
            args.k,
            args.channel_batch_size,
            device,
        )
        save_data(path, data)
        return data

    rows = {}
    for split_number, name in enumerate(names):
        val_idx = split_indices(
            len(positions),
            name,
            args.external_name,
            args.external_indices,
        )
        pool_idx = np.setdiff1d(all_idx, val_idx)
        training = select_features(
            cached_data(
                f"{name}_train", pool_idx, pool_idx, True
            ),
            feature_names,
        )
        validation = select_features(
            cached_data(
                f"{name}_val", val_idx, pool_idx, False
            ),
            feature_names,
        )
        checkpoint, row = fit_model(
            training,
            validation,
            device,
            args.epochs,
            args.learning_rate,
            args.regularization,
            5101 + split_number,
            feature_names,
            args.architecture,
        )
        checkpoint["metadata"] = {
            "split": name,
            "k": args.k,
            "selection_policy": (
                "training pool excludes the complete validation split"
            ),
        }
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_dir / f"s{name}.pt")
        rows[name] = row
        validation_row = row["validation"]
        print(
            f"[complex-gate-val] split={name} "
            f"baseline={validation_row['baseline']['NMSE']:.6f} "
            f"learned={validation_row['learned']['NMSE']:.6f} "
            f"delta={validation_row['learned']['NMSE'] - validation_row['baseline']['NMSE']:+.6f}",
            flush=True,
        )
        torch.cuda.empty_cache()

    final_training = select_features(
        cached_data(
            "final_train", all_idx, all_idx, True
        ),
        feature_names,
    )
    final_checkpoint, final_row = fit_model(
        final_training,
        None,
        device,
        args.epochs,
        args.learning_rate,
        args.regularization,
        6101,
        feature_names,
        args.architecture,
    )
    final_checkpoint["metadata"] = {
        "split": "full_leave_one_out",
        "k": args.k,
        "selection_policy": (
            "hyperparameters frozen before final all-data leave-one-out fit"
        ),
    }
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_path = checkpoint_dir / "selected.pt"
    torch.save(final_checkpoint, final_path)

    tune_delta = [
        rows[name]["validation"]["learned"]["NMSE"]
        - rows[name]["validation"]["baseline"]["NMSE"]
        for name in tune
    ]
    payload = {
        "selection_policy": (
            "one fixed low-capacity architecture; four tune splits and "
            "audit/external use independently trained clean models"
        ),
        "tune_seeds": tune,
        "audit_seed": args.audit_seed,
        "external_name": (
            args.external_name if args.external_indices else None
        ),
        "k": args.k,
        "epochs": args.epochs,
        "regularization": args.regularization,
        "feature_set": args.feature_set,
        "architecture": args.architecture,
        "feature_names": list(feature_names),
        "tune_delta_nmse": {
            "median": float(np.median(tune_delta)),
            "mean": float(np.mean(tune_delta)),
            "worst": float(np.max(tune_delta)),
        },
        "splits": rows,
        "final_training": final_row,
        "final_checkpoint": str(final_path),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"COMPLEX_NEIGHBOR_GATE_DONE out={out} checkpoint={final_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
