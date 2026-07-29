#!/usr/bin/env python3
"""Train clean complex-correlation predictors for regularized kriging."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from _bootstrap import ROOT  # noqa: F401
from complex_neighbor_gate import FEATURE_NAMES


INVARIANT_EXCLUSIONS = {
    "query_radius",
    "query_angle_sin",
    "query_angle_cos",
}


class CorrelationPredictor(nn.Module):
    """Predict the target-to-neighbor correlation residual as a set."""

    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(feature_dim + 2, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.output = nn.Linear(hidden_dim, 2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        features: torch.Tensor,
        prior: torch.Tensor,
    ) -> torch.Tensor:
        augmented = torch.cat(
            [features, torch.view_as_real(prior)], dim=-1
        )
        hidden = self.encoder(self.input_projection(augmented))
        residual = torch.tanh(self.output(hidden))
        return prior + torch.view_as_complex(residual.contiguous())


def parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item]


def load_data(
    path: Path,
    feature_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    payload = np.load(path)
    amplitude = payload["amplitude"].astype(np.float64)
    gram = payload["gram"].astype(np.complex128)
    truth_energy = payload["truth_energy"].astype(np.float64)
    correlation = gram / np.maximum(
        amplitude[:, :, None] * amplitude[:, None, :],
        1e-30,
    )
    target_correlation = payload["cross"].astype(np.complex128) / np.maximum(
        amplitude * np.sqrt(truth_energy[:, None]),
        1e-30,
    )
    base_weight = payload["base_weight"].astype(np.float64)
    prior = np.einsum(
        "bij,bj->bi", correlation, base_weight
    )
    return {
        "features": payload["features"][..., feature_indices].astype(
            np.float32
        ),
        "amplitude": amplitude,
        "base_weight": base_weight,
        "gram": gram,
        "cross": payload["cross"].astype(np.complex128),
        "truth_energy": truth_energy,
        "correlation": correlation,
        "target_correlation": target_correlation,
        "prior": prior,
    }


def train_model(
    train: dict[str, np.ndarray],
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    hidden_dim: int,
    seed: int,
) -> tuple[dict[str, object], dict[str, float]]:
    features = torch.as_tensor(
        train["features"], dtype=torch.float32, device=device
    )
    prior = torch.as_tensor(
        train["prior"], dtype=torch.complex64, device=device
    )
    target = torch.as_tensor(
        train["target_correlation"],
        dtype=torch.complex64,
        device=device,
    )
    feature_mean = features.mean(dim=(0, 1))
    feature_std = features.std(
        dim=(0, 1), correction=0
    ).clamp_min(1e-5)
    normalized = (
        (features - feature_mean) / feature_std
    ).clamp(-8.0, 8.0)

    torch.manual_seed(seed)
    model = CorrelationPredictor(
        normalized.shape[-1], hidden_dim
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1009)
    baseline_mse = float((prior - target).abs().square().mean())
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(
            len(features), generator=generator
        )
        total = 0.0
        seen = 0
        for start in range(0, len(features), batch_size):
            index = permutation[start : start + batch_size].to(device)
            prediction = model(normalized[index], prior[index])
            difference = torch.view_as_real(
                prediction - target[index]
            )
            loss = F.smooth_l1_loss(
                difference,
                torch.zeros_like(difference),
                beta=0.15,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total += float(loss) * len(index)
            seen += len(index)
        if epoch in {0, epochs - 1} or (epoch + 1) % 25 == 0:
            print(
                f"[cov-kriging] epoch={epoch + 1}/{epochs} "
                f"loss={total / max(seen, 1):.6f}",
                flush=True,
            )

    model.eval()
    with torch.inference_mode():
        prediction = model(normalized, prior)
        train_mse = float(
            (prediction - target).abs().square().mean()
        )
    checkpoint = {
        "model_state": model.state_dict(),
        "feature_mean": feature_mean.detach().cpu().numpy(),
        "feature_std": feature_std.detach().cpu().numpy(),
        "feature_dim": int(normalized.shape[-1]),
        "hidden_dim": hidden_dim,
    }
    diagnostics = {
        "baseline_correlation_mse": baseline_mse,
        "learned_correlation_mse": train_mse,
    }
    return checkpoint, diagnostics


def predict(
    checkpoint: dict[str, object],
    data: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model = CorrelationPredictor(
        int(checkpoint["feature_dim"]),
        int(checkpoint["hidden_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    mean = torch.as_tensor(
        checkpoint["feature_mean"],
        dtype=torch.float32,
        device=device,
    )
    std = torch.as_tensor(
        checkpoint["feature_std"],
        dtype=torch.float32,
        device=device,
    )
    features = torch.as_tensor(
        data["features"], dtype=torch.float32, device=device
    )
    prior = torch.as_tensor(
        data["prior"], dtype=torch.complex64, device=device
    )
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            normalized = (
                (features[start : start + batch_size] - mean) / std
            ).clamp(-8.0, 8.0)
            chunks.append(
                model(
                    normalized,
                    prior[start : start + batch_size],
                ).cpu()
            )
    return torch.cat(chunks).numpy().astype(np.complex128)


def coefficients(
    data: dict[str, np.ndarray],
    predicted_correlation: np.ndarray,
    diagonal_loading: float,
    blend: float,
) -> np.ndarray:
    correlation = data["correlation"]
    base = data["base_weight"]
    k = correlation.shape[1]
    identity = np.eye(k, dtype=np.complex128)[None]
    system = correlation + diagonal_loading * identity
    rhs = predicted_correlation + diagonal_loading * base
    solved = np.linalg.solve(system, rhs[..., None])[..., 0]
    normalized = base + blend * (solved - base)
    target_amplitude = (
        base * data["amplitude"]
    ).sum(axis=1, keepdims=True)
    return (
        normalized
        * target_amplitude
        / np.maximum(data["amplitude"], 1e-30)
    )


def sufficient(
    coefficient: np.ndarray,
    data: dict[str, np.ndarray],
) -> dict[str, complex | float]:
    cross = np.einsum(
        "bi,bi->b", coefficient.conj(), data["cross"]
    )
    energy = np.real(
        np.einsum(
            "bi,bij,bj->b",
            coefficient.conj(),
            data["gram"],
            coefficient,
        )
    )
    return {
        "cross": complex(cross.sum()),
        "prediction_energy": float(energy.sum()),
        "truth_energy": float(data["truth_energy"].sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir", default="cache/complex_neighbor_gate_probe"
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints/covariance_kriging",
    )
    parser.add_argument(
        "--out", default="docs/covariance_kriging/result.json"
    )
    parser.add_argument(
        "--tune-seeds", default="1890,3716,962,1022"
    )
    parser.add_argument("--audit-seed", default="2262")
    parser.add_argument(
        "--external-names",
        default="testmatched,geomp0,geomp1,geomp2",
        help="clean geometry-shift splits used only as diagnostics",
    )
    parser.add_argument(
        "--feature-set", choices=("full", "invariant"), default="full"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--train-final",
        action="store_true",
        help="train one deployable model from final_train.npz and exit",
    )
    parser.add_argument(
        "--diagonal-loading-grid", default="0.03,0.1,0.3,1,3"
    )
    parser.add_argument(
        "--blend-grid", default="0,0.05,0.1,0.2,0.3,0.5,0.75,1"
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    tune = [
        value for value in args.tune_seeds.split(",") if value
    ]
    external = [
        value for value in args.external_names.split(",") if value
    ]
    names = tune + [args.audit_seed] + external
    feature_names = (
        FEATURE_NAMES
        if args.feature_set == "full"
        else tuple(
            name
            for name in FEATURE_NAMES
            if name not in INVARIANT_EXCLUSIONS
        )
    )
    feature_indices = np.asarray(
        [FEATURE_NAMES.index(name) for name in feature_names],
        dtype=np.int64,
    )
    cache_dir = Path(args.cache_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if args.train_final:
        training = load_data(
            cache_dir / "final_train.npz", feature_indices
        )
        checkpoint, diagnostics = train_model(
            training,
            device,
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.weight_decay,
            args.hidden_dim,
            seed=8301,
        )
        checkpoint["feature_names"] = list(feature_names)
        checkpoint["split"] = "final"
        checkpoint_path = checkpoint_dir / "selected.pt"
        torch.save(checkpoint, checkpoint_path)
        payload = {
            "selection_policy": (
                "architecture and epoch count frozen on the clean panel; "
                "final model uses all training rows with leave-one-out "
                "neighbors"
            ),
            "feature_set": args.feature_set,
            "feature_names": list(feature_names),
            "training": diagnostics,
            "checkpoint": str(checkpoint_path),
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        print(
            f"COVARIANCE_KRIGING_FINAL_DONE checkpoint={checkpoint_path} "
            f"out={out}",
            flush=True,
        )
        return
    rows: dict[str, dict[str, object]] = {}

    for split_number, name in enumerate(names):
        print(f"[cov-kriging] split={name}", flush=True)
        training = load_data(
            cache_dir / f"{name}_train.npz", feature_indices
        )
        validation = load_data(
            cache_dir / f"{name}_val.npz", feature_indices
        )
        checkpoint, diagnostics = train_model(
            training,
            device,
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.weight_decay,
            args.hidden_dim,
            seed=7301 + split_number,
        )
        checkpoint["feature_names"] = list(feature_names)
        checkpoint["split"] = name
        checkpoint_path = checkpoint_dir / f"s{name}.pt"
        torch.save(checkpoint, checkpoint_path)
        predicted = predict(
            checkpoint, validation, device, args.batch_size
        )
        target = validation["target_correlation"]
        prior = validation["prior"]
        rows[name] = {
            "checkpoint": str(checkpoint_path),
            "training": diagnostics,
            "validation_correlation_mse": float(
                np.mean(np.abs(predicted - target) ** 2)
            ),
            "baseline_correlation_mse": float(
                np.mean(np.abs(prior - target) ** 2)
            ),
            "data": validation,
            "predicted": predicted,
        }
        print(
            f"[cov-kriging] split={name} corr_mse "
            f"{rows[name]['baseline_correlation_mse']:.6f} -> "
            f"{rows[name]['validation_correlation_mse']:.6f}",
            flush=True,
        )

    loadings = parse_floats(args.diagonal_loading_grid)
    blends = parse_floats(args.blend_grid)
    candidates = []
    for loading in loadings:
        for blend in blends:
            stats = {}
            for name in names:
                row = rows[name]
                coefficient = coefficients(
                    row["data"],
                    row["predicted"],
                    loading,
                    blend,
                )
                stats[name] = sufficient(
                    coefficient, row["data"]
                )
            tune_cross = sum(stats[name]["cross"] for name in tune)
            tune_energy = sum(
                stats[name]["prediction_energy"] for name in tune
            )
            phase = float(np.angle(tune_cross))
            rotation = np.exp(-1j * phase)
            scale = max(
                float(np.real(tune_cross * rotation))
                / max(tune_energy, 1e-30),
                0.0,
            )
            scores = {}
            for name in names:
                row = stats[name]
                cross_real = float(
                    np.real(row["cross"] * rotation)
                )
                scores[name] = float(
                    (
                        row["truth_energy"]
                        + scale**2 * row["prediction_energy"]
                        - 2.0 * scale * cross_real
                    )
                    / max(row["truth_energy"], 1e-30)
                )
            tune_scores = [scores[name] for name in tune]
            candidates.append(
                {
                    "diagonal_loading": loading,
                    "blend": blend,
                    "global_phase": phase,
                    "scale": scale,
                    "tune_mean_nmse": float(np.mean(tune_scores)),
                    "tune_median_nmse": float(
                        np.median(tune_scores)
                    ),
                    "tune_worst_nmse": float(np.max(tune_scores)),
                    "audit_nmse": scores[args.audit_seed],
                    "external_nmse": {
                        name: scores[name] for name in external
                    },
                    "scores": scores,
                }
            )
    candidates.sort(
        key=lambda row: (
            row["tune_mean_nmse"],
            row["tune_worst_nmse"],
            row["tune_median_nmse"],
        )
    )
    baseline = next(
        row
        for row in candidates
        if row["blend"] == 0.0
        and row["diagonal_loading"] == loadings[0]
    )
    for row in candidates:
        row["tune_delta_nmse"] = {
            name: row["scores"][name] - baseline["scores"][name]
            for name in tune
        }
        row["audit_delta_nmse"] = (
            row["audit_nmse"] - baseline["audit_nmse"]
        )
        row["external_delta_nmse"] = {
            name: row["external_nmse"][name]
            - baseline["external_nmse"][name]
            for name in external
        }
    robust = [
        row
        for row in candidates
        if max(row["tune_delta_nmse"].values()) <= 0.0
    ]
    output_rows = {
        name: {
            key: value
            for key, value in row.items()
            if key not in {"data", "predicted"}
        }
        for name, row in rows.items()
    }
    payload = {
        "selection_policy": (
            "models train only on each split's clean pool; loading and "
            "blend rank on four tune splits; audit is diagnostic only"
        ),
        "feature_set": args.feature_set,
        "feature_names": list(feature_names),
        "tune_seeds": tune,
        "audit_seed": args.audit_seed,
        "external_names": external,
        "grid": {
            "diagonal_loading": loadings,
            "blend": blends,
        },
        "splits": output_rows,
        "baseline": baseline,
        "best": candidates[0],
        "best_robust": robust[0] if robust else None,
        "ranked": candidates,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "baseline": baseline,
                "best": candidates[0],
                "best_robust": robust[0] if robust else None,
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"COVARIANCE_KRIGING_DONE out={out}", flush=True)


if __name__ == "__main__":
    main()
