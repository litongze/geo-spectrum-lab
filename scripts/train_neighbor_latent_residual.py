#!/usr/bin/env python3
"""Train a clean grouped-neighbor model in a low-rank spectrum space."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn
import torch.nn.functional as F

from _bootstrap import ROOT  # noqa: F401
from score_holdout import reproduce_val_indices
from wireless_twin.data.setup_config import load_setup


def stable_unit(value: torch.Tensor) -> torch.Tensor:
    maximum = value.amax(dim=-1, keepdim=True)
    scaled = torch.where(
        maximum > 0,
        value / maximum.clamp_min(torch.finfo(value.dtype).tiny),
        torch.zeros_like(value),
    )
    return scaled / scaled.norm(dim=-1, keepdim=True).clamp_min(
        torch.finfo(value.dtype).tiny
    )


def spectrum_cosine(
    prediction: torch.Tensor, truth: torch.Tensor
) -> float:
    return float(
        (
            stable_unit(prediction)
            * stable_unit(truth)
        ).sum(dim=-1).mean()
    )


def neighbor_geometry(
    positions: np.ndarray,
    query_idx: np.ndarray,
    neighbors: np.ndarray,
    distance: np.ndarray,
    bs: np.ndarray,
    power: float,
) -> tuple[np.ndarray, np.ndarray]:
    delta = (
        positions[neighbors, :2]
        - positions[query_idx, None, :2]
    )
    scale = np.maximum(
        np.median(distance, axis=1, keepdims=True), 0.3
    )
    radial = positions[query_idx, :2] - bs[None, :2]
    radial /= np.maximum(
        np.linalg.norm(radial, axis=1, keepdims=True), 1e-6
    )
    tangent = np.column_stack([-radial[:, 1], radial[:, 0]])
    radial_delta = np.einsum("bkd,bd->bk", delta, radial)
    tangent_delta = np.einsum("bkd,bd->bk", delta, tangent)
    base_weight = (
        1.0 / np.maximum(distance, 0.05) ** power
    )
    base_weight /= np.maximum(
        base_weight.sum(axis=1, keepdims=True), 1e-30
    )
    rank = np.arange(neighbors.shape[1], dtype=np.float32)
    rank /= max(neighbors.shape[1] - 1, 1)
    rank = np.broadcast_to(rank[None], distance.shape)
    geometry = np.stack(
        [
            np.log(np.maximum(distance, 0.05)),
            distance / scale,
            delta[..., 0] / scale,
            delta[..., 1] / scale,
            radial_delta / scale,
            tangent_delta / scale,
            rank,
            np.log(np.maximum(base_weight, 1e-30)),
        ],
        axis=-1,
    ).astype(np.float32)
    return geometry, base_weight.astype(np.float32)


class GroupedNeighborLatent(nn.Module):
    """Use several shared neighbor gates for groups of latent factors."""

    def __init__(
        self,
        rank: int,
        geometry_dim: int,
        heads: int,
        hidden: int,
        residual_bound: float,
    ) -> None:
        super().__init__()
        content_dim = 24
        self.content = nn.Sequential(
            nn.Linear(rank, 48),
            nn.SiLU(),
            nn.Linear(48, content_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(geometry_dim + content_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, heads),
        )
        self.assignment = nn.Parameter(torch.zeros(rank, heads))
        self.residual = nn.Sequential(
            nn.LayerNorm(rank * 3),
            nn.Linear(rank * 3, hidden * 2),
            nn.SiLU(),
            nn.Linear(hidden * 2, hidden * 2),
            nn.SiLU(),
            nn.Linear(hidden * 2, rank),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.residual_bound = residual_bound

    def forward(
        self,
        neighbor_coefficients: torch.Tensor,
        geometry: torch.Tensor,
        base_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        base = torch.einsum(
            "bk,bkr->br", base_weight, neighbor_coefficients
        )
        centered = neighbor_coefficients - base[:, None]
        content = self.content(centered)
        logit_delta = 0.75 * torch.tanh(
            self.gate(torch.cat([geometry, content], dim=-1))
        )
        weight = torch.softmax(
            base_weight.clamp_min(1e-30).log()[..., None]
            + logit_delta,
            dim=1,
        )
        head_value = torch.einsum(
            "bkh,bkr->bhr", weight, neighbor_coefficients
        )
        assignment = torch.softmax(self.assignment, dim=1)
        grouped = torch.einsum(
            "rh,bhr->br", assignment, head_value
        )
        variance = torch.einsum(
            "bk,bkr->br", base_weight, centered.square()
        ).clamp_min(1e-6)
        local_std = variance.sqrt()
        residual_input = torch.cat(
            [
                grouped,
                neighbor_coefficients[:, 0] - grouped,
                local_std,
            ],
            dim=1,
        )
        residual = (
            self.residual_bound
            * local_std
            * torch.tanh(self.residual(residual_input))
        )
        return grouped + residual, {
            "logit_delta": logit_delta,
            "residual": residual,
            "weight": weight,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", default="Round1_Map(2)")
    parser.add_argument(
        "--cache-dir", default="cache/teammate_knn_hvp"
    )
    parser.add_argument(
        "--domain", choices=("pas_phv", "pas_hvp", "pdp"),
        default="pas_phv",
    )
    parser.add_argument("--split-seed", type=int, default=1890)
    parser.add_argument("--val-indices")
    parser.add_argument("--val-name")
    parser.add_argument("--expert-val")
    parser.add_argument("--rank", type=int, default=96)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--distance-power", type=float, default=2.0)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--residual-bound", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--loss", choices=("spectrum", "coefficient"),
        default="spectrum",
    )
    parser.add_argument("--loss-slices", type=int, default=32)
    parser.add_argument(
        "--coefficient-aux-weight", type=float, default=0.02
    )
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--pca-iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--select-final-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out",
        default="docs/neighbor_latent_residual/result.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    datadir = Path(args.datadir)
    spec = load_setup(datadir / "Round1_Setup.json")
    positions = np.load(
        datadir / "Round1_Train_Pos.npy"
    ).astype(np.float32)
    if args.val_indices:
        val_idx = np.asarray(
            sorted(
                np.load(args.val_indices).astype(np.int64).tolist()
            ),
            dtype=np.int64,
        )
    else:
        val_idx = np.asarray(
            sorted(
                reproduce_val_indices(
                    len(positions), 0.1, args.split_seed
                )
            ),
            dtype=np.int64,
        )
    all_idx = np.arange(len(positions), dtype=np.int64)
    train_idx = np.setdiff1d(all_idx, val_idx)
    split_name = args.val_name or str(args.split_seed)

    cache_names = {
        "pas_phv": "train_pas_phv.npy",
        "pas_hvp": "train_pas_hvp.npy",
        "pdp": "train_pdp.npy",
    }
    raw_spectra = np.load(
        Path(args.cache_dir) / cache_names[args.domain],
        mmap_mode="r",
    )
    spectra = stable_unit(
        torch.as_tensor(
            np.array(raw_spectra, copy=True),
            dtype=torch.float32,
            device=device,
        )
    )
    original_shape = tuple(spectra.shape[1:])
    flat = spectra.reshape(len(spectra), -1)
    train_tensor = torch.as_tensor(
        train_idx, dtype=torch.long, device=device
    )
    val_tensor = torch.as_tensor(
        val_idx, dtype=torch.long, device=device
    )

    print(
        f"[latent-neighbor] PCA split={split_name} "
        f"train={len(train_idx)} rank={args.rank} "
        f"flat={flat.shape[1]}",
        flush=True,
    )
    train_flat = flat[train_tensor]
    spectrum_mean = train_flat.mean(dim=0)
    centered = train_flat - spectrum_mean
    _, singular, components = torch.pca_lowrank(
        centered,
        q=args.rank,
        center=False,
        niter=args.pca_iterations,
    )
    coefficients = (flat - spectrum_mean) @ components
    coefficient_mean = coefficients[train_tensor].mean(dim=0)
    coefficient_std = coefficients[train_tensor].std(
        dim=0
    ).clamp_min(1e-6)
    normalized_coefficients = (
        coefficients - coefficient_mean
    ) / coefficient_std

    tree = cKDTree(positions[train_idx, :2])
    train_distance, train_local = tree.query(
        positions[train_idx, :2], k=args.k + 1
    )
    train_neighbors = train_idx[np.asarray(train_local)[:, 1:]]
    train_distance = np.asarray(train_distance)[:, 1:]
    val_distance, val_local = tree.query(
        positions[val_idx, :2], k=args.k
    )
    val_neighbors = train_idx[np.asarray(val_local)]
    val_distance = np.asarray(val_distance)
    bs = np.asarray(spec.bs_position, dtype=np.float32)
    train_geometry, train_base_weight = neighbor_geometry(
        positions,
        train_idx,
        train_neighbors,
        train_distance,
        bs,
        args.distance_power,
    )
    val_geometry, val_base_weight = neighbor_geometry(
        positions,
        val_idx,
        val_neighbors,
        val_distance,
        bs,
        args.distance_power,
    )

    train_neighbor_t = torch.as_tensor(
        train_neighbors, dtype=torch.long, device=device
    )
    val_neighbor_t = torch.as_tensor(
        val_neighbors, dtype=torch.long, device=device
    )
    train_geometry_t = torch.as_tensor(
        train_geometry, dtype=torch.float32, device=device
    )
    val_geometry_t = torch.as_tensor(
        val_geometry, dtype=torch.float32, device=device
    )
    train_base_t = torch.as_tensor(
        train_base_weight, dtype=torch.float32, device=device
    )
    val_base_t = torch.as_tensor(
        val_base_weight, dtype=torch.float32, device=device
    )

    expert = None
    if args.expert_val:
        expert_np = np.load(args.expert_val, mmap_mode="r")
        if tuple(expert_np.shape) != (
            len(val_idx),
            *original_shape,
        ):
            raise ValueError(
                "expert spectrum shape mismatch: "
                f"{expert_np.shape} != "
                f"{(len(val_idx), *original_shape)}"
            )
        expert = stable_unit(
            torch.as_tensor(
                np.array(expert_np, copy=True),
                dtype=torch.float32,
                device=device,
            )
        )

    model = GroupedNeighborLatent(
        args.rank,
        train_geometry.shape[-1],
        args.heads,
        args.hidden,
        args.residual_bound,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05
    )
    train_target = normalized_coefficients[train_tensor]
    slice_length = original_shape[-1]
    slice_count = int(np.prod(original_shape[:-1]))
    if args.loss_slices > slice_count:
        raise ValueError(
            f"loss_slices={args.loss_slices} exceeds {slice_count}"
        )
    coefficient_loss_weight = coefficient_std.square()
    coefficient_loss_weight /= coefficient_loss_weight.mean()

    def reconstruct(
        normalized: torch.Tensor,
    ) -> torch.Tensor:
        coefficient = (
            normalized * coefficient_std + coefficient_mean
        )
        value = (
            spectrum_mean + coefficient @ components.T
        ).clamp_min(0.0)
        return stable_unit(
            value.reshape(len(normalized), *original_shape)
        )

    def evaluate() -> dict:
        model.eval()
        predictions = []
        with torch.inference_mode():
            for start in range(0, len(val_idx), 32):
                stop = min(start + 32, len(val_idx))
                prediction, _ = model(
                    normalized_coefficients[
                        val_neighbor_t[start:stop]
                    ],
                    val_geometry_t[start:stop],
                    val_base_t[start:stop],
                )
                predictions.append(prediction)
            predicted_coefficients = torch.cat(predictions)
            base_coefficients = torch.einsum(
                "bk,bkr->br",
                val_base_t,
                normalized_coefficients[val_neighbor_t],
            )
            truth = spectra[val_tensor]
            pca_truth = reconstruct(
                normalized_coefficients[val_tensor]
            )
            latent_prediction = reconstruct(predicted_coefficients)
            latent_base = reconstruct(base_coefficients)
            direct_chunks = []
            for start in range(0, len(val_idx), 16):
                stop = min(start + 16, len(val_idx))
                direct_chunks.append(
                    stable_unit(
                        torch.einsum(
                            "bk,bk...->b...",
                            val_base_t[start:stop],
                            spectra[
                                val_neighbor_t[start:stop]
                            ],
                        )
                    )
                )
            direct = torch.cat(direct_chunks)
            rows = []
            if expert is not None:
                for beta in (
                    0.0,
                    0.025,
                    0.05,
                    0.1,
                    0.15,
                    0.2,
                    0.3,
                    0.5,
                    1.0,
                ):
                    blended = stable_unit(
                        (1.0 - beta) * expert
                        + beta * latent_prediction
                    )
                    rows.append(
                        {
                            "beta": beta,
                            "score": spectrum_cosine(
                                blended, truth
                            ),
                        }
                    )
            selected = (
                max(rows, key=lambda row: row["score"])
                if rows
                else {
                    "beta": 1.0,
                    "score": spectrum_cosine(
                        latent_prediction, truth
                    ),
                }
            )
            return {
                "pca_upper": spectrum_cosine(pca_truth, truth),
                "direct_idw": spectrum_cosine(direct, truth),
                "latent_idw": spectrum_cosine(latent_base, truth),
                "latent_model": spectrum_cosine(
                    latent_prediction, truth
                ),
                "expert": (
                    spectrum_cosine(expert, truth)
                    if expert is not None
                    else None
                ),
                "blend_rows": rows,
                "selected": selected,
                "predicted_coefficients": (
                    predicted_coefficients.detach().cpu()
                ),
            }

    best_score = -float("inf")
    best_state = None
    history = []
    initial_metrics = evaluate()
    initial_row = {
        "epoch": 0,
        "train_loss": None,
        **{
            key: value
            for key, value in initial_metrics.items()
            if key != "predicted_coefficients"
        },
    }
    history.append(initial_row)
    if not args.select_final_only:
        best_score = initial_row["selected"]["score"]
        best_state = {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        }
        best_row = initial_row
    print(
        f"[latent-neighbor] epoch=0 "
        f"model={initial_row['latent_model']:.6f} "
        f"blend={initial_row['selected']['score']:.6f}"
        f"@{initial_row['selected']['beta']:g}",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train_idx), device=device)
        losses = []
        for start in range(0, len(order), args.batch_size):
            batch = order[start : start + args.batch_size]
            prediction, diagnostics = model(
                normalized_coefficients[
                    train_neighbor_t[batch]
                ],
                train_geometry_t[batch],
                train_base_t[batch],
            )
            coefficient_loss = (
                coefficient_loss_weight
                * (
                    prediction - train_target[batch]
                ).square()
            ).mean()
            if args.loss == "spectrum":
                selected_slices = torch.randperm(
                    slice_count, device=device
                )[: args.loss_slices]
                selected_columns = (
                    selected_slices[:, None] * slice_length
                    + torch.arange(
                        slice_length, device=device
                    )[None]
                ).reshape(-1)
                raw_coefficient = (
                    prediction * coefficient_std
                    + coefficient_mean
                )
                predicted_spectrum = (
                    spectrum_mean[selected_columns]
                    + raw_coefficient
                    @ components[selected_columns].T
                ).clamp_min(0.0)
                predicted_spectrum = stable_unit(
                    predicted_spectrum.reshape(
                        len(batch),
                        args.loss_slices,
                        slice_length,
                    )
                )
                target_spectrum = spectra[
                    train_tensor[batch]
                ].reshape(
                    len(batch), slice_count, slice_length
                )[:, selected_slices]
                data_loss = (
                    1.0
                    - (
                        predicted_spectrum * target_spectrum
                    ).sum(dim=-1).mean()
                    + args.coefficient_aux_weight
                    * coefficient_loss
                )
            else:
                data_loss = coefficient_loss
            regularization = (
                1e-3
                * diagnostics["logit_delta"].square().mean()
                + 1e-3
                * diagnostics["residual"].square().mean()
            )
            loss = data_loss + regularization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 2.0
            )
            optimizer.step()
            losses.append(float(data_loss))
        scheduler.step()

        if (
            epoch == 1
            or epoch % args.eval_every == 0
            or epoch == args.epochs
        ):
            metrics = evaluate()
            row = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                **{
                    key: value
                    for key, value in metrics.items()
                    if key != "predicted_coefficients"
                },
            }
            history.append(row)
            eligible = (
                not args.select_final_only or epoch == args.epochs
            )
            if eligible and row["selected"]["score"] > best_score:
                best_score = row["selected"]["score"]
                best_state = {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                }
                best_row = row
            print(
                f"[latent-neighbor] epoch={epoch} "
                f"loss={row['train_loss']:.6f} "
                f"model={row['latent_model']:.6f} "
                f"blend={row['selected']['score']:.6f}"
                f"@{row['selected']['beta']:g}",
                flush=True,
            )

    if best_state is None:
        raise RuntimeError("no eligible checkpoint")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = out.with_suffix(".pt")
    torch.save(
        {
            "model_state": best_state,
            "domain": args.domain,
            "rank": args.rank,
            "k": args.k,
            "distance_power": args.distance_power,
            "heads": args.heads,
            "hidden": args.hidden,
            "residual_bound": args.residual_bound,
            "components": components.detach().cpu(),
            "spectrum_mean": spectrum_mean.detach().cpu(),
            "coefficient_mean": coefficient_mean.detach().cpu(),
            "coefficient_std": coefficient_std.detach().cpu(),
            "singular_values": singular.detach().cpu(),
            "split_name": split_name,
            "selection": best_row,
        },
        checkpoint,
    )
    result = {
        "selection_policy": (
            "validation labels excluded from PCA, training, and "
            "neighbor bank"
        ),
        "split_name": split_name,
        "domain": args.domain,
        "train_size": len(train_idx),
        "validation_size": len(val_idx),
        "rank": args.rank,
        "k": args.k,
        "distance_power": args.distance_power,
        "heads": args.heads,
        "hidden": args.hidden,
        "residual_bound": args.residual_bound,
        "loss": args.loss,
        "loss_slices": args.loss_slices,
        "coefficient_aux_weight": args.coefficient_aux_weight,
        "epochs": args.epochs,
        "select_final_only": args.select_final_only,
        "expert_val": args.expert_val,
        "selected": best_row,
        "history": history,
        "checkpoint": str(checkpoint),
    }
    out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"NEIGHBOR_LATENT_DONE out={out} "
        f"score={best_score:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
