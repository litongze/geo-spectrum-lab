#!/usr/bin/env python3
"""Train a complex-neighbor gate on label-free test-geometry proxies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from _bootstrap import ROOT  # noqa: F401
from complex_neighbor_gate import (
    FEATURE_NAMES,
    ComplexNeighborGate,
    baseline_coefficients,
)
from train_complex_neighbor_gate import (
    GateData,
    build_data,
    evaluate,
    fit_model,
    load_data,
    optimal_calibration,
    save_data,
    select_features,
    sufficient,
    tensors,
)
from score_holdout import reproduce_val_indices
from wireless_twin.data.setup_config import load_setup


def coefficients(
    data: GateData,
    checkpoint: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    current = tensors(data, device)
    feature_mean = torch.as_tensor(
        checkpoint["feature_mean"],
        dtype=torch.float32,
        device=device,
    )
    feature_std = torch.as_tensor(
        checkpoint["feature_std"],
        dtype=torch.float32,
        device=device,
    )
    normalized = (
        (current["features"] - feature_mean) / feature_std
    ).clamp(-8.0, 8.0)
    model = ComplexNeighborGate(
        len(checkpoint["feature_names"]),
        checkpoint.get("architecture", "mlp16"),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.inference_mode():
        learned, _ = model.coefficients(
            normalized,
            current["base_weight"],
            current["amplitude"],
        )
        baseline = baseline_coefficients(
            current["base_weight"], current["amplitude"]
        )
    return baseline, learned, current


def evaluate_blends(
    training: GateData,
    validation: GateData,
    checkpoint: dict,
    alphas: list[float],
    device: torch.device,
) -> dict[str, dict]:
    train_baseline, train_learned, train_tensors = coefficients(
        training, checkpoint, device
    )
    val_baseline, val_learned, val_tensors = coefficients(
        validation, checkpoint, device
    )
    rows = {}
    for alpha in alphas:
        train_coefficient = (
            (1.0 - alpha) * train_baseline
            + alpha * train_learned
        )
        train_cross, train_energy = sufficient(
            train_coefficient, train_tensors
        )
        phase, scale = optimal_calibration(
            train_cross, train_energy
        )
        validation_coefficient = (
            (1.0 - alpha) * val_baseline
            + alpha * val_learned
        )
        rows[f"{alpha:g}"] = {
            "training": evaluate(
                train_coefficient,
                train_tensors,
                phase,
                scale,
            ),
            "validation": evaluate(
                validation_coefficient,
                val_tensors,
                phase,
                scale,
            ),
        }
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", default="Round1_Map(2)")
    parser.add_argument(
        "--config", default="configs/gs43_dual_mix10.json"
    )
    parser.add_argument(
        "--panel-dir", default="cache/test_geometry_panel_dw1"
    )
    parser.add_argument("--panel-count", type=int, default=3)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--channel-batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--regularization", type=float, default=0.01)
    parser.add_argument(
        "--alpha-grid",
        default="0,0.2,0.4,0.5,0.6,0.7,0.8,1",
    )
    parser.add_argument(
        "--diagnostic-seeds",
        default="",
        help=(
            "optional clean random splits; each model trains only on "
            "geometry proxies outside its complete validation fold"
        ),
    )
    parser.add_argument(
        "--feature-set",
        choices=("full", "invariant"),
        default="invariant",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--cache-dir",
        default="cache/geometry_matched_complex_gate",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints/geometry_matched_complex_gate",
    )
    parser.add_argument(
        "--out",
        default="docs/geometry_matched_complex_gate/result.json",
    )
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if config["source"].get("moment") is not None:
        raise ValueError(
            "signed moment weights are unsupported by this gate"
        )
    datadir = Path(args.datadir)
    spec = load_setup(datadir / "Round1_Setup.json")
    positions = np.load(
        datadir / "Round1_Train_Pos.npy"
    ).astype(np.float64)
    channels = np.load(
        datadir / "Round1_Train_Channel.npy", mmap_mode="r"
    )
    all_idx = np.arange(len(positions), dtype=np.int64)
    panel_dir = Path(args.panel_dir)
    proxy_all = np.asarray(
        sorted(np.load(panel_dir / "proxy_all.npy").tolist()),
        dtype=np.int64,
    )
    panels = [
        np.asarray(
            sorted(
                np.load(
                    panel_dir / f"proxy_p{number}.npy"
                ).tolist()
            ),
            dtype=np.int64,
        )
        for number in range(args.panel_count)
    ]
    if set(np.concatenate(panels).tolist()) != set(
        proxy_all.tolist()
    ):
        raise ValueError("panels must partition proxy_all")
    if sum(len(panel) for panel in panels) != len(proxy_all):
        raise ValueError("geometry panels overlap")

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
    alphas = [
        float(value)
        for value in args.alpha_grid.split(",")
        if value
    ]
    if any(not 0.0 <= alpha <= 1.0 for alpha in alphas):
        raise ValueError("alpha grid values must be in [0, 1]")
    device = torch.device(args.device)
    cache_dir = Path(args.cache_dir)
    checkpoint_dir = Path(args.checkpoint_dir)

    def cached(
        name: str,
        query_idx: np.ndarray,
        pool_idx: np.ndarray,
        leave_self_out: bool,
    ) -> GateData:
        path = cache_dir / f"{name}.npz"
        if path.exists() and not args.rebuild_cache:
            return load_data(path)
        data = build_data(
            name,
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

    fold_rows = {}
    for fold, validation_idx in enumerate(panels):
        pool_idx = np.setdiff1d(all_idx, validation_idx)
        training_idx = np.setdiff1d(
            proxy_all, validation_idx
        )
        training = select_features(
            cached(
                f"p{fold}_train",
                training_idx,
                pool_idx,
                True,
            ),
            feature_names,
        )
        validation = select_features(
            cached(
                f"p{fold}_val",
                validation_idx,
                pool_idx,
                False,
            ),
            feature_names,
        )
        checkpoint, endpoint = fit_model(
            training,
            validation,
            device,
            args.epochs,
            args.learning_rate,
            args.regularization,
            7301 + fold,
            feature_names,
        )
        checkpoint["metadata"] = {
            "split": f"geometry_proxy_p{fold}",
            "k": args.k,
            "training_queries": int(len(training_idx)),
            "validation_queries": int(len(validation_idx)),
            "selection_policy": (
                "validation proxy labels and channels are excluded "
                "from the complete neighbor pool"
            ),
        }
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            checkpoint, checkpoint_dir / f"fold_p{fold}.pt"
        )
        blends = evaluate_blends(
            training, validation, checkpoint, alphas, device
        )
        fold_rows[f"p{fold}"] = {
            "training_count": int(len(training_idx)),
            "validation_count": int(len(validation_idx)),
            "endpoint": endpoint,
            "blends": blends,
        }
        print(
            f"[geometry-gate] fold=p{fold} "
            f"baseline={blends['0']['validation']['NMSE']:.6f} "
            f"learned={blends['1']['validation']['NMSE']:.6f}",
            flush=True,
        )
        torch.cuda.empty_cache()

    ranking = []
    for alpha in alphas:
        key = f"{alpha:g}"
        deltas = []
        validation_nmse = []
        for fold in range(args.panel_count):
            blends = fold_rows[f"p{fold}"]["blends"]
            baseline = blends["0"]["validation"]["NMSE"]
            current = blends[key]["validation"]["NMSE"]
            deltas.append(current - baseline)
            validation_nmse.append(current)
        ranking.append(
            {
                "alpha": alpha,
                "delta_nmse_mean": float(np.mean(deltas)),
                "delta_nmse_median": float(np.median(deltas)),
                "delta_nmse_worst": float(np.max(deltas)),
                "fold_delta_nmse": deltas,
                "fold_nmse": validation_nmse,
            }
        )
    robust = [
        row
        for row in ranking
        if row["delta_nmse_worst"] <= 0.0
    ]
    selected = min(
        robust or ranking,
        key=lambda row: (
            row["delta_nmse_mean"],
            row["delta_nmse_worst"],
        ),
    )

    diagnostic_rows = {}
    diagnostic_seeds = [
        int(value)
        for value in args.diagnostic_seeds.split(",")
        if value
    ]
    for offset, seed in enumerate(diagnostic_seeds):
        validation_idx = np.asarray(
            sorted(
                reproduce_val_indices(
                    len(positions), 0.1, seed
                )
            ),
            dtype=np.int64,
        )
        pool_idx = np.setdiff1d(all_idx, validation_idx)
        training_idx = np.intersect1d(proxy_all, pool_idx)
        training = select_features(
            cached(
                f"s{seed}_train",
                training_idx,
                pool_idx,
                True,
            ),
            feature_names,
        )
        validation = select_features(
            cached(
                f"s{seed}_val",
                validation_idx,
                pool_idx,
                False,
            ),
            feature_names,
        )
        checkpoint, endpoint = fit_model(
            training,
            validation,
            device,
            args.epochs,
            args.learning_rate,
            args.regularization,
            9301 + offset,
            feature_names,
        )
        checkpoint["metadata"] = {
            "split": str(seed),
            "k": args.k,
            "training_queries": int(len(training_idx)),
            "selected_alpha": selected["alpha"],
            "selection_policy": (
                "the complete random validation fold is excluded "
                "from labels and the channel neighbor pool"
            ),
        }
        torch.save(
            checkpoint, checkpoint_dir / f"s{seed}.pt"
        )
        blends = evaluate_blends(
            training,
            validation,
            checkpoint,
            [0.0, selected["alpha"], 1.0],
            device,
        )
        diagnostic_rows[str(seed)] = {
            "training_count": int(len(training_idx)),
            "validation_count": int(len(validation_idx)),
            "endpoint": endpoint,
            "blends": blends,
        }
        selected_key = f"{selected['alpha']:g}"
        print(
            f"[geometry-gate-diagnostic] split={seed} "
            f"baseline={blends['0']['validation']['NMSE']:.6f} "
            f"selected="
            f"{blends[selected_key]['validation']['NMSE']:.6f}",
            flush=True,
        )
        torch.cuda.empty_cache()

    final_training = select_features(
        cached("final_train", proxy_all, all_idx, True),
        feature_names,
    )
    final_checkpoint, final_row = fit_model(
        final_training,
        None,
        device,
        args.epochs,
        args.learning_rate,
        args.regularization,
        8301,
        feature_names,
    )
    final_checkpoint["metadata"] = {
        "split": "all_geometry_proxies",
        "k": args.k,
        "training_queries": int(len(proxy_all)),
        "selected_alpha": selected["alpha"],
        "selection_policy": (
            "architecture and alpha frozen by three clean proxy folds; "
            "final gate uses all label-free-selected geometry proxies"
        ),
    }
    final_path = checkpoint_dir / "selected.pt"
    torch.save(final_checkpoint, final_path)

    payload = {
        "selection_policy": (
            "three proxy folds selected from coordinates and neighbor "
            "density only; each validation fold is excluded from labels "
            "and the channel neighbor pool"
        ),
        "config": args.config,
        "panel_dir": args.panel_dir,
        "proxy_count": int(len(proxy_all)),
        "feature_set": args.feature_set,
        "feature_names": list(feature_names),
        "epochs": args.epochs,
        "regularization": args.regularization,
        "folds": fold_rows,
        "ranking": sorted(
            ranking,
            key=lambda row: (
                row["delta_nmse_mean"],
                row["delta_nmse_worst"],
            ),
        ),
        "selected": selected,
        "diagnostic_splits": diagnostic_rows,
        "final_training": final_row,
        "final_checkpoint": str(final_path),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(
        "GEOMETRY_MATCHED_COMPLEX_GATE_DONE "
        f"alpha={selected['alpha']:g} "
        f"mean_delta={selected['delta_nmse_mean']:+.6f} "
        f"worst_delta={selected['delta_nmse_worst']:+.6f} "
        f"checkpoint={final_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
