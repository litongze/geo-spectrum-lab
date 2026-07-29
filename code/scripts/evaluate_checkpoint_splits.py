#!/usr/bin/env python3
"""Evaluate one checkpoint on all v0.5 validation splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import _bootstrap  # noqa: F401
from wireless_twin.data import load_round
from wireless_twin.evaluation import load_model_from_checkpoint
from wireless_twin.evaluation.metric_variants import (
    robust_channel_metrics,
    robust_metrics_dict,
)
from wireless_twin.evaluation.validation_splits import build_validation_splits
from wireless_twin.training.trainer import Trainer


def _predict_indices(model, positions: np.ndarray, indices: np.ndarray, batch_size: int,
                     precision: str) -> np.ndarray:
    device = next(model.parameters()).device
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    out = np.empty(
        (len(indices), model.spec.m, model.spec.n, model.spec.s),
        dtype=np.complex64,
    )
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            local = indices[start:start + batch_size]
            pos = torch.from_numpy(positions[local]).to(device)
            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=device.type == "cuda" and precision != "fp32",
            ):
                pred = model(pos)
            out[start:start + len(local)] = pred.cpu().numpy().astype(np.complex64)
    return out


def _write_markdown(path: Path, scores: dict[str, dict[str, float]],
                    offline_score: float, worst_spatial: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "c_current", "c_robust", "c1_robust", "c2_robust", "nmse",
        "pas_2d_sum_pol", "pas_2d_sep_pol", "pas_1d_flat_m",
        "pdp_per_mn", "pdp_sum_m", "pdp_sum_mn",
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("# Checkpoint Split Audit\n\n")
        f.write(
            "> Warning: unless a split matches the checkpoint's original "
            "holdout, these scores may include samples seen during training. "
            "Use them for metric-definition auditing, not final model selection.\n\n"
        )
        f.write(f"- offline_score: `{offline_score:.9f}`\n")
        f.write(f"- worst_spatial_c_robust: `{worst_spatial:.9f}`\n\n")
        f.write("| split | " + " | ".join(keys) + " |\n")
        f.write("|---|" + "|".join(["---:"] * len(keys)) + "|\n")
        for name, row in scores.items():
            f.write("| " + name + " | " + " | ".join(
                f"{row.get(key, 0.0):.6f}" for key in keys) + " |\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--datadir", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default=None)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default="reports/P1_P2_checkpoint_splits.md")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    model, meta = load_model_from_checkpoint(args.ckpt, args.device)
    train_cfg = meta.get("train_config", {})
    checkpoint_split_seed = train_cfg.get("split_seed")
    if checkpoint_split_seed is None:
        checkpoint_split_seed = train_cfg.get("seed", 0)
    seed = int(args.seed if args.seed is not None else checkpoint_split_seed)
    precision = str(args.precision or train_cfg.get("precision", "fp32"))
    rd = load_round(
        args.datadir,
        scaler_mode=meta.get("scaler", {}).get("mode", "std"),
        load_test=True,
        mmap_channels=True,
    )
    raw_train_pos = rd.train.positions * rd.pos_std + rd.pos_mean
    test_pos = None if rd.test_positions is None else rd.test_positions
    splits = build_validation_splits(
        raw_train_pos,
        rd.train.channels,
        test_pos,
        val_fraction=args.val_fraction,
        seed=seed,
    )
    val_indices_file = train_cfg.get("val_indices_file")
    if val_indices_file:
        from wireless_twin.evaluation.validation_splits import Split

        original_val = np.load(val_indices_file).astype(np.int64)
        all_idx = np.arange(len(rd.train))
        splits = {
            "checkpoint_original_holdout": Split(
                "checkpoint_original_holdout",
                np.setdiff1d(all_idx, original_val, assume_unique=False),
                np.sort(np.unique(original_val)).astype(np.int64),
                0.0,
            ),
            **splits,
        }
    n_val = int(len(rd.train) * float(train_cfg.get("val_fraction", args.val_fraction)))
    split_mode = str(train_cfg.get("val_split_mode", "spatial"))
    if not val_indices_file and n_val > 0 and split_mode == "spatial":
        _, original_val = Trainer._spatial_split_indices(
            rd.train.positions,
            n_val,
            int(train_cfg.get("val_regions", 4)),
            seed,
        )
        from wireless_twin.evaluation.validation_splits import Split

        all_idx = np.arange(len(rd.train))
        splits = {
            "checkpoint_original_holdout": Split(
                "checkpoint_original_holdout",
                np.setdiff1d(all_idx, original_val, assume_unique=False),
                original_val.astype(np.int64),
                0.0,
            ),
            **splits,
        }

    scores: dict[str, dict[str, float]] = {}
    weighted = 0.0
    weight_sum = 0.0
    spatial_scores = []
    for name, split in splits.items():
        pred = _predict_indices(
            model, rd.train.positions, split.val_idx, args.batch_size, precision)
        gt = np.asarray(rd.train.channels[split.val_idx]) / float(meta["scaler"]["scale"])
        result = robust_channel_metrics(pred, gt, rd.spec, chunk=args.chunk)
        row = robust_metrics_dict(result)
        scores[name] = row
        if split.weight > 0:
            weighted += split.weight * row["c_robust"]
            weight_sum += split.weight
        if name.startswith("spatial_block_"):
            spatial_scores.append(row["c_robust"])
        print(f"{name}: c_robust={row['c_robust']:.9f}, nmse={row['nmse']:.9f}")

    offline_score = weighted / max(weight_sum, 1e-12)
    worst_spatial = min(spatial_scores) if spatial_scores else float("nan")
    out = Path(args.out)
    _write_markdown(out, scores, offline_score, worst_spatial)
    print(f"[checkpoint_splits] offline_score={offline_score:.9f}")
    print(f"[checkpoint_splits] wrote {out}")
    if args.json_out:
        payload = {
            "offline_score": offline_score,
            "worst_spatial_c_robust": worst_spatial,
            "splits": scores,
        }
        json_out = Path(args.json_out)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        with json_out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[checkpoint_splits] wrote {json_out}")


if __name__ == "__main__":
    main()
