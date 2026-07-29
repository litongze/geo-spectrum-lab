#!/usr/bin/env python3
"""Diagnose why a checkpoint's whole-set NMSE is close to the zero baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import _bootstrap  # noqa: F401
from wireless_twin.data import load_round
from wireless_twin.evaluation import load_model_from_checkpoint
from wireless_twin.signal import cosine_similarity_along_last, pas_spectrum, pdp_spectrum
from wireless_twin.training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--datadir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--split-mode", choices=("spatial", "random"), default=None)
    parser.add_argument("--distance-space", choices=("normalized", "raw"), default="normalized")
    parser.add_argument("--nearest-neighbor", action="store_true")
    args = parser.parse_args()

    model, meta = load_model_from_checkpoint(args.ckpt, args.device)
    train_cfg = meta.get("train_config", {})
    rd = load_round(
        Path(args.datadir),
        scaler_mode=meta.get("scaler", {}).get("mode", "std"),
        load_test=False,
        mmap_channels=True,
    )
    n_val = int(len(rd.train) * float(train_cfg.get("val_fraction", 0.1)))
    split_mode = str(args.split_mode or train_cfg.get("val_split_mode", "spatial"))
    seed = int(args.seed if args.seed is not None else train_cfg.get("seed", 0))
    if split_mode == "spatial":
        train_idx, val_idx = Trainer._spatial_split_indices(
            rd.train.positions,
            n_val,
            int(train_cfg.get("val_regions", 4)),
            seed,
        )
    else:
        order = torch.randperm(len(rd.train), generator=torch.Generator().manual_seed(seed))
        train_idx = order[:-n_val].numpy()
        val_idx = order[-n_val:].numpy()

    device = next(model.parameters()).device
    precision = str(args.precision or train_cfg.get("precision", "fp32"))
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    scale = float(meta["scaler"]["scale"])
    pred_power = 0.0
    gt_power = 0.0
    error_power = 0.0
    cross = 0.0j
    pas_sum = 0.0
    pdp_sum = 0.0

    model.eval()
    batch_size = max(1, args.batch_size)
    with torch.inference_mode():
        for start in range(0, len(val_idx), batch_size):
            indices = val_idx[start:start + batch_size]
            pos = torch.from_numpy(rd.train.positions[indices]).to(device)
            with torch.autocast(
                device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"
            ):
                pred = model(pos)
            pred_np = pred.cpu().numpy().astype(np.complex64, copy=False)
            gt_np = np.asarray(rd.train.channels[indices]) / scale
            gt = torch.from_numpy(np.ascontiguousarray(gt_np)).to(device)
            pas_sum += float(cosine_similarity_along_last(
                pas_spectrum(pred, model.spec), pas_spectrum(gt, model.spec)
            )) * len(indices)
            pdp_sum += float(cosine_similarity_along_last(
                pdp_spectrum(pred, model.spec), pdp_spectrum(gt, model.spec)
            )) * len(indices)
            pred_power += float(np.vdot(pred_np, pred_np).real)
            gt_power += float(np.vdot(gt_np, gt_np).real)
            diff = pred_np - gt_np
            error_power += float(np.vdot(diff, diff).real)
            cross += complex(np.vdot(pred_np, gt_np))

    nmse = error_power / gt_power
    power_ratio = pred_power / gt_power
    cross_ratio = cross.real / gt_power
    coherence_sq = abs(cross) ** 2 / max(pred_power * gt_power, 1e-300)
    optimal_nmse = 1.0 - coherence_sq
    pas = pas_sum / len(val_idx)
    pdp = pdp_sum / len(val_idx)
    w1, w2, w3 = model.spec.metric_weights
    score = w1 * pas + w2 * pdp + w3 / (1.0 + nmse)

    print(f"checkpoint={args.ckpt}")
    print(f"validation_samples={len(val_idx)}, best_epoch={meta.get('best_epoch', 'n/a')}")
    print("zero_prediction_nmse=1.000000")
    print(f"checkpoint_nmse={nmse:.9f}")
    print(f"checkpoint_pas={pas:.9f}")
    print(f"checkpoint_pdp={pdp:.9f}")
    print(f"checkpoint_score={score:.9f}")
    print(f"prediction_to_target_power={power_ratio:.9e}")
    print(f"real_cross_to_target_power={cross_ratio:.9e}")
    print(f"identity_check={1.0 + power_ratio - 2.0 * cross_ratio:.9f}")
    print(f"complex_coherence_squared={coherence_sq:.9e}")
    print(f"nmse_after_optimal_complex_scalar={optimal_nmse:.9f}")

    if args.nearest_neighbor:
        all_pos = rd.train.positions
        if args.distance_space == "raw":
            all_pos = all_pos * rd.pos_std + rd.pos_mean
        train_pos = all_pos[train_idx]
        val_pos = all_pos[val_idx]
        nn_error = 0.0
        nn_gt_power = 0.0
        nn_pred_power = 0.0
        nn_cross_real = 0.0
        nn_pas_sum = 0.0
        nn_pdp_sum = 0.0
        distances = []
        for pos, index in zip(val_pos, val_idx):
            dist2 = np.sum((train_pos - pos) ** 2, axis=1)
            nearest_local = int(np.argmin(dist2))
            nearest_index = int(train_idx[nearest_local])
            pred_np = np.asarray(rd.train.channels[nearest_index]) / scale
            gt_np = np.asarray(rd.train.channels[index]) / scale
            diff = pred_np - gt_np
            nn_error += float(np.vdot(diff, diff).real)
            nn_gt_power += float(np.vdot(gt_np, gt_np).real)
            nn_pred_power += float(np.vdot(pred_np, pred_np).real)
            nn_cross_real += float(np.vdot(pred_np, gt_np).real)
            pred_t = torch.from_numpy(np.ascontiguousarray(pred_np[None])).to(device)
            gt_t = torch.from_numpy(np.ascontiguousarray(gt_np[None])).to(device)
            nn_pas_sum += float(cosine_similarity_along_last(
                pas_spectrum(pred_t, model.spec), pas_spectrum(gt_t, model.spec)
            ))
            nn_pdp_sum += float(cosine_similarity_along_last(
                pdp_spectrum(pred_t, model.spec), pdp_spectrum(gt_t, model.spec)
            ))
            distances.append(float(np.sqrt(dist2[nearest_local])))
        nn_nmse = nn_error / nn_gt_power
        nn_pas = nn_pas_sum / len(val_idx)
        nn_pdp = nn_pdp_sum / len(val_idx)
        nn_score = w1 * nn_pas + w2 * nn_pdp + w3 / (1.0 + nn_nmse)
        print(f"nearest_neighbor_nmse={nn_nmse:.9f}")
        print(f"nearest_neighbor_pas={nn_pas:.9f}")
        print(f"nearest_neighbor_pdp={nn_pdp:.9f}")
        print(f"nearest_neighbor_score={nn_score:.9f}")
        nn_power_ratio = nn_pred_power / nn_gt_power
        nn_cross_ratio = nn_cross_real / nn_gt_power
        print(f"nearest_neighbor_power_ratio={nn_power_ratio:.9f}")
        print(f"nearest_neighbor_cross_ratio={nn_cross_ratio:.9f}")
        for alpha in (0.02, 0.05, 0.1):
            scaled_nmse = 1.0 + alpha * alpha * nn_power_ratio - 2.0 * alpha * nn_cross_ratio
            scaled_score = w1 * nn_pas + w2 * nn_pdp + w3 / (1.0 + scaled_nmse)
            print(f"nearest_scaled_{alpha:g}_nmse={scaled_nmse:.9f}, score={scaled_score:.9f}")
        print(f"nearest_distance_median_normalized={np.median(distances):.6f}")


if __name__ == "__main__":
    main()
