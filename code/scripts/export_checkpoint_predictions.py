#!/usr/bin/env python3
"""Export checkpoint predictions for test or a fixed validation-index file."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import _bootstrap  # noqa: F401
from wireless_twin.data import load_round
from wireless_twin.evaluation import load_model_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--datadir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--val-indices", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, meta = load_model_from_checkpoint(args.ckpt, args.device)
    precision = str(args.precision or meta.get("train_config", {}).get("precision", "fp32"))
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    rd = load_round(
        args.datadir,
        scaler_mode=meta.get("scaler", {}).get("mode", "std"),
        load_test=True,
        mmap_channels=True,
    )
    if args.val_indices:
        idx = np.load(args.val_indices).astype(np.int64)
        positions = rd.train.positions[idx]
    else:
        if rd.test_positions is None:
            raise ValueError("test positions are unavailable")
        positions = (rd.test_positions.astype(np.float32) - rd.pos_mean) / rd.pos_std

    device = next(model.parameters()).device
    pred = np.empty((len(positions), model.spec.m, model.spec.n, model.spec.s), dtype=np.complex64)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(positions), args.batch_size):
            end = min(start + args.batch_size, len(positions))
            batch = torch.from_numpy(positions[start:end]).to(device)
            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=device.type == "cuda" and precision != "fp32",
            ):
                h = model(batch)
            pred[start:end] = h.cpu().numpy().astype(np.complex64)
    pred *= np.float32(meta["scaler"]["scale"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, pred)
    print(f"[export_checkpoint_predictions] wrote {out}; shape={pred.shape}")


if __name__ == "__main__":
    main()
