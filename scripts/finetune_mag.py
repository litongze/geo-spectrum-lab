#!/usr/bin/env python3
"""Two-stage: load a PAS/PDP-trained checkpoint (good shape) and fine-tune with a
magnitude loss so it also predicts the true magnitude, with minimal shape loss.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import _bootstrap  # noqa: F401

from wireless_twin.data import load_round
from wireless_twin.data.map_loader import load_point_cloud
from wireless_twin.models import build_model
from wireless_twin.training import TrainConfig, Trainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="base checkpoint (good shape)")
    ap.add_argument("--datadir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-magabs", type=float, default=1.0)
    ap.add_argument("--lambda-magdb", type=float, default=0.0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    payload = torch.load(args.init, map_location="cpu", weights_only=False)
    meta = payload["meta"]
    from wireless_twin.data.setup_config import ChannelSpec
    spec = ChannelSpec(**meta["spec"])
    rd = load_round(args.datadir, scaler_mode="std", load_test=False,
                    use_bs_geometry=meta.get("use_bs_geometry", False),
                    use_map_features=meta.get("use_map_features", False))
    model = build_model(meta["model_name"], spec, **meta["model_kwargs"])
    # scatterers buffer must be restored (part of state_dict)
    model.load_state_dict(payload["model_state"])
    print(f"[finetune] loaded {args.init}; fine-tuning with "
          f"magabs={args.lambda_magabs} magdb={args.lambda_magdb} for {args.epochs} ep")

    cfg = TrainConfig(epochs=args.epochs, batch_size=64, lr=args.lr,
                      lambda_pas=1.0, lambda_pdp=1.0, lambda_mag=0.0,
                      lambda_magabs=args.lambda_magabs, lambda_magdb=args.lambda_magdb,
                      log_every=20, device=args.device, seed=0)
    trainer = Trainer(model, rd.train, cfg, checkpoint_meta=meta)
    trainer.fit()
    trainer.save_checkpoint(args.out)


if __name__ == "__main__":
    main()
