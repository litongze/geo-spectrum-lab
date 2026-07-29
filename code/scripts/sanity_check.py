#!/usr/bin/env python3
"""Check official round files and instantiate a configured model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from wireless_twin.data import load_round, load_point_cloud
from wireless_twin.models import build_model
from wireless_twin.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--datadir", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    datadir = Path(args.datadir or cfg["data"]["datadir"])
    rd = load_round(datadir, cfg["data"].get("scaler_mode", "std"), load_test=True)
    model_cfg = dict(cfg["model"])
    name = model_cfg.pop("name")
    model = build_model(name, rd.spec, **model_cfg)
    map_path = datadir / f"{rd.round_tag}_Map.ply"
    points = load_point_cloud(map_path)
    print(rd.spec)
    print(f"train positions: {rd.train.positions.shape}")
    print(f"train channel: {rd.train.channels.shape}, dtype={rd.train.channels.dtype}")
    print(f"test positions: {None if rd.test_positions is None else rd.test_positions.shape}")
    print(f"map points: {points.shape}, finite={np.isfinite(points).all()}")
    print(f"channel scale: {rd.scaler.scale:.6e}")
    print(f"model: {name}, parameters={sum(p.numel() for p in model.parameters())/1e6:.3f}M")


if __name__ == "__main__":
    main()
