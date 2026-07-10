#!/usr/bin/env python3
"""Train a Physical-AI channel model for one competition round.

Usage
-----
    python scripts/train.py --config configs/round1.yaml
    python scripts/train.py --config configs/round1.yaml --set train.epochs=300
    python scripts/train.py --config configs/round1.yaml --datadir data/Data1
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import _bootstrap  # noqa: F401  (adds repo root to sys.path)

from wireless_twin.data import load_round
from wireless_twin.models import available_models, build_model
from wireless_twin.training import TrainConfig, Trainer
from wireless_twin.utils import load_config, merge_overrides, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="path to a YAML/JSON config")
    p.add_argument("--datadir", default=None, help="override data.datadir")
    p.add_argument("--ckpt", default=None, help="override output.ckpt")
    p.add_argument("--set", dest="overrides", nargs="*", default=[],
                   help="config overrides, e.g. train.epochs=300 model.n_paths=128")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = merge_overrides(load_config(args.config), args.overrides)

    data_cfg = cfg.get("data", {})
    model_cfg = dict(cfg.get("model", {}))
    train_cfg = cfg.get("train", {})
    out_cfg = cfg.get("output", {})

    datadir = args.datadir or data_cfg.get("datadir")
    ckpt = args.ckpt or out_cfg.get("ckpt", "checkpoints/model.pt")
    scaler_mode = data_cfg.get("scaler_mode", "std")
    use_geo = bool(data_cfg.get("use_bs_geometry", False))
    use_map = bool(data_cfg.get("use_map_features", False))

    set_seed(int(train_cfg.get("seed", 0)))

    print(f"[train] loading round from {datadir} "
          f"(bs_geometry={use_geo}, map_features={use_map})")
    rd = load_round(datadir, scaler_mode=scaler_mode, load_test=False,
                    use_bs_geometry=use_geo, use_map_features=use_map)
    print(f"[train] {rd.round_tag}: {rd.spec}")
    print(f"[train] train positions: {len(rd.train)} | input dim: {rd.in_dim}")

    model_name = model_cfg.pop("name", "path_field")
    if model_name not in available_models():
        raise SystemExit(
            f"model '{model_name}' not available. Have: {available_models()}")
    model_cfg["in_dim"] = rd.in_dim
    model = build_model(model_name, rd.spec, **model_cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model '{model_name}' with {n_params/1e6:.2f}M parameters")

    tcfg = TrainConfig(**{k: v for k, v in train_cfg.items()
                          if k in TrainConfig.__dataclass_fields__})

    meta = {
        "model_name": model_name,
        "model_kwargs": model_cfg,
        "spec": asdict(rd.spec),
        "scaler": rd.scaler.state_dict(),
        "pos_mean": rd.pos_mean.tolist(),
        "pos_std": rd.pos_std.tolist(),
        "round_tag": rd.round_tag,
        "use_bs_geometry": use_geo,
        "use_map_features": use_map,
        "map_file": str(Path(datadir) / f"{rd.round_tag}_Map.ply") if use_map else None,
    }

    trainer = Trainer(model, rd.train, tcfg, checkpoint_meta=meta)
    print(f"[train] device: {trainer.device}")
    trainer.fit()
    trainer.save_checkpoint(ckpt)
    print("[train] done.")


if __name__ == "__main__":
    main()
