#!/usr/bin/env python3
"""Train a PathField or map-conditioned Transformer channel model.

Single GPU:
    python scripts/train.py --config configs/round1_transformer_base.yaml

Four GPUs:
    torchrun --standalone --nproc_per_node=4 scripts/train.py \
        --config configs/round1_transformer_large.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TextIO

import torch
import torch.distributed as dist

import _bootstrap  # noqa: F401
from wireless_twin.data import load_round
from wireless_twin.data.map_loader import load_point_cloud
from wireless_twin.models import available_models, build_model
from wireless_twin.training import TrainConfig, Trainer
from wireless_twin.utils import load_config, merge_overrides, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--datadir", default=None)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument(
        "--init-ckpt",
        default=None,
        help="Optional checkpoint whose model weights initialize this run.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help=(
            "Optional terminal log path. Defaults to the checkpoint path with "
            "a .log suffix, or output.log_file from the config."
        ),
    )
    parser.add_argument(
        "--history-file",
        default=None,
        help=(
            "Optional epoch-history JSON path. Defaults to the checkpoint path "
            "with a .history.json suffix, or output.history_file from the config."
        ),
    )
    parser.add_argument(
        "--set", dest="overrides", nargs="*", default=[],
        help="overrides such as train.epochs=20 model.d_model=128",
    )
    return parser.parse_args()


def init_distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    return rank, local_rank, world_size


class Tee:
    """Mirror terminal output to a log file while preserving console output."""

    def __init__(self, stream: TextIO, log: TextIO) -> None:
        self.stream = stream
        self.log = log

    def write(self, data: str) -> int:
        self.stream.write(data)
        self.log.write(data)
        return len(data)

    def flush(self) -> None:
        self.stream.flush()
        self.log.flush()


def _setup_rank0_logging(path: Path) -> tuple[TextIO, TextIO, TextIO]:
    path.parent.mkdir(parents=True, exist_ok=True)
    log = path.open("w", encoding="utf-8", buffering=1)
    stdout = sys.stdout
    stderr = sys.stderr
    sys.stdout = Tee(stdout, log)  # type: ignore[assignment]
    sys.stderr = Tee(stderr, log)  # type: ignore[assignment]
    print(f"[train] logging terminal output -> {path}")
    return log, stdout, stderr


def _save_history(history: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    if history:
        csv_path = path.with_suffix(".csv")
        keys = sorted({key for row in history for key in row})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(history)
        print(f"[train] saved epoch history -> {path} and {csv_path}")
    else:
        print(f"[train] saved empty epoch history -> {path}")


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size = init_distributed()
    is_main = rank == 0

    cfg = merge_overrides(load_config(args.config), args.overrides)
    data_cfg = cfg.get("data", {})
    model_cfg = dict(cfg.get("model", {}))
    train_cfg = dict(cfg.get("train", {}))
    out_cfg = cfg.get("output", {})

    datadir = Path(args.datadir or data_cfg.get("datadir"))
    ckpt = args.ckpt or out_cfg.get("ckpt", "checkpoints/model.pt")
    ckpt_path = Path(ckpt)
    log_file = Path(
        args.log_file
        or out_cfg.get("log_file")
        or ckpt_path.with_suffix(".log")
    )
    history_file = Path(
        args.history_file
        or out_cfg.get("history_file")
        or ckpt_path.with_suffix(".history.json")
    )
    log_state = _setup_rank0_logging(log_file) if is_main else None
    scaler_mode = data_cfg.get("scaler_mode", "std")
    seed = int(train_cfg.get("seed", 0))
    set_seed(seed + rank)

    if torch.cuda.is_available():
        train_cfg["device"] = f"cuda:{local_rank}"
    if is_main:
        print(f"[train] loading {datadir} with world_size={world_size}")
    rd = load_round(
        datadir,
        scaler_mode=scaler_mode,
        load_test=False,
        mmap_channels=bool(data_cfg.get("mmap_channels", True)),
        eager_targets=bool(data_cfg.get("eager_targets", False)),
    )

    model_name = model_cfg.pop("name", "transformer")
    if model_name not in available_models():
        raise SystemExit(
            f"model '{model_name}' unavailable; have {available_models()}")
    model = build_model(model_name, rd.spec, **model_cfg)
    if args.init_ckpt:
        init_payload = torch.load(
            args.init_ckpt, map_location="cpu", weights_only=False)
        init_name = init_payload.get("meta", {}).get("model_name")
        if init_name is not None and init_name != model_name:
            raise ValueError(
                f"initial checkpoint model={init_name}, requested model={model_name}")
        model.load_state_dict(init_payload["model_state"], strict=True)
        if is_main:
            print(f"[train] initialized model weights from {args.init_ckpt}")

    use_map = bool(data_cfg.get("use_map", True))
    map_path = datadir / f"{rd.round_tag}_Map.ply"
    if use_map and hasattr(model, "set_scene"):
        if not map_path.exists():
            raise FileNotFoundError(f"map requested but not found: {map_path}")
        if is_main:
            print(f"[train] preprocessing map: {map_path}")
        points = load_point_cloud(map_path)
        model.set_scene(  # type: ignore[attr-defined]
            points,
            pos_mean=rd.pos_mean,
            pos_std=rd.pos_std,
            bs_position=rd.spec.bs_position,
        )

    if is_main:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[train] {rd.round_tag}: {rd.spec}")
        print(
            "[train] position normalisation: "
            f"mean={rd.pos_mean.tolist()}, scale={rd.pos_std.tolist()}")
        if hasattr(model, "export_scene_state"):
            scene_state = model.export_scene_state()  # type: ignore[attr-defined]
            sf = scene_state.get("scene_features")
            if sf is not None and sf.numel() > 0:
                print(
                    "[train] scene features: "
                    f"finite={bool(torch.isfinite(sf).all())}, "
                    f"abs_max={float(sf.abs().max()):.3f}")
        print(f"[train] model={model_name}, params={n_params/1e6:.2f}M")

    meta = {
        "model_name": model_name,
        "model_kwargs": model_cfg,
        "spec": asdict(rd.spec),
        "scaler": rd.scaler.state_dict(),
        "pos_mean": rd.pos_mean.tolist(),
        "pos_std": rd.pos_std.tolist(),
        "round_tag": rd.round_tag,
    }
    if hasattr(model, "export_scene_state"):
        meta["scene_state"] = model.export_scene_state()  # type: ignore[attr-defined]

    accepted = TrainConfig.__dataclass_fields__
    tcfg = TrainConfig(**{k: v for k, v in train_cfg.items() if k in accepted})
    meta["train_config"] = asdict(tcfg)
    trainer = Trainer(model, rd.train, tcfg, checkpoint_meta=meta)
    if is_main:
        print(f"[train] device={trainer.device}, precision={tcfg.precision}")
    interrupted = False
    try:
        history = trainer.fit()
    except KeyboardInterrupt:
        interrupted = True
        if is_main:
            print(
                "[train] interrupted by user; saving best checkpoint seen so far",
                flush=True,
            )
        history = []
    finally:
        if is_main and "history" in locals():
            _save_history(history, history_file)
        trainer.save_checkpoint(ckpt)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    if is_main:
        print("[train] interrupted" if interrupted else "[train] done")
    if log_state is not None:
        log_handle, stdout, stderr = log_state
        sys.stdout = stdout
        sys.stderr = stderr
        log_handle.close()


if __name__ == "__main__":
    main()
