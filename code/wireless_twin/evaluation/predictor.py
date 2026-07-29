"""Checkpoint reconstruction and submission-file prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

from ..data.normalization import ChannelScaler
from ..data.setup_config import ChannelSpec
from ..models.base import ChannelModel
from ..models.registry import build_model


def load_model_from_checkpoint(
    path: Union[str, Path],
    device: Optional[str] = None,
) -> Tuple[ChannelModel, Dict]:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    meta = payload["meta"]
    spec = ChannelSpec(**meta["spec"])
    model = build_model(meta["model_name"], spec, **meta.get("model_kwargs", {}))
    if hasattr(model, "load_scene_state"):
        model.load_scene_state(meta.get("scene_state"))  # type: ignore[attr-defined]
    model.load_state_dict(payload["model_state"])
    model.to(dev).eval()
    return model, meta


def predict_test_channels(
    model: ChannelModel,
    positions_raw: np.ndarray,
    meta: Dict,
    device: Optional[str] = None,
    batch_size: int = 8,
    precision: Optional[str] = None,
) -> np.ndarray:
    dev = next(model.parameters()).device if device is None else torch.device(device)
    pos_mean = np.asarray(meta["pos_mean"], dtype=np.float32)
    pos_std = np.asarray(meta["pos_std"], dtype=np.float32)
    scaler = ChannelScaler().load_state_dict(meta["scaler"])
    pos_norm = (positions_raw.astype(np.float32) - pos_mean) / pos_std

    channels = np.empty(
        (len(pos_norm), model.spec.m, model.spec.n, model.spec.s),
        dtype=np.complex64,
    )
    model.eval()
    use_amp = dev.type == "cuda"
    precision = precision or meta.get("train_config", {}).get("precision", "fp32")
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError(f"unknown inference precision: {precision}")
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    with torch.inference_mode():
        for i in range(0, len(pos_norm), batch_size):
            j = min(i + batch_size, len(pos_norm))
            batch = torch.from_numpy(pos_norm[i:j]).to(dev)
            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=use_amp and precision != "fp32",
            ):
                h = model(batch)
            channels[i:j] = h.cpu().numpy().astype(np.complex64)
    channels *= np.float32(scaler.scale)
    return channels


def save_test_channels(channels: np.ndarray, out_path: Union[str, Path]) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), channels.astype(np.complex64))
    return out_path
