"""Run a trained model over test positions and write the submission file.

The output ``RoundX_Test_Channel.npy`` must be a complex array of shape
``(P_test, M, N, S)`` (task book §2.3.2.2).  This module rebuilds a model from a
self-contained checkpoint, predicts in the *normalised* space it was trained in,
then maps channels back to physical units with the fitted scaler.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

from ..data.augment import augment_positions
from ..data.map_features import compute_map_features
from ..data.map_loader import load_point_cloud
from ..data.normalization import ChannelScaler
from ..data.setup_config import ChannelSpec
from ..models.base import ChannelModel
from ..models.registry import build_model


def load_model_from_checkpoint(
    path: Union[str, Path],
    device: Optional[str] = None,
) -> Tuple[ChannelModel, Dict]:
    """Rebuild the model + return its training metadata from a checkpoint."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    payload = torch.load(str(path), map_location=dev)
    meta = payload["meta"]

    spec = ChannelSpec(**meta["spec"])
    model = build_model(meta["model_name"], spec, **meta.get("model_kwargs", {}))
    model.load_state_dict(payload["model_state"])
    model.to(dev).eval()
    return model, meta


def predict_test_channels(
    model: ChannelModel,
    positions_raw: np.ndarray,
    meta: Dict,
    device: Optional[str] = None,
    batch_size: int = 64,
) -> np.ndarray:
    """Predict physical-unit complex channels for raw ``(P, 3)`` positions."""
    dev = next(model.parameters()).device if device is None else torch.device(device)
    pos_mean = np.asarray(meta["pos_mean"], dtype=np.float32)
    pos_std = np.asarray(meta["pos_std"], dtype=np.float32)
    scaler = ChannelScaler().load_state_dict(meta["scaler"])

    bs = meta["spec"]["bs_position"]
    feats = positions_raw.astype(np.float32)
    if meta.get("use_bs_geometry"):
        feats = augment_positions(feats, bs)
    if meta.get("use_map_features"):
        points = load_point_cloud(meta["map_file"])
        feats = np.concatenate(
            [feats, compute_map_features(positions_raw, bs, points)], axis=1)
    pos_norm = (feats - pos_mean) / pos_std

    outputs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(pos_norm), batch_size):
            batch = torch.from_numpy(pos_norm[i:i + batch_size]).to(dev)
            h = model(batch).cpu().numpy().astype(np.complex64)   # normalised
            outputs.append(h)
    channels = np.concatenate(outputs, axis=0)
    return scaler.inverse_transform(channels).astype(np.complex64)


def save_test_channels(channels: np.ndarray, out_path: Union[str, Path]) -> Path:
    """Write ``RoundX_Test_Channel.npy`` (complex64, ``(P, M, N, S)``)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), channels.astype(np.complex64))
    return out_path
