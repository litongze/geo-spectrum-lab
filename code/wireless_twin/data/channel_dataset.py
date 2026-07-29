"""Competition dataset and round loader.

The channel array is kept as a read-only NumPy memory map and each target is
normalised/converted to real-imag form on demand.  This avoids duplicating the
multi-gigabyte Round1 channel tensor in every DDP worker process.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from .normalization import ChannelScaler, complex_to_ri
from .setup_config import ChannelSpec, load_setup


def detect_round(datadir: Union[str, Path]) -> str:
    datadir = Path(datadir)
    for path in sorted(datadir.iterdir()):
        match = re.match(r"(Round\d+)_", path.name)
        if match:
            return match.group(1)
    raise FileNotFoundError(f"No RoundX_* files found in {datadir}")


class ChannelDataset(Dataset):
    def __init__(
        self,
        positions: np.ndarray,
        channels: Optional[np.ndarray],
        scaler: ChannelScaler,
        eager: bool = False,
    ) -> None:
        self.positions = np.asarray(positions, dtype=np.float32)
        self.scaler = scaler
        self.has_channels = channels is not None
        self.channels = channels
        self.targets: Optional[np.ndarray] = None
        if self.has_channels:
            assert channels is not None
            if len(channels) != len(self.positions):
                raise ValueError(
                    f"positions/channels length mismatch: "
                    f"{len(self.positions)} vs {len(channels)}")
            self.channel_shape = tuple(channels.shape[1:])
            if eager:
                ri = complex_to_ri(channels / scaler.scale)
                self.targets = ri.reshape(len(self.positions), -1)
        else:
            self.channel_shape = None

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, idx: int):
        pos = torch.from_numpy(self.positions[idx])
        if not self.has_channels:
            return pos, torch.tensor(idx)
        if self.targets is not None:
            target = self.targets[idx]
        else:
            assert self.channels is not None
            channel = np.asarray(self.channels[idx]) / self.scaler.scale
            target = complex_to_ri(channel).reshape(-1)
        return pos, torch.from_numpy(np.asarray(target, dtype=np.float32))


@dataclass
class RoundData:
    spec: ChannelSpec
    scaler: ChannelScaler
    train: ChannelDataset
    test: Optional[ChannelDataset]
    test_positions: Optional[np.ndarray]
    round_tag: str
    pos_mean: np.ndarray
    pos_std: np.ndarray

    def normalize_positions(self, pos: np.ndarray) -> np.ndarray:
        return (pos - self.pos_mean) / self.pos_std


def load_round(
    datadir: Union[str, Path],
    scaler_mode: str = "std",
    load_test: bool = True,
    mmap_channels: bool = True,
    eager_targets: bool = False,
) -> RoundData:
    datadir = Path(datadir)
    tag = detect_round(datadir)
    spec = load_setup(datadir / f"{tag}_Setup.json")
    train_pos = np.load(datadir / f"{tag}_Train_Pos.npy").astype(np.float32)
    train_ch = np.load(
        datadir / f"{tag}_Train_Channel.npy",
        mmap_mode="r" if mmap_channels else None,
    )
    if train_ch.ndim != 4 or tuple(train_ch.shape[1:]) != spec.channel_shape:
        raise ValueError(
            f"channel shape {train_ch.shape} incompatible with {spec.channel_shape}")
    if spec.p_train and spec.p_train != len(train_pos):
        print(
            f"[data] Setup P_Train={spec.p_train}, actual={len(train_pos)}; "
            "using actual file length")
    spec.p_train = len(train_pos)

    pos_mean = train_pos.mean(axis=0).astype(np.float32)
    raw_pos_std = train_pos.std(axis=0).astype(np.float32)
    # Receiver height is often constant. Dividing map/BS z coordinates by an
    # almost-zero std creates values around 1e8, which overflow FP16 in the
    # first scene-encoder linear layer. Use a physical scale floor instead.
    scale_floor = max(float(raw_pos_std.max()) * 0.05, 1.0)
    pos_std = np.maximum(raw_pos_std, scale_floor).astype(np.float32)
    scaler = ChannelScaler(mode=scaler_mode).fit(train_ch)
    train_ds = ChannelDataset(
        (train_pos - pos_mean) / pos_std,
        train_ch,
        scaler,
        eager=eager_targets,
    )

    test_ds: Optional[ChannelDataset] = None
    test_pos: Optional[np.ndarray] = None
    test_path = datadir / f"{tag}_Test_Pos.npy"
    if load_test and test_path.exists():
        test_pos = np.load(test_path).astype(np.float32)
        spec.p_test = len(test_pos)
        test_ds = ChannelDataset(
            (test_pos - pos_mean) / pos_std, None, scaler)

    return RoundData(
        spec=spec,
        scaler=scaler,
        train=train_ds,
        test=test_ds,
        test_positions=test_pos,
        round_tag=tag,
        pos_mean=pos_mean,
        pos_std=pos_std,
    )
