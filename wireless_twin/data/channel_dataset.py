"""Competition dataset + round loader.

``ChannelDataset`` is a plain ``torch.utils.data.Dataset`` yielding
``(position, channel_ri)`` pairs, where ``channel_ri`` is the normalised
real/imag representation of one position's ``(M, N, S)`` complex channel,
flattened to a 1-D vector so it is model-agnostic.

``load_round`` is the high-level helper that wires a whole round together:
it finds the ``RoundX_*`` files in a directory, parses the Setup, fits the
scaler on the training channels and returns everything the trainer needs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import AUGMENTED_DIM, augment_positions
from .normalization import ChannelScaler, complex_to_ri
from .setup_config import ChannelSpec, load_setup


def detect_round(datadir: Union[str, Path]) -> str:
    """Return the round tag (e.g. ``"Round1"``) present in ``datadir``."""
    datadir = Path(datadir)
    for p in sorted(datadir.iterdir()):
        m = re.match(r"(Round\d+)_", p.name)
        if m:
            return m.group(1)
    raise FileNotFoundError(
        f"No 'RoundX_*' files found in {datadir}. "
        "Expected e.g. Round1_Setup.json / Round1_Train_Pos.npy.")


class ChannelDataset(Dataset):
    """Positions + (optional) channels for one split of one round.

    Parameters
    ----------
    positions : (P, 3) float array of sample coordinates.
    channels  : optional (P, M, N, S) complex array. ``None`` for the test
                split, where only positions are known.
    scaler    : fitted :class:`ChannelScaler` used to normalise channels.
    """

    def __init__(
        self,
        positions: np.ndarray,
        channels: Optional[np.ndarray],
        scaler: ChannelScaler,
    ) -> None:
        self.positions = np.asarray(positions, dtype=np.float32)
        self.scaler = scaler
        self.has_channels = channels is not None

        if self.has_channels:
            self.channel_shape = channels.shape[1:]        # (M, N, S)
            normed = scaler.transform(channels)            # complex
            # (P, M, N, S) complex -> (P, M*N*S*2) float32
            ri = complex_to_ri(normed)                     # (P, M, N, S, 2)
            self.targets = ri.reshape(len(self.positions), -1)
        else:
            self.channel_shape = None
            self.targets = None

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, idx: int):
        pos = torch.from_numpy(self.positions[idx])
        if self.has_channels:
            tgt = torch.from_numpy(self.targets[idx])
            return pos, tgt
        return pos, torch.tensor(idx)


@dataclass
class RoundData:
    """Bundle returned by :func:`load_round`."""

    spec: ChannelSpec
    scaler: ChannelScaler
    train: ChannelDataset
    test: Optional[ChannelDataset]
    test_positions: Optional[np.ndarray]
    round_tag: str
    pos_mean: np.ndarray
    pos_std: np.ndarray
    use_bs_geometry: bool = False
    in_dim: int = 3

    def normalize_positions(self, pos: np.ndarray) -> np.ndarray:
        """Standardise coordinates the same way the model was trained."""
        return (pos - self.pos_mean) / self.pos_std


def load_round(
    datadir: Union[str, Path],
    scaler_mode: str = "std",
    load_test: bool = True,
    use_bs_geometry: bool = False,
) -> RoundData:
    """Load a full competition round from ``datadir``.

    The directory is expected to contain the ``RoundX_*`` files described in
    the task book.  Position coordinates are standardised (zero mean / unit
    std over the training set) because raw metres are a poor network input.
    """
    datadir = Path(datadir)
    tag = detect_round(datadir)

    spec = load_setup(datadir / f"{tag}_Setup.json")

    train_pos = np.load(datadir / f"{tag}_Train_Pos.npy").astype(np.float32)
    train_ch = np.load(datadir / f"{tag}_Train_Channel.npy")

    # Optionally augment raw coordinates with BS-relative geometry.
    train_feat = (augment_positions(train_pos, spec.bs_position)
                  if use_bs_geometry else train_pos)
    in_dim = AUGMENTED_DIM if use_bs_geometry else 3

    pos_mean = train_feat.mean(axis=0)
    pos_std = train_feat.std(axis=0) + 1e-8

    scaler = ChannelScaler(mode=scaler_mode).fit(train_ch)

    train_ds = ChannelDataset(
        (train_feat - pos_mean) / pos_std, train_ch, scaler)

    test_ds: Optional[ChannelDataset] = None
    test_pos: Optional[np.ndarray] = None
    test_path = datadir / f"{tag}_Test_Pos.npy"
    if load_test and test_path.exists():
        test_pos = np.load(test_path).astype(np.float32)
        test_feat = (augment_positions(test_pos, spec.bs_position)
                     if use_bs_geometry else test_pos)
        test_ds = ChannelDataset((test_feat - pos_mean) / pos_std, None, scaler)

    return RoundData(
        spec=spec,
        scaler=scaler,
        train=train_ds,
        test=test_ds,
        test_positions=test_pos,
        round_tag=tag,
        pos_mean=pos_mean,
        pos_std=pos_std,
        use_bs_geometry=use_bs_geometry,
        in_dim=in_dim,
    )
