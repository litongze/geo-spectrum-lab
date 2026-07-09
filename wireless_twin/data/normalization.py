"""Channel normalisation.

Raw MIMO-OFDM channels span many orders of magnitude.  Networks train far more
reliably on standardised targets, so we fit a single real-valued scale on the
training set and apply it symmetrically to the real and imaginary parts.  The
transform is fully invertible so predictions can be mapped back to physical
units before being written to ``RoundX_Test_Channel.npy``.

Two modes:
    * ``"max"``  -- divide by the global max magnitude (keeps phase, bounds to 1)
    * ``"std"``  -- divide by the RMS magnitude (unit average power)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np


@dataclass
class ChannelScaler:
    """Invertible real scale applied to complex channels."""

    mode: str = "std"
    scale: float = 1.0

    def fit(self, channels: np.ndarray) -> "ChannelScaler":
        """Estimate the scale from a complex training array ``(P, M, N, S)``."""
        mag = np.abs(channels)
        if self.mode == "max":
            self.scale = float(mag.max())
        elif self.mode == "std":
            self.scale = float(np.sqrt(np.mean(mag ** 2)))
        else:
            raise ValueError(f"unknown scaler mode: {self.mode}")
        self.scale = max(self.scale, 1e-12)
        return self

    def transform(self, channels: np.ndarray) -> np.ndarray:
        return channels / self.scale

    def inverse_transform(self, channels: np.ndarray) -> np.ndarray:
        return channels * self.scale

    # -- persistence ---------------------------------------------------
    def state_dict(self) -> dict:
        return {"mode": self.mode, "scale": self.scale}

    def load_state_dict(self, state: dict) -> "ChannelScaler":
        self.mode = state["mode"]
        self.scale = float(state["scale"])
        return self


def complex_to_ri(channels: np.ndarray) -> np.ndarray:
    """Stack a complex array's real/imag parts on a trailing axis: ``(..., 2)``."""
    return np.stack([channels.real, channels.imag], axis=-1).astype(np.float32)


def ri_to_complex(ri: Union[np.ndarray]) -> np.ndarray:
    """Inverse of :func:`complex_to_ri`; last axis must be size 2."""
    return (ri[..., 0] + 1j * ri[..., 1]).astype(np.complex64)
