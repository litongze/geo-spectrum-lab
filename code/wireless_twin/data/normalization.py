"""Memory-efficient complex-channel normalisation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np


@dataclass
class ChannelScaler:
    mode: str = "std"
    scale: float = 1.0

    def fit(self, channels: np.ndarray, chunk_size: int = 32) -> "ChannelScaler":
        """Fit from an ndarray or memory map without materialising ``abs(H)``."""
        if self.mode == "max":
            value = 0.0
            for i in range(0, len(channels), chunk_size):
                value = max(value, float(np.abs(channels[i:i + chunk_size]).max()))
            self.scale = value
        elif self.mode == "std":
            power_sum = 0.0
            count = 0
            for i in range(0, len(channels), chunk_size):
                chunk = channels[i:i + chunk_size]
                # |z|^2 computed without creating a second complex array.
                power_sum += float(
                    np.square(chunk.real, dtype=np.float64).sum()
                    + np.square(chunk.imag, dtype=np.float64).sum()
                )
                count += int(chunk.size)
            self.scale = float(np.sqrt(power_sum / max(count, 1)))
        else:
            raise ValueError(f"unknown scaler mode: {self.mode}")
        self.scale = max(float(self.scale), 1e-12)
        return self

    def transform(self, channels: np.ndarray) -> np.ndarray:
        return channels / self.scale

    def inverse_transform(self, channels: np.ndarray) -> np.ndarray:
        return channels * self.scale

    def state_dict(self) -> dict:
        return {"mode": self.mode, "scale": self.scale}

    def load_state_dict(self, state: dict) -> "ChannelScaler":
        self.mode = state["mode"]
        self.scale = float(state["scale"])
        return self


def complex_to_ri(channels: np.ndarray) -> np.ndarray:
    return np.stack([channels.real, channels.imag], axis=-1).astype(np.float32)


def ri_to_complex(ri: Union[np.ndarray]) -> np.ndarray:
    return (ri[..., 0] + 1j * ri[..., 1]).astype(np.complex64)
