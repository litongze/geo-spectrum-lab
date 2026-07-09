"""Positional encodings.

Coordinate networks (NeRF-style) cannot fit high-frequency spatial variation
from raw ``(x, y, z)`` inputs.  A Fourier feature mapping lifts the 3-D position
into a higher-dimensional space of sines/cosines so a plain MLP can represent
the sharp, multipath-driven variation of the wireless channel.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    """Deterministic axis-aligned Fourier features (NeRF positional encoding).

    For each input coordinate and each of ``n_freqs`` octaves we emit
    ``sin`` and ``cos``.  Output dim = ``in_dim * (1 + 2 * n_freqs)`` when
    ``include_input`` is True.
    """

    def __init__(self, in_dim: int = 3, n_freqs: int = 10,
                 include_input: bool = True) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.n_freqs = n_freqs
        self.include_input = include_input
        freqs = 2.0 ** torch.arange(n_freqs) * math.pi
        self.register_buffer("freqs", freqs, persistent=False)

    @property
    def out_dim(self) -> int:
        base = self.in_dim * 2 * self.n_freqs
        return base + (self.in_dim if self.include_input else 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim) -> (B, in_dim, n_freqs)
        scaled = x.unsqueeze(-1) * self.freqs
        enc = torch.cat([scaled.sin(), scaled.cos()], dim=-1)  # (B,in_dim,2F)
        enc = enc.reshape(x.shape[0], -1)
        if self.include_input:
            enc = torch.cat([x, enc], dim=-1)
        return enc
