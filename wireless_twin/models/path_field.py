"""PathField: a parameter-efficient Physical-AI channel baseline.

Motivation
----------
A MIMO-OFDM channel is the superposition of a handful of propagation *paths*.
Each path has

* a complex gain that depends on the receiver **position** ``x``, and
* a deterministic signature over the BS array (steering), the UE array and the
  sub-carriers (delay phase-ramp).

We mirror that structure with a low-rank (CANDECOMP/PARAFAC) factorisation of
``K`` learnable modes::

    H(x)[m, n, s] = sum_k  c_k(x) * U[k, m] * V[k, n] * W[k, s]

where ``c_k(x)`` are complex per-position gains produced by a Fourier-feature
MLP, and ``U, V, W`` are learnable complex signatures shared across positions.

This keeps the parameter count tiny (``K*(M+N+S)`` complex entries instead of
``M*N*S`` per output) yet is expressive, differentiable, and — crucially — runs
in pure PyTorch on CPU/CUDA so the Windows teammate can train it without
compiling any CUDA extension.  It is a strong, honest starting point that the
3D-GS backend can later replace behind the same :class:`ChannelModel` interface.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..data.setup_config import ChannelSpec
from .base import ChannelModel
from .encodings import FourierFeatures


class _ComplexFactor(nn.Module):
    """A learnable complex matrix ``(K, D)`` stored as two real parameters."""

    def __init__(self, k: int, d: int) -> None:
        super().__init__()
        std = 1.0 / (d ** 0.5)
        self.real = nn.Parameter(torch.randn(k, d) * std)
        self.imag = nn.Parameter(torch.randn(k, d) * std)

    def forward(self) -> torch.Tensor:
        return torch.complex(self.real, self.imag)


class PathFieldModel(ChannelModel):
    """Fourier-MLP + CP-factorised complex channel generator."""

    def __init__(
        self,
        spec: ChannelSpec,
        n_paths: int = 256,
        n_freqs: int = 10,
        hidden_dim: int = 256,
        n_layers: int = 4,
    ) -> None:
        super().__init__(spec)
        self.k = n_paths
        m, n, s = spec.m, spec.n, spec.s

        # --- position -> complex path gains c_k(x) -----------------------
        self.encoder = FourierFeatures(3, n_freqs=n_freqs, include_input=True)
        layers = [nn.Linear(self.encoder.out_dim, hidden_dim), nn.ReLU(inplace=True)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        self.trunk = nn.Sequential(*layers)
        # 2*K outputs -> real & imag of the K path gains
        self.gain_head = nn.Linear(hidden_dim, 2 * self.k)

        # --- shared complex signatures over the three physical axes ------
        self.u = _ComplexFactor(self.k, m)   # BS-array signature
        self.v = _ComplexFactor(self.k, n)   # UE-array signature
        self.w = _ComplexFactor(self.k, s)   # sub-carrier / delay signature

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        feats = self.trunk(self.encoder(positions))          # (B, H)
        gains = self.gain_head(feats)                        # (B, 2K)
        gr, gi = gains[:, : self.k], gains[:, self.k:]
        c = torch.complex(gr, gi)                            # (B, K) complex

        u, v, w = self.u(), self.v(), self.w()               # complex factors
        # (B,K),(K,M),(K,N),(K,S) -> (B,M,N,S)
        h = torch.einsum("bk,km,kn,ks->bmns", c, u, v, w)
        return h
