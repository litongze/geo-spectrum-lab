"""ScatterField: a physics-structured, differentiable scatterer model.

The position-only baseline hits a wall because its position -> channel mapping
is *memorised* by an MLP and cannot extrapolate to unseen locations.  This model
instead makes the position-dependence **geometric**, so it extrapolates by
construction:

    H(x)[m, n, s] = sum_k  a_k * amp_k(x) * U_k[m] * V_k[n]
                          * exp(-i * 2*pi * kappa * tau_k(x) * s)

* ``p_k`` — K candidate scatterers sampled from the environment point cloud.
* ``tau_k(x) = |x - p_k| + |p_k - BS|`` — geometric path length via scatterer k,
  which sets the delay phase-ramp over sub-carriers (drives the PDP).
* ``amp_k(x) = 1 / (|x-p_k| * |p_k-BS|)`` — geometric activation, so which
  scatterers dominate (and thus the BS angular pattern / PAS) varies with x.
* Learned & shared across all positions: complex reflectivity ``a_k``, the BS
  and UE array signatures ``U_k / V_k``, and the frequency constant ``kappa``.

The only thing that changes with position is the geometry, so a location the
model never saw still gets physically-consistent delays and scatterer weights.
This is "Physical AI" (learned reflectivities, gradient-trained) — not ray
tracing (no physical-optics computation).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..data.setup_config import ChannelSpec
from .base import ChannelModel


class _ComplexVec(nn.Module):
    def __init__(self, *shape) -> None:
        super().__init__()
        std = 1.0 / (shape[-1] ** 0.5)
        self.real = nn.Parameter(torch.randn(*shape) * std)
        self.imag = nn.Parameter(torch.randn(*shape) * std)

    def forward(self) -> torch.Tensor:
        return torch.complex(self.real, self.imag)


class ScatterFieldModel(ChannelModel):
    def __init__(
        self,
        spec: ChannelSpec,
        n_scatterers: int = 1024,
        scatterers: np.ndarray | None = None,
        kappa_init: float = 1e-3,
        amp_power: float = 1.0,
        n_freq_basis: int = 0,   # >0 adds a learned per-scatterer freq envelope
        in_dim: int = 3,   # accepted for API symmetry; raw xyz expected
    ) -> None:
        super().__init__(spec)
        self.k = n_scatterers
        self.amp_power = amp_power
        self.n_freq_basis = n_freq_basis
        m, n, s = spec.m, spec.n, spec.s

        # scatterer positions (buffer -> saved/restored with the model)
        if scatterers is None:
            scatterers = np.zeros((n_scatterers, 3), dtype=np.float32)
        self.register_buffer("scatterers", torch.tensor(
            np.asarray(scatterers, dtype=np.float32)))
        self.register_buffer("bs", torch.tensor(
            np.asarray(spec.bs_position, dtype=np.float32)))
        # BS<->scatterer distance is position-independent -> precompute
        self.register_buffer("d_bs", torch.linalg.norm(
            self.scatterers - self.bs, dim=1).clamp_min(1e-3))

        # subcarrier index vector
        self.register_buffer("s_idx", torch.arange(s, dtype=torch.float32))

        # learned, environment-shared parameters
        self.a = _ComplexVec(self.k)          # complex reflectivity per scatterer
        self.u = _ComplexVec(self.k, m)       # BS array signature
        self.v = _ComplexVec(self.k, n)       # UE array signature
        self.log_kappa = nn.Parameter(torch.tensor(float(np.log(kappa_init))))
        self.log_gain = nn.Parameter(torch.zeros(()))

        # optional learned per-scatterer frequency envelope (low-rank, shared)
        if n_freq_basis > 0:
            self.wc = _ComplexVec(self.k, n_freq_basis)   # coeffs (K, R)
            self.wb = _ComplexVec(n_freq_basis, s)        # basis  (R, S)

    def set_scatterers(self, pts: np.ndarray) -> None:
        """Assign scatterer positions and recompute the BS distances (buffers)."""
        t = torch.tensor(np.asarray(pts, dtype=np.float32),
                         device=self.scatterers.device)
        assert t.shape == self.scatterers.shape, \
            f"expected {tuple(self.scatterers.shape)}, got {tuple(t.shape)}"
        self.scatterers.copy_(t)
        self.d_bs.copy_(torch.linalg.norm(
            self.scatterers - self.bs, dim=1).clamp_min(1e-3))

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        # positions: (B, 3) raw metres
        b = positions.shape[0]
        # geometry
        d_ue = torch.cdist(positions, self.scatterers).clamp_min(1e-3)  # (B,K)
        tau = d_ue + self.d_bs[None, :]                                 # (B,K)
        amp = torch.exp(self.log_gain) / (d_ue * self.d_bs[None, :]) ** self.amp_power

        a = self.a()                                                   # (K,)
        g = amp.to(a.dtype) * a[None, :]                               # (B,K) complex
        # delay phase ramp: exp(-i 2pi kappa tau s)
        kappa = torch.exp(self.log_kappa)
        phase = (-2 * np.pi * kappa) * (tau[:, :, None] * self.s_idx[None, None, :])
        d = torch.complex(torch.cos(phase), torch.sin(phase))          # (B,K,S)
        gd = g[:, :, None] * d                                         # (B,K,S)
        if self.n_freq_basis > 0:
            w = self.wc() @ self.wb()                                  # (K,S)
            gd = gd * w[None, :, :]

        uv = self.u()[:, :, None] * self.v()[:, None, :]               # (K,M,N)
        # H(b,m,n,s) = sum_k uv(k,m,n) * gd(b,k,s)  -- no (B,K,M,N,S) intermediate
        h = torch.einsum("kmn,bks->bmns", uv, gd)
        return h
