"""NeRF2-style neural radiance field adapted to output a MIMO-OFDM channel.

Faithful to the NeRF2 method (MobiCom'23): an **attenuation** MLP over scene
points and a **signal/radiance** MLP conditioned on the transmitter position,
combined by **volume rendering along rays** (transmittance-weighted integral).
The reference NeRF2 renders a scalar/served signal per ray direction; here the
same machinery is adapted to produce the competition's ``(M, N, S)`` complex
channel:

* rays are cast from the BS over an ``MH x MV`` angular grid;
* along each ray we volume-render a complex contribution, and the sample
  **distance** provides the delay -> sub-carrier phase ramp (giving frequency /
  PDP structure that the original NeRF2 does not model);
* the resulting angular response is mapped to the BS antenna array by an inverse
  2-D DFT (the inverse of the PAS transform), then extended over polarisation
  and UE antennas.

The attenuation field is UE-independent (an environment property), so it is
evaluated once per forward pass; only the radiance field is conditioned on the
UE position.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.setup_config import ChannelSpec
from .base import ChannelModel
from .encodings import FourierFeatures


class NeRF2Field(ChannelModel):
    def __init__(
        self,
        spec: ChannelSpec,
        n_samples: int = 48,
        near: float = 1.0,
        far: float = 400.0,
        hidden_dim: int = 128,
        n_freqs: int = 10,
        az_deg: float = 60.0,
        el_deg: tuple = (-30.0, 10.0),
        kappa_init: float = 1e-4,
        in_dim: int = 3,
    ) -> None:
        super().__init__(spec)
        m, n, s = spec.m, spec.n, spec.s
        self.mh, self.mv, self.mp = spec.mh, spec.mv, spec.mp
        self.a = self.mh * self.mv          # angular bins == BS grid
        self.t = n_samples

        # --- fixed ray geometry (from BS over an MH x MV angular fan) -----
        bs = torch.tensor(spec.bs_position, dtype=torch.float32)
        az = torch.linspace(-az_deg, az_deg, self.mh) * np.pi / 180.0
        el = torch.linspace(el_deg[0], el_deg[1], self.mv) * np.pi / 180.0
        AZ, EL = torch.meshgrid(az, el, indexing="ij")             # (MH,MV)
        dirs = torch.stack([torch.cos(EL) * torch.cos(AZ),
                            torch.cos(EL) * torch.sin(AZ),
                            torch.sin(EL)], dim=-1)                 # (MH,MV,3)
        dirs = dirs.reshape(self.a, 3)                             # (A,3)
        t_vals = torch.linspace(near, far, n_samples)             # (T,)
        pts = bs[None, None, :] + dirs[:, None, :] * t_vals[None, :, None]  # (A,T,3)
        self.register_buffer("pts", pts)
        self.register_buffer("t_vals", t_vals)
        self.coord_scale = float(far)   # normalise metres -> ~[-1,1] for Fourier
        self.register_buffer("s_idx", torch.arange(s, dtype=torch.float32))

        # --- attenuation MLP (scene point -> sigma) ----------------------
        self.enc_pts = FourierFeatures(3, n_freqs=n_freqs, include_input=True)
        self.att_mlp = nn.Sequential(
            nn.Linear(self.enc_pts.out_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1), nn.Softplus())

        # --- radiance MLP (scene point + UE position -> complex emission) -
        self.enc_ue = FourierFeatures(3, n_freqs=n_freqs, include_input=True)
        sig_in = self.enc_pts.out_dim + self.enc_ue.out_dim
        self.sig_mlp = nn.Sequential(
            nn.Linear(sig_in, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2))              # (amp-real, phase) -> complex

        # UE-antenna signatures and polarisation weights (learned, shared)
        std = 1.0 / (n ** 0.5)
        self.v_real = nn.Parameter(torch.randn(self.a, n) * std)
        self.v_imag = nn.Parameter(torch.randn(self.a, n) * std)
        self.pol_real = nn.Parameter(torch.ones(self.mp))
        self.pol_imag = nn.Parameter(torch.zeros(self.mp))
        self.log_kappa = nn.Parameter(torch.tensor(float(np.log(kappa_init))))

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        b = positions.shape[0]
        pts = self.pts                                    # (A,T,3)
        enc_pts = self.enc_pts(pts.reshape(-1, 3) / self.coord_scale)  # (A*T, F)

        # attenuation field (UE-independent) -> volume-render weights
        sigma = self.att_mlp(enc_pts).reshape(self.a, self.t)   # (A,T)
        dists = torch.diff(self.t_vals, append=self.t_vals[-1:] + 1e3)  # (T,)
        alpha = 1.0 - torch.exp(-sigma * dists[None, :])         # (A,T)
        trans = torch.cumprod(
            torch.cat([torch.ones(self.a, 1, device=alpha.device),
                       1.0 - alpha + 1e-10], dim=1), dim=1)[:, :-1]
        w = alpha * trans                                        # (A,T)

        # radiance field (UE-conditioned)
        enc_ue = self.enc_ue(positions / self.coord_scale)      # (B,F)
        ep = enc_pts.reshape(self.a * self.t, -1)[None].expand(b, -1, -1)
        eu = enc_ue[:, None, :].expand(-1, self.a * self.t, -1)
        sig = self.sig_mlp(torch.cat([ep, eu], dim=-1))         # (B,A*T,2)
        sig = sig.reshape(b, self.a, self.t, 2)
        amp = F.softplus(sig[..., 0])
        phase = sig[..., 1]
        e = torch.complex(amp * torch.cos(phase), amp * torch.sin(phase))  # (B,A,T)

        # delay phase ramp from sample distance
        kappa = torch.exp(self.log_kappa)
        ramp_ph = (-2 * np.pi * kappa) * (self.t_vals[None, :, None]
                                          * self.s_idx[None, None, :])  # (1,T,S)
        ramp = torch.complex(torch.cos(ramp_ph), torch.sin(ramp_ph))     # (1,T,S)

        we = w[None, :, :] * e                                   # (B,A,T)
        r = torch.einsum("bat,ts->bas", we, ramp[0])            # (B,A,S)

        # angle -> BS antenna via inverse 2-D DFT; add UE + polarisation
        v = torch.complex(self.v_real, self.v_imag)             # (A,N)
        ran = r[:, :, None, :] * v[None, :, :, None]            # (B,A,N,S)
        ran = ran.reshape(b, self.mh, self.mv, self.spec.n, self.spec.s)
        hbs = torch.fft.ifft2(ran, dim=(1, 2))                  # (B,MH,MV,N,S)
        pol = torch.complex(self.pol_real, self.pol_imag)       # (MP,)
        h = hbs[:, :, :, None] * pol[None, None, None, :, None, None]
        # (B,MH,MV,MP,N,S) -> (B, M, N, S)
        return h.reshape(b, self.spec.m, self.spec.n, self.spec.s)
