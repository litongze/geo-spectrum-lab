"""NeRF2-Channel: a volumetric radio-radiance field for MIMO-OFDM channels.

Unlike ScatterField (a *discrete* set of learned scatterers), this model marches
rays from the BS array into the environment and integrates a *continuous* learned
radiance field.  The received signal is rendered per angular direction, which is
exactly what the PAS metric measures (PAS = |DFT_array(H)|^2), so the angular
structure is produced natively rather than as a by-product of an FFT.

Rendering pipeline for a UE position x:
  * cast one ray per angular bin (d_h, d_v) on the M_H x M_V grid from the BS
  * sample points p_i = BS + t_i * dir along each ray
  * a NeRF-style MLP maps (p_i, dir, x) -> density sigma_i and emission (a_i, phi_i)
  * integrate with transmittance and a geometric delay phase over sub-carriers:
        R[d,s] = sum_i T_i (1-e^{-sigma_i dt}) a_i e^{j phi_i} e^{-j 2pi kappa tau_i s}
    where tau_i = (|x - p_i| + t_i) is the UE->p_i->BS path length.
  * R is the angular spectrum on the array-DFT grid, so H = IDFT2_array(R),
    expanded over polarisation (M_P) and UE ports (N) by small learned factors.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..data.setup_config import ChannelSpec
from .base import ChannelModel
from .encodings import FourierFeatures


class _CVec(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        s = 1.0 / (shape[-1] ** 0.5)
        self.re = nn.Parameter(torch.randn(*shape) * s)
        self.im = nn.Parameter(torch.randn(*shape) * s)

    def forward(self):
        return torch.complex(self.re, self.im)


class NeRF2ChannelModel(ChannelModel):
    def __init__(
        self,
        spec: ChannelSpec,
        n_samples: int = 48,
        near: float = 1.0,
        far: float = 400.0,
        hidden: int = 128,
        depth: int = 6,
        kappa_init: float = 1e-3,
        dir_fov: float = 1.0,     # direction-cosine half-extent of the array grid
        in_dim: int = 3,
        **_ignore,
    ):
        super().__init__(spec)
        self.mh, self.mv, self.mp = spec.mh, spec.mv, spec.mp
        self.n, self.s = spec.n, spec.s
        self.n_samples = n_samples
        self.near, self.far = near, far
        self.register_buffer("bs", torch.tensor(spec.bs_position, dtype=torch.float32))
        self.register_buffer("s_idx", torch.arange(spec.s, dtype=torch.float32))
        self._cscale = float(max(abs(v) for v in spec.bs_position) * 4 + 200)

        # angular grid: one ray direction per (d_h, d_v) array-DFT bin.  Bin k on
        # an M-point DFT of a lambda/2 array maps to direction cosine u = 2*k'/M
        # (k' = fftshift index in [-M/2, M/2)).  Boresight = +x, array plane = y-z.
        kh = ((torch.arange(self.mh) + self.mh // 2) % self.mh) - self.mh // 2
        kv = ((torch.arange(self.mv) + self.mv // 2) % self.mv) - self.mv // 2
        u = (2.0 * dir_fov) * kh.float() / self.mh
        v = (2.0 * dir_fov) * kv.float() / self.mv
        uu, vv = torch.meshgrid(u, v, indexing="ij")          # (MH, MV)
        w2 = (1.0 - uu**2 - vv**2).clamp_min(0.04)
        dirs = torch.stack([w2.sqrt(), uu, vv], dim=-1)        # (MH,MV,3) boresight +x
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)
        self.register_buffer("dirs", dirs.reshape(-1, 3))     # (D,3), D=MH*MV

        # encoders
        self.enc_p = FourierFeatures(3, n_freqs=10, include_input=True)
        self.enc_d = FourierFeatures(3, n_freqs=4, include_input=True)
        self.enc_x = FourierFeatures(3, n_freqs=10, include_input=True)

        # density (attenuation) MLP: point -> sigma + feature
        din = self.enc_p.out_dim
        layers, d = [], din
        self.skip = depth // 2
        self.sigma_layers = nn.ModuleList()
        for i in range(depth):
            inp = din if i == 0 else (hidden + din if i == self.skip else hidden)
            self.sigma_layers.append(nn.Linear(inp, hidden))
        self.sigma_head = nn.Linear(hidden, 1)
        self.feat = nn.Linear(hidden, hidden)
        # emission MLP: (feature, dir, x) -> amp, phase
        self.sig_layers = nn.Sequential(
            nn.Linear(hidden + self.enc_d.out_dim + self.enc_x.out_dim, hidden),
            nn.ReLU(inplace=True), nn.Linear(hidden, 2))

        self.log_kappa = nn.Parameter(torch.tensor(float(np.log(kappa_init))))
        self.log_gain = nn.Parameter(torch.zeros(()))
        # per-polarisation and per-UE-port complex signatures (small learned)
        self.pol = _CVec(self.mp)
        self.ue = _CVec(self.n)

    def _field(self, pts, dirs, x):
        """pts:(P,3) dirs:(P,3) x:(P,3) -> sigma:(P,), amp:(P,), phase:(P,)."""
        h = self.enc_p(pts / self._cscale)
        e = h
        for i, lin in enumerate(self.sigma_layers):
            if i == self.skip:
                h = torch.cat([h, e], -1)
            h = torch.relu(lin(h))
        sigma = torch.nn.functional.softplus(self.sigma_head(h).squeeze(-1))
        feat = self.feat(h)
        se = torch.cat([feat, self.enc_d(dirs), self.enc_x(x / self._cscale)], -1)
        out = self.sig_layers(se)
        amp = torch.nn.functional.softplus(out[..., 0])
        phase = out[..., 1]
        return sigma, amp, phase

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        positions = positions[:, :3]
        B = positions.shape[0]
        D = self.dirs.shape[0]
        dev = positions.device
        # sample points along each ray: p = BS + t*dir
        t = torch.linspace(0.0, 1.0, self.n_samples, device=dev) * (self.far - self.near) + self.near
        dt = (self.far - self.near) / self.n_samples
        # broadcast: (B, D, T, 3)
        dirs = self.dirs.to(dev)                                    # (D,3)
        pts = self.bs.view(1, 1, 1, 3) + t.view(1, 1, -1, 1) * dirs.view(1, D, 1, 3)
        pts = pts.expand(B, D, self.n_samples, 3)
        xrep = positions.view(B, 1, 1, 3).expand(B, D, self.n_samples, 3)
        drep = dirs.view(1, D, 1, 3).expand(B, D, self.n_samples, 3)
        flat = self.n_samples * D * B
        sigma, amp, phase = self._field(
            pts.reshape(flat, 3), drep.reshape(flat, 3), xrep.reshape(flat, 3))
        sigma = sigma.reshape(B, D, self.n_samples)
        amp = amp.reshape(B, D, self.n_samples)
        phase = phase.reshape(B, D, self.n_samples)
        # transmittance T_i = prod_{j<i}(1 - alpha_j); alpha = 1-exp(-sigma dt)
        alpha = 1.0 - torch.exp(-sigma * dt)
        T = torch.cumprod(
            torch.cat([torch.ones(B, D, 1, device=dev), 1.0 - alpha + 1e-10], -1), -1)[..., :-1]
        weight = T * alpha                                          # (B,D,T)
        # geometric delay: tau = |x - p| + t  (UE->p->BS path)
        d_ue = torch.linalg.norm(xrep - pts, dim=-1)                # (B,D,T)
        tau = d_ue + t.view(1, 1, -1)
        kappa = torch.exp(self.log_kappa)
        # angular-subcarrier spectrum R[b,d,s]
        base = weight * amp * torch.exp(1j * phase.to(torch.complex64))   # (B,D,T) complex
        ph_s = (-2 * np.pi * kappa) * (tau.unsqueeze(-1) * self.s_idx.view(1, 1, 1, -1))  # (B,D,T,S)
        dphase = torch.complex(torch.cos(ph_s), torch.sin(ph_s))
        R = torch.einsum("bdt,bdts->bds", base.to(torch.complex64), dphase)   # (B,D,S)
        R = R.reshape(B, self.mh, self.mv, self.s) * torch.exp(self.log_gain)
        # H over antennas = inverse 2D-DFT of the angular spectrum
        H = torch.fft.ifft2(R, dim=(1, 2), norm="ortho")           # (B,MH,MV,S)
        # expand polarisation and UE ports
        H = H[:, :, :, None, None, :] * self.pol().view(1, 1, 1, self.mp, 1, 1) \
              * self.ue().view(1, 1, 1, 1, self.n, 1)              # (B,MH,MV,MP,N,S)
        return H.reshape(B, self.mh * self.mv * self.mp, self.n, self.s)
