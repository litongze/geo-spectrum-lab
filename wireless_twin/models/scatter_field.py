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
from .encodings import FourierFeatures


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
        scatter_dropout: float = 0.0,  # train-time fraction of scatterers dropped
        n_vis_rank: int = 0,     # >0 adds a low-rank UE-dependent visibility gate
        learn_scatterers: bool = False,  # gradient-refine scatterer positions
        structured_u: bool = False,  # geometric BS steering dictionary for U_k
        n_mag_paths: int = 0,    # >0: independent magnitude head (normalized-dB)
        mag_freqs: int = 8,
        n_attn_heads: int = 0,   # >0: multi-head cross-attention visibility gate
        attn_dim: int = 32,
        u_rank: int = 0,         # >0: low-rank BS signature U=C@B (shared basis)
        structured_u2: bool = False,  # fitted-constant steering U (150deg/1.0lambda) + learnable v-pattern
        u2_rank: int = 0,        # >0 with structured_u2: low-rank free modulation on top of steering
        in_dim: int = 3,   # accepted for API symmetry; raw xyz expected
    ) -> None:
        super().__init__(spec)
        self.k = n_scatterers
        self.amp_power = amp_power
        self.n_freq_basis = n_freq_basis
        self.scatter_dropout = scatter_dropout
        self.n_vis_rank = n_vis_rank
        self.learn_scatterers = learn_scatterers
        self.structured_u = structured_u
        self.n_mag_paths = n_mag_paths
        self.n_attn_heads = n_attn_heads
        self.attn_dim = attn_dim
        self.u_rank = u_rank
        self.structured_u2 = structured_u2
        self.u2_rank = u2_rank
        # extra input dims beyond xyz are ray-traced map-context features
        # (LoS-to-BS blockage + directional openness) -> condition occlusion
        self.map_dim = max(0, int(in_dim) - 3)
        self._cscale = float(max(abs(v) for v in spec.bs_position) * 4 + 200)
        m, n, s = spec.m, spec.n, spec.s
        self.mh, self.mv, self.mp = spec.mh, spec.mv, spec.mp

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
        if learn_scatterers:   # learnable offset from the point-cloud init
            self.scat_offset = nn.Parameter(torch.zeros(n_scatterers, 3))

        # subcarrier index vector
        self.register_buffer("s_idx", torch.arange(s, dtype=torch.float32))

        # learned, environment-shared parameters
        self.a = _ComplexVec(self.k)          # complex reflectivity per scatterer
        self.v = _ComplexVec(self.k, n)       # UE array signature
        self.log_kappa = nn.Parameter(torch.tensor(float(np.log(kappa_init))))
        self.log_gain = nn.Parameter(torch.zeros(()))

        # BS array signature U_k: either freely learned (K x M) or a structured
        # geometric steering dictionary toward each scatterer (grafts the
        # model-based / physics-informed INR idea: a steering-vector dictionary
        # activated by learned coefficients -> strong regulariser).
        if structured_u:
            self.register_buffer("mh_idx", torch.arange(self.mh, dtype=torch.float32))
            self.register_buffer("mv_idx", torch.arange(self.mv, dtype=torch.float32))
            self.log_kappa_bs = nn.Parameter(torch.zeros(()))   # electrical size
            self.pol = _ComplexVec(self.mp)                     # polarisation gains
        elif structured_u2:
            # fitted array: boresight 150deg, horizontal 1.0*lambda, sign -1 (93% LoS-bin hit).
            # steering is GEOMETRIC (from scatterer az/el); only shared params learned.
            self.psi = nn.Parameter(torch.tensor(150.0 * np.pi / 180))
            self.sh = nn.Parameter(torch.tensor(-2.0 * np.pi))        # 1.0 lambda, sign -1
            self.EB = 32
            self.vpat = nn.Parameter(torch.randn(self.EB, self.mv, 2) * 0.3)
            self.pol2 = _ComplexVec(self.k, self.mp)                   # per-scatterer pol pair
            self.register_buffer("mh_i2", torch.arange(self.mh, dtype=torch.float32))
            if u2_rank > 0:   # steering-aligned low-rank deviation (init ~identity)
                self.u2c = _ComplexVec(self.k, u2_rank)
                self.u2b = _ComplexVec(u2_rank, spec.m)
            self.lobe = nn.Parameter(torch.full((self.k, 2), 3.0))  # softplus->taper宽度(元素数)
        elif u_rank > 0:
            self.u_c = _ComplexVec(self.k, u_rank)   # (K,r) per-scatterer coeffs
            self.u_b = _ComplexVec(u_rank, m)         # (r,M) shared angular basis
        else:
            self.u = _ComplexVec(self.k, m)

        # optional low-rank UE-dependent visibility gate (grafts NeRF2's
        # occlusion modelling: vis(x,k) = sigmoid(f(x) . g_k) in [0,1])
        if n_vis_rank > 0:
            self.enc_vis = FourierFeatures(3, n_freqs=6, include_input=True)
            # feed the ray-traced map-context features alongside the position
            # encoding so the gate learns geometry-driven occlusion, not just a
            # smooth function of (x,y,z) -> should generalise to unseen positions.
            self.vis_mlp = nn.Sequential(
                nn.Linear(self.enc_vis.out_dim + self.map_dim, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, n_vis_rank))
            self.vis_g = nn.Parameter(torch.randn(self.k, n_vis_rank) * 0.1)
            self.coord_scale = float(max(np.abs(spec.bs_position)) * 4 + 200)

        # optional multi-head cross-attention visibility gate: the UE position is
        # the QUERY, each scatterer's GEOMETRY (position) is a KEY.  The learned
        # embedding makes "which scatterers this position sees" a smooth function
        # of geometry -> generalises to unseen positions better than a free
        # low-rank gate.  Sigmoid (not softmax) so many paths can stay active
        # (superposition), and per-scatterer bias + per-head weight combine heads.
        if n_attn_heads > 0:
            self.enc_q = FourierFeatures(3, n_freqs=6, include_input=True)
            self.enc_k = FourierFeatures(3, n_freqs=6, include_input=True)
            h_out = n_attn_heads * attn_dim
            self.q_proj = nn.Sequential(
                nn.Linear(self.enc_q.out_dim, 128), nn.ReLU(inplace=True),
                nn.Dropout(0.1), nn.Linear(128, h_out))
            self.k_proj = nn.Sequential(
                nn.Linear(self.enc_k.out_dim, 128), nn.ReLU(inplace=True),
                nn.Linear(128, h_out))
            self.head_w = nn.Parameter(torch.ones(n_attn_heads))
            self.attn_bias = nn.Parameter(torch.zeros(self.k))
            self._ascale = self._cscale

        # optional learned per-scatterer frequency envelope (low-rank, shared)
        if n_freq_basis > 0:
            self.wc = _ComplexVec(self.k, n_freq_basis)   # coeffs (K, R)
            self.wb = _ComplexVec(n_freq_basis, s)        # basis  (R, S)

        # optional INDEPENDENT magnitude head: predicts |H| in NORMALISED dB.
        # A position MLP drives a CP-factorised per-element field; a sigmoid
        # keeps it in [0,1] which is de-normalised to [db_lo, db_hi] dB then to
        # linear.  The scatter branch supplies the phase; this head the gain.
        if n_mag_paths > 0:
            self.mag_enc = FourierFeatures(3, n_freqs=mag_freqs, include_input=True)
            self.mag_mlp = nn.Sequential(
                nn.Linear(self.mag_enc.out_dim, 256), nn.ReLU(inplace=True),
                nn.Linear(256, n_mag_paths))
            sc = 1.0 / (n_mag_paths ** 0.5)
            self.mag_u = nn.Parameter(torch.randn(n_mag_paths, m) * sc)
            self.mag_v = nn.Parameter(torch.randn(n_mag_paths, n) * sc)
            self.mag_w = nn.Parameter(torch.randn(n_mag_paths, s) * sc)
            self.register_buffer("db_lo", torch.tensor(-80.0))
            self.register_buffer("db_hi", torch.tensor(20.0))

    def set_db_range(self, lo: float, hi: float) -> None:
        self.db_lo.fill_(float(lo)); self.db_hi.fill_(float(hi))

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
        # positions: (B, 3) raw metres, optionally followed by map-context feats
        b = positions.shape[0]
        mapfeat = positions[:, 3:3 + self.map_dim] if self.map_dim > 0 else None
        positions = positions[:, :3]                                   # xyz only
        # geometry (scatterer positions optionally gradient-refined)
        scat = self.scatterers + self.scat_offset if self.learn_scatterers \
            else self.scatterers
        d_bs = (torch.linalg.norm(scat - self.bs, dim=1).clamp_min(1e-3)
                if self.learn_scatterers else self.d_bs)
        d_ue = torch.cdist(positions, scat).clamp_min(1e-3)            # (B,K)
        tau = d_ue + d_bs[None, :]                                     # (B,K)
        amp = torch.exp(self.log_gain) / (d_ue * d_bs[None, :]) ** self.amp_power

        a = self.a()                                                   # (K,)
        g = amp.to(a.dtype) * a[None, :]                               # (B,K) complex
        # grafted visibility gate (UE-dependent occlusion, low-rank)
        if self.n_vis_rank > 0:
            enc = self.enc_vis(positions / self.coord_scale)
            if mapfeat is not None:                    # append map-context feats
                enc = torch.cat([enc, mapfeat], dim=1)
            f = self.vis_mlp(enc)                                      # (B,R)
            vis = torch.sigmoid(f @ self.vis_g.t())                    # (B,K) in [0,1]
            g = g * vis.to(g.dtype)
        # multi-head cross-attention gate: position (query) attends to scatterer
        # geometry (keys) -> per-scatterer visibility in [0,1]
        if self.n_attn_heads > 0:
            q = self.q_proj(self.enc_q(positions / self._ascale)).reshape(
                b, self.n_attn_heads, self.attn_dim)
            k = self.k_proj(self.enc_k(scat / self._ascale)).reshape(
                self.k, self.n_attn_heads, self.attn_dim)
            sc = torch.einsum("bhd,khd->bhk", q, k) / (self.attn_dim ** 0.5)
            logit = torch.einsum("bhk,h->bk", sc, self.head_w) + self.attn_bias
            g = g * torch.sigmoid(logit).to(g.dtype)
        # scatterer dropout: randomly silence a fraction each step (regulariser)
        if self.training and self.scatter_dropout > 0:
            keep = 1.0 - self.scatter_dropout
            mask = (torch.rand(self.k, device=g.device) < keep).to(g.real.dtype)
            g = g * (mask / keep)[None, :]
        # delay phase ramp: exp(-i 2pi kappa tau s)
        kappa = torch.exp(self.log_kappa)
        phase = (-2 * np.pi * kappa) * (tau[:, :, None] * self.s_idx[None, None, :])
        d = torch.complex(torch.cos(phase), torch.sin(phase))          # (B,K,S)
        gd = g[:, :, None] * d                                         # (B,K,S)
        if self.n_freq_basis > 0:
            w = self.wc() @ self.wb()                                  # (K,S)
            gd = gd * w[None, :, :]

        uv = self._bs_signature(scat)[:, :, None] * self.v()[:, None, :]  # (K,M,N)
        # H(b,m,n,s) = sum_k uv(k,m,n) * gd(b,k,s)  -- no (B,K,M,N,S) intermediate
        h = torch.einsum("kmn,bks->bmns", uv, gd)

        # independent magnitude head: replace |H| with a normalised-dB prediction
        # while keeping the scatter branch's phase.
        if self.n_mag_paths > 0:
            f = self.mag_mlp(self.mag_enc(positions / self._cscale))        # (B,Km)
            raw = torch.einsum("bk,km,kn,ks->bmns", f, self.mag_u,
                               self.mag_v, self.mag_w)
            db = self.db_lo + torch.sigmoid(raw) * (self.db_hi - self.db_lo)
            mag = torch.pow(10.0, db / 20.0)
            phase = h / h.abs().clamp_min(1e-20)
            h = mag.to(h.dtype) * phase
        return h

    def _bs_signature(self, scat: torch.Tensor) -> torch.Tensor:
        """BS array signature U_k: free-learned, or geometric steering toward p_k."""
        if not self.structured_u:
            if self.structured_u2:
                rel = scat - self.bs
                rn = torch.linalg.norm(rel, dim=1).clamp_min(1e-3)
                ah = torch.stack([-torch.sin(self.psi), torch.cos(self.psi),
                                  torch.zeros((), device=rel.device)])
                u = (rel / rn[:, None] * ah[None]).sum(-1)             # (K,) 方位余弦
                wz = rel[:, 2] / rn                                     # (K,) 仰角余弦
                stH = torch.exp(1j * (self.sh * u[:, None] * self.mh_i2))   # (K,MH)
                eb = ((wz + 1) / 2 * (self.EB - 1)).clamp(0, self.EB - 1)
                lo = eb.floor().long(); hi = (lo + 1).clamp(max=self.EB - 1)
                fr = (eb - lo.float())[:, None]
                pat = torch.view_as_complex(self.vpat)                  # (EB,MV)
                stV = pat[lo] * (1 - fr) + pat[hi] * fr                 # (K,MV)
                wh = torch.nn.functional.softplus(self.lobe[:, 0:1]) * self.mh
                wv = torch.nn.functional.softplus(self.lobe[:, 1:2]) * self.mv
                th = torch.exp(-((self.mh_i2[None] - (self.mh-1)/2)/wh)**2)
                tv_i = torch.arange(self.mv, device=wh.device, dtype=torch.float32)
                tv = torch.exp(-((tv_i[None] - (self.mv-1)/2)/wv)**2)
                U = ((stH*th)[:, :, None, None] * (stV*tv)[:, None, :, None]
                     * self.pol2()[:, None, None, :]).reshape(self.k, self.spec.m)
                if self.u2_rank > 0:
                    U = U * (1 + self.u2c() @ self.u2b())
                return U                                                  # (K, MH*MV*MP)
            if self.u_rank > 0:
                return self.u_c() @ self.u_b()          # (K,r)@(r,M) -> (K,M)
            return self.u()
        rel = scat - self.bs                                   # (K,3)
        rn = torch.linalg.norm(rel, dim=1).clamp_min(1e-3)
        ud, vd = rel[:, 1] / rn, rel[:, 2] / rn                # dir cosines (y-z UPA)
        kbs = torch.exp(self.log_kappa_bs)
        ph = kbs * (self.mh_idx[None, :, None] * ud[:, None, None]
                    + self.mv_idx[None, None, :] * vd[:, None, None])   # (K,MH,MV)
        steer = torch.complex(torch.cos(ph), torch.sin(ph))
        pol = self.pol()                                       # (MP,)
        return (steer[:, :, :, None] * pol[None, None, None, :]).reshape(self.k, self.spec.m)
