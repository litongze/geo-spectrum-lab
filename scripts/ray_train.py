#!/usr/bin/env python3
"""Ray-traced channel assembly: geometric paths + ~25 structural learnables.

Paths (LoS / ground / facade reflections, transmission-aware) are enumerated by
the image method; the assembly learns only structural constants (per-type gains
and phases, per-wall-crossing attenuation, array steering constants, pol
responses).  Geometry does the generalising: val PAS/PDP is the verdict.
"""
from __future__ import annotations
import sys, json, argparse
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from wireless_twin.data.map_loader import load_point_cloud
from wireless_twin.models.raytrace2 import (build_heightmap, extract_facades,
                                            HeightmapScene, trace_paths3)
from score_holdout import reproduce_val_indices

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=300)
ap.add_argument("--lr", type=float, default=3e-2)
ap.add_argument("--max-paths", type=int, default=24)
ap.add_argument("--out", default="checkpoints/raymodel.pt")
args = ap.parse_args()

dd = "Round1_Map(2)/"; st = json.load(open(dd + "Round1_Setup.json"))
MH, MV, MP, N, S = st["M_H"], st["M_V"], st["M_P"], st["N"], st["S"]
NV, NP = st["N_V"], st["N_P"]
dev = "cuda"; tiny = torch.finfo(torch.float32).tiny
pos = np.load(dd + "Round1_Train_Pos.npy").astype(np.float32)
ch = np.load(dd + "Round1_Train_Channel.npy")
vi = sorted(reproduce_val_indices(len(pos), 0.1, 0)); vs = set(vi)
tri = np.array([i for i in range(len(pos)) if i not in vs]); vai = np.array(vi)

print("[ray] building scene...", flush=True)
pts = load_point_cloud(dd + "Round1_Map.ply")
hm, x0, y0, res = build_heightmap(pts)
fac = extract_facades(hm, x0, y0, res)
scene = HeightmapScene(hm, x0, y0, res, fac, device=dev)
bs = torch.tensor([50., 0., 25.], device=dev)
print("[ray] tracing %d UEs x %d facades..." % (len(pos), scene.F), flush=True)
P = {}
allp = trace_paths3(scene, bs, torch.tensor(pos, device=dev), max_paths=args.max_paths)
print("[ray] paths ready: mean crossings %.1f" % float(allp["ncx"].mean()), flush=True)


class Assembly(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_kappa = nn.Parameter(torch.tensor(np.log(9.95e-5), dtype=torch.float32))
        self.g0 = nn.Parameter(torch.zeros(3))            # per-type log-amp
        self.beta_x = nn.Parameter(torch.tensor(1.2))     # nepers per wall crossing
        self.beta_L = nn.Parameter(torch.tensor(0.05))    # nepers per indoor metre
        self.gamma = nn.Parameter(torch.tensor(1.0))      # tau exponent
        self.phi0 = nn.Parameter(torch.zeros(3))          # per-type phase
        self.beta_tau = nn.Parameter(torch.tensor(0.0))   # rad per metre
        self.psi = nn.Parameter(torch.tensor(150.0*np.pi/180))  # 拟合值150°(93%命中)
        self.sh = nn.Parameter(torch.tensor(-2*np.pi, dtype=torch.float32))  # 1.0λ,符号-1(拟合)
        self.sv = nn.Parameter(torch.tensor(np.pi, dtype=torch.float32))
        self.su = nn.Parameter(torch.tensor(np.pi, dtype=torch.float32))
        self.polB = nn.Parameter(torch.tensor([1., 0., 0., 1.]))  # (2pol x re/im)
        self.polU = nn.Parameter(torch.tensor([1., 0., 0., 1.]))
        # per-type angular spread: antenna-domain Laplacian窗 -> 展宽PAS峰
        self.wH = nn.Parameter(torch.full((3,), -1.0))
        self.wV = nn.Parameter(torch.full((3,), -1.0))
        self.mh_i = torch.arange(MH, device=dev).float()
        self.mv_i = torch.arange(MV, device=dev).float()
        self.s_i = torch.arange(S, device=dev).float()

    def forward(self, p):
        tau, dep, arr = p["tau"], p["dep"], p["arr"]
        typ, ncx, lin = p["typ"], p["ncx"], p["lin"]
        amp = torch.exp(self.g0[typ] - F.softplus(self.beta_x) * ncx
                        - F.softplus(self.beta_L) * lin) / tau.clamp_min(1.0) ** self.gamma
        phase = self.phi0[typ] + self.beta_tau * tau
        c = amp * torch.exp(1j * phase.to(torch.complex64))                    # (B,P)
        # BS array frame: horizontal axis (-sin psi, cos psi, 0), vertical z
        ah = torch.stack([-torch.sin(self.psi), torch.cos(self.psi),
                          torch.zeros((), device=dev)])
        u = (dep * ah[None, None]).sum(-1)                                    # (B,P)
        wz = dep[..., 2]
        envH = torch.exp(-F.softplus(self.wH)[typ][..., None]
                         * (self.mh_i - (len(self.mh_i) - 1) / 2).abs() / len(self.mh_i))
        envV = torch.exp(-F.softplus(self.wV)[typ][..., None]
                         * (self.mv_i - (len(self.mv_i) - 1) / 2).abs() / len(self.mv_i))
        stH = torch.exp(1j * (self.sh * u[..., None] * self.mh_i)) * envH     # (B,P,MH)
        stV = torch.exp(1j * (self.sv * wz[..., None] * self.mv_i)) * envV    # (B,P,MV)
        stU = torch.exp(1j * (self.su * arr[..., 2:3] * torch.arange(NV, device=dev).float()))  # (B,P,NV)
        pb = torch.complex(self.polB[[0, 2]], self.polB[[1, 3]])              # (2,)
        pu = torch.complex(self.polU[[0, 2]], self.polU[[1, 3]])
        dly = torch.exp(-2j * np.pi * torch.exp(self.log_kappa)
                        * tau[..., None] * self.s_i)                          # (B,P,S)
        # H[b, mh, mv, pb, nv, pu, s]
        H = torch.einsum("bp,bph,bpv,q,bpu,r,bps->bhvqurs",
                         c, stH, stV, pb, stU, pu, dly)
        B = tau.shape[0]
        return H.reshape(B, MH * MV * MP, NV * NP, S)


def pas(x):
    a = x.reshape(-1, MH, MV, MP, N, S)
    return torch.fft.fft2(a, dim=(1, 2), norm="ortho").abs().square().sum(3).reshape(-1, MH * MV, N, S)
def pdp(x):
    return torch.fft.ifft(x, dim=-1, norm="ortho").abs().square()


model = Assembly().to(dev)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
gt = torch.tensor(ch.reshape(len(pos), MH * MV * MP, N, S), dtype=torch.complex64, device=dev)
gt = gt / gt.abs().pow(2).mean().sqrt()
def batch_paths(idx):
    return {k: v[idx] for k, v in allp.items()}
tr_t = torch.tensor(tri, device=dev); va_t = torch.tensor(vai, device=dev)
gpasv, gpdpv = pas(gt[va_t]), pdp(gt[va_t])

best = -1
for ep in range(1, args.epochs + 1):
    model.train()
    perm = tr_t[torch.randperm(len(tri), device=dev)]
    for i in range(0, len(tri), 256):
        j = perm[i:i + 256]
        opt.zero_grad()
        Hp = model(batch_paths(j))
        Hp = Hp / Hp.abs().pow(2).mean().clamp_min(1e-30).sqrt()
        c1 = F.cosine_similarity(pas(Hp), pas(gt[j]), 1, eps=1e-12).mean()
        c2 = F.cosine_similarity(pdp(Hp), pdp(gt[j]), -1, eps=1e-12).mean()
        (2 - c1 - c2).backward()
        opt.step()
    if ep % 10 == 0 or ep == 1:
        model.eval()
        with torch.no_grad():
            Hv = model(batch_paths(va_t))
            Hv = Hv / Hv.abs().pow(2).mean().clamp_min(1e-30).sqrt()
            v1 = float(F.cosine_similarity(pas(Hv), gpasv, 1, eps=tiny).mean())
            v2 = float(F.cosine_similarity(pdp(Hv), gpdpv, -1, eps=tiny).mean())
        if v1 + v2 > best:
            best = v1 + v2
            torch.save({"state": model.state_dict()}, args.out)
        print("[ray] ep%3d val PAS=%.4f PDP=%.4f (best_sum=%.4f)" % (ep, v1, v2, best), flush=True)
print("[ray] done. (对比: 散射模型0.711/0.813, blend目标0.738/0.833)", flush=True)
