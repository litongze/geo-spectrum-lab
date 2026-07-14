#!/usr/bin/env python3
"""Clean learning curve: does MORE training data raise the eps-capped score on a
truly-held-out set?  Fix a clean holdout K (never trained on), train the base on
N points (excluding K) with the shape loss, fine-tune with the eps-capped loss,
and score on K.  A rising trend => the all-1800 model beats the 1620 one.
"""
from __future__ import annotations
import sys, json
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np, torch
import torch.nn.functional as F
from pathlib import Path
from wireless_twin.data.map_loader import load_point_cloud
from wireless_twin.data.setup_config import ChannelSpec
from wireless_twin.models import build_model
from score_holdout import reproduce_val_indices

dd = "Round1_Map(2)"
st = json.load(open(dd + "/Round1_Setup.json"))
MH, MV, MP, N, S = st["M_H"], st["M_V"], st["M_P"], st["N"], st["S"]
w = st["w"]; dev = "cuda"
pos = np.load(dd + "/Round1_Train_Pos.npy").astype(np.float32)
ch = np.load(dd + "/Round1_Train_Channel.npy")
spec = ChannelSpec(m=256, n=N, s=S, mh=MH, mv=MV, mp=MP,
                   bs_position=tuple(st["BS_Position"]) if "BS_Position" in st else (0, 0, 20),
                   metric_weights=tuple(w))
# get bs_position + scaler exactly as graft did (reuse its meta)
gp0 = torch.load("checkpoints/round1_graft.pt", map_location="cpu", weights_only=False)
spec = ChannelSpec(**gp0["meta"]["spec"])
scale = gp0["meta"]["scaler"]["scale"]
scale = float(np.asarray(scale).flatten()[0])
pts = load_point_cloud(Path(dd) / "Round1_Map.ply")

K = np.array(sorted(reproduce_val_indices(len(pos), 0.1, 0)))     # clean holdout
pool = np.array([i for i in range(len(pos)) if i not in set(K)])   # 1600

Gk = torch.tensor(ch[K], dtype=torch.complex64, device=dev)
def T(h):
    a = h.reshape(-1, MH, MV, MP, N, S)
    return (torch.fft.fft2(a, dim=(1, 2), norm="ortho").abs().square().sum(3).reshape(-1, MH*MV, N, S),
            torch.fft.ifft(h, dim=-1, norm="ortho").abs().square())
gpk, gdk = T(Gk)
def scoreK(model, meta):
    from wireless_twin.evaluation.predictor import predict_test_channels
    p = predict_test_channels(model, pos[K], meta, device=dev)
    p = p / np.sqrt(np.mean(np.abs(p) ** 2))
    P = torch.tensor(p, dtype=torch.complex64, device=dev)
    best = -1
    for r in [1.5e-5, 2e-5, 2.5e-5, 3e-5, 4e-5]:
        p1, p2 = T(P * r)
        c1 = float(F.cosine_similarity(p1, gpk, 1, eps=1e-9).mean())
        c2 = float(F.cosine_similarity(p2, gdk, -1, eps=1e-9).mean())
        nm = float((P*r - Gk).abs().square().sum() / Gk.abs().square().sum())
        best = max(best, (w[0]*c1 + w[1]*c2 + w[2]/(1+nm)) / sum(w))
    return best

def make_model(seed):
    m = build_model("scatter_field", spec, **gp0["meta"]["model_kwargs"])
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pts), size=m.k, replace=len(pts) < m.k)
    m.set_scatterers(pts[idx])
    return m.to(dev)

def train(model, idx, base_ep, ft_ep):
    tp = torch.tensor(pos[idx], device=dev)
    tg = torch.tensor(ch[idx].reshape(len(idx), spec.m, N, S), dtype=torch.complex64, device=dev)
    tgn = tg / scale
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    trms = float(tgn.abs().pow(2).mean().sqrt())
    for ep in range(base_ep):     # base: shape loss (normalised pred)
        perm = torch.randperm(len(idx), device=dev)
        for i in range(0, len(idx), 64):
            j = perm[i:i+64]; opt.zero_grad()
            ph = model(tp[j]); ph = ph / ph.abs().pow(2).mean().clamp_min(1e-30).sqrt()
            p1, p2 = T(ph); g1, g2 = T(tgn[j])
            c1 = F.cosine_similarity(p1, g1, 1, eps=1e-12).mean()
            c2 = F.cosine_similarity(p2, g2, -1, eps=1e-12).mean()
            (2 - c1 - c2).backward(); opt.step()
    for g in opt.param_groups: g["lr"] = 2e-4
    for ep in range(ft_ep):       # ft: eps-capped loss at physical scale
        perm = torch.randperm(len(idx), device=dev)
        for i in range(0, len(idx), 64):
            j = perm[i:i+64]; opt.zero_grad()
            ph = model(tp[j]); ph = ph / ph.abs().pow(2).mean().clamp_min(1e-30).sqrt() * trms
            p1, p2 = T(ph); g1, g2 = T(tgn[j])
            d1 = p1.norm(dim=1).clamp_min(1e-9) * g1.norm(dim=1).clamp_min(1e-9)
            d2 = p2.norm(dim=-1).clamp_min(1e-9) * g2.norm(dim=-1).clamp_min(1e-9)
            c1 = ((p1*g1).sum(1) / d1).mean(); c2 = ((p2*g2).sum(-1) / d2).mean()
            (2 - c1 - c2).backward(); opt.step()

meta = dict(gp0["meta"])
for Ntr in [800, 1200, 1600]:
    model = make_model(0)
    train(model, pool[:Ntr], base_ep=110, ft_ep=30)
    print(f"[curve] N_train={Ntr:4d} -> clean-K eps1e-9 C={scoreK(model, meta):.4f}", flush=True)
