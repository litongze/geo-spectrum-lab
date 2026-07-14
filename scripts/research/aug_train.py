#!/usr/bin/env python3
"""信道插值数据增强(用户思路2): 相邻同状态训练点的中点生成伪样本,
伪目标=两端真实PAS/PDP的距离加权平均(功率域,不碰复数相位)。
训练: 真样本标准损失 + 伪样本功率域损失(权重lam)。验证val shape能否破0.711/0.813。
"""
import sys, os, json, argparse
sys.path.insert(0, '/home/ltz/Huawei-wireless-competition')
sys.path.insert(0, '/home/ltz/Huawei-wireless-competition/scripts')
os.chdir('/home/ltz/Huawei-wireless-competition')
import numpy as np, torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from wireless_twin.data.map_loader import load_point_cloud
from wireless_twin.models.raytrace2 import build_heightmap
from wireless_twin.models import build_model
from wireless_twin.data.setup_config import ChannelSpec
from score_holdout import reproduce_val_indices

ap = argparse.ArgumentParser()
ap.add_argument("--lam", type=float, default=0.5)
ap.add_argument("--epochs", type=int, default=180)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="checkpoints/aug0.pt")
args = ap.parse_args()

dd = "Round1_Map(2)/"; st = json.load(open(dd + "Round1_Setup.json"))
MH, MV, MP, N, S = st["M_H"], st["M_V"], st["M_P"], st["N"], st["S"]
dev = "cuda"; tiny = torch.finfo(torch.float32).tiny
pos = np.load(dd + "Round1_Train_Pos.npy").astype(np.float32)
ch = np.load(dd + "Round1_Train_Channel.npy")
vi = sorted(reproduce_val_indices(len(pos), 0.1, 0)); vs = set(vi)
tri = np.array([i for i in range(len(pos)) if i not in vs]); vai = np.array(vi)

def PAS(x):
    a = x.reshape(-1, MH, MV, MP, N, S)
    return torch.fft.fft2(a, dim=(1, 2), norm="ortho").abs().square().sum(3).reshape(-1, MH*MV, N, S)
def PDP(x):
    return torch.fft.ifft(x, dim=-1, norm="ortho").abs().square()

# O2I标记(同状态才配对, 避免跨墙插值)
pts = load_point_cloud(dd + "Round1_Map.ply")
hm, x0, y0, res = build_heightmap(pts)
gx = ((pos[:, 0] - x0) / res).astype(int); gy = ((pos[:, 1] - y0) / res).astype(int)
indoor = hm[gx, gy] > 2.0

# ---- 生成伪样本: 每个训练点与其最近同状态邻居的中点 ----
trp = pos[tri]; trind = indoor[tri]
tree = cKDTree(trp[:, :2])
d, idx = tree.query(trp[:, :2], k=4)
pairs = set()
for i in range(len(tri)):
    for j in range(1, 4):
        jj = idx[i, j]
        if d[i, j] <= 3.5 and trind[i] == trind[jj]:
            pairs.add((min(i, jj), max(i, jj)))
            break
pairs = np.array(sorted(pairs))
print("[aug] 伪样本对: %d (阈值3.5m, 同室内状态)" % len(pairs), flush=True)
mid_pos = (trp[pairs[:, 0]] + trp[pairs[:, 1]]) / 2

# 伪目标(功率域, 归一化每切片再平均 -> shape目标)
gt_all = torch.tensor(ch[tri].reshape(len(tri), MH*MV*MP, N, S), dtype=torch.complex64, device=dev)
def nrm(P, dim): return P / P.norm(dim=dim, keepdim=True).clamp_min(1e-30)
with torch.no_grad():
    pas_all = PAS(gt_all); pdp_all = PDP(gt_all)
    mid_pas = nrm(nrm(pas_all[pairs[:, 0]], 1) + nrm(pas_all[pairs[:, 1]], 1), 1)
    mid_pdp = nrm(nrm(pdp_all[pairs[:, 0]], -1) + nrm(pdp_all[pairs[:, 1]], -1), -1)
    del pas_all, pdp_all

# ---- 模型(graft配方) ----
gp0 = torch.load("checkpoints/round1_graft.pt", map_location="cpu", weights_only=False)
spec = ChannelSpec(**gp0["meta"]["spec"])
scale = float(np.asarray(gp0["meta"]["scaler"]["scale"]).flatten()[0])
torch.manual_seed(args.seed); np.random.seed(args.seed)
model = build_model("scatter_field", spec, **gp0["meta"]["model_kwargs"])
rng = np.random.default_rng(args.seed)
model.set_scatterers(pts[rng.choice(len(pts), size=model.k, replace=False)])
model = model.to(dev)
opt = torch.optim.Adam(model.parameters(), lr=3e-3)

tp = torch.tensor(trp, device=dev)
tg = gt_all / scale
mp = torch.tensor(mid_pos, device=dev)
vp = torch.tensor(pos[vai], device=dev)
vg = torch.tensor(ch[vai].reshape(len(vai), MH*MV*MP, N, S), dtype=torch.complex64, device=dev)
gpasv = nrm(PAS(vg), 1); gpdpv = nrm(PDP(vg), -1)

best = -1
n, m_ = len(tri), len(pairs)
for ep in range(1, args.epochs + 1):
    model.train()
    perm = torch.randperm(n, device=dev)
    permM = torch.randperm(m_, device=dev)
    ratio = max(1, int(np.ceil(n / max(m_, 1))))
    for bi, i in enumerate(range(0, n, 64)):
        j = perm[i:i + 64]
        opt.zero_grad()
        h = model(tp[j]); h = h / h.abs().pow(2).mean().clamp_min(1e-30).sqrt()
        c1 = F.cosine_similarity(PAS(h), PAS(tg[j]), 1, eps=1e-12).mean()
        c2 = F.cosine_similarity(PDP(h), PDP(tg[j]), -1, eps=1e-12).mean()
        loss = (1 - c1) + (1 - c2)
        # 伪样本batch(按比例混入)
        k0 = (bi * 32) % max(m_ - 32, 1)
        jm = permM[k0:k0 + 32]
        if len(jm) > 0:
            hm_ = model(mp[jm]); hm_ = hm_ / hm_.abs().pow(2).mean().clamp_min(1e-30).sqrt()
            a1 = F.cosine_similarity(nrm(PAS(hm_), 1), mid_pas[jm], 1, eps=1e-12).mean()
            a2 = F.cosine_similarity(nrm(PDP(hm_), -1), mid_pdp[jm], -1, eps=1e-12).mean()
            loss = loss + args.lam * ((1 - a1) + (1 - a2))
        loss.backward(); opt.step()
    if ep % 20 == 0 or ep == 1:
        model.eval()
        with torch.no_grad():
            hv = model(vp); hv = hv / hv.abs().pow(2).mean().clamp_min(1e-30).sqrt()
            v1 = float(F.cosine_similarity(nrm(PAS(hv), 1), gpasv, 1, eps=tiny).mean())
            v2 = float(F.cosine_similarity(nrm(PDP(hv), -1), gpdpv, -1, eps=tiny).mean())
        if v1 + v2 > best:
            best = v1 + v2
            payload = dict(gp0)
            payload["model_state"] = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(payload, args.out)
        print("[aug lam=%.1f seed=%d] ep%3d val PAS=%.4f PDP=%.4f (best=%.4f)" % (
            args.lam, args.seed, ep, v1, v2, best), flush=True)
print("[aug] done. 基线graft: 0.711/0.813 (和1.524)", flush=True)
