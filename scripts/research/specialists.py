#!/usr/bin/env python3
"""室内/室外专家: graft配方在O2I子集上训练, 存为池成员"""
import sys, os, json
sys.path.insert(0, '/home/ltz/Huawei-wireless-competition')
sys.path.insert(0, '/home/ltz/Huawei-wireless-competition/scripts')
os.chdir('/home/ltz/Huawei-wireless-competition')
import numpy as np, torch
import torch.nn.functional as F
from scipy import ndimage
from wireless_twin.data.map_loader import load_point_cloud
from wireless_twin.models.raytrace2 import build_heightmap
from wireless_twin.models import build_model
from wireless_twin.data.setup_config import ChannelSpec
from score_holdout import reproduce_val_indices

dd = "Round1_Map(2)/"
st = json.load(open(dd + "Round1_Setup.json"))
MH, MV, MP, N, S = st["M_H"], st["M_V"], st["M_P"], st["N"], st["S"]
dev = "cuda"
pos = np.load(dd + "Round1_Train_Pos.npy").astype(np.float32)
ch = np.load(dd + "Round1_Train_Channel.npy")
vi = sorted(reproduce_val_indices(len(pos), 0.1, 0)); vs = set(vi)
tri = np.array([i for i in range(len(pos)) if i not in vs])
pts = load_point_cloud(dd + "Round1_Map.ply")
hm, x0, y0, res = build_heightmap(pts)
gx = ((pos[:, 0] - x0) / res).astype(int); gy = ((pos[:, 1] - y0) / res).astype(int)
indoor = hm[gx, gy] > 2.0

gp0 = torch.load("checkpoints/round1_graft.pt", map_location="cpu", weights_only=False)
spec = ChannelSpec(**gp0["meta"]["spec"])
scale = float(np.asarray(gp0["meta"]["scaler"]["scale"]).flatten()[0])

def PAS(x):
    a = x.reshape(-1, MH, MV, MP, N, S)
    return torch.fft.fft2(a, dim=(1, 2), norm="ortho").abs().square().sum(3).reshape(-1, MH*MV, N, S)
def PDP(x):
    return torch.fft.ifft(x, dim=-1, norm="ortho").abs().square()

for tag, mask in [("spec_in", indoor), ("spec_out", ~indoor)]:
    sub = tri[mask[tri]]
    print(f"[{tag}] n={len(sub)}", flush=True)
    m = build_model("scatter_field", spec, **gp0["meta"]["model_kwargs"])
    rng = np.random.default_rng(0)
    idxp = rng.choice(len(pts), size=m.k, replace=False)
    m.set_scatterers(pts[idxp]); m = m.to(dev)
    tp = torch.tensor(pos[sub], device=dev)
    tg = torch.tensor(ch[sub].reshape(len(sub), spec.m, N, S) / scale,
                      dtype=torch.complex64, device=dev)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for ep in range(160):
        perm = torch.randperm(len(sub), device=dev)
        for i in range(0, len(sub), 64):
            j = perm[i:i + 64]; opt.zero_grad()
            hpred = m(tp[j])
            hpred = hpred / hpred.abs().pow(2).mean().clamp_min(1e-30).sqrt()
            c1 = F.cosine_similarity(PAS(hpred), PAS(tg[j]), 1, eps=1e-12).mean()
            c2 = F.cosine_similarity(PDP(hpred), PDP(tg[j]), -1, eps=1e-12).mean()
            (2 - c1 - c2).backward(); opt.step()
        if (ep + 1) % 40 == 0:
            print(f"[{tag}] ep{ep+1}", flush=True)
    payload = dict(gp0); payload["model_state"] = {k: v.cpu() for k, v in m.state_dict().items()}
    torch.save(payload, f"checkpoints/{tag}.pt")
    print(f"[{tag}] saved", flush=True)
print("SPECIALISTSDONE", flush=True)
