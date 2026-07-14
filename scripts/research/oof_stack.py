"""OOF堆叠器: bag1-8各自val划分提供诚实的(成员OOF预测,真值)对,
学逐切片λ(特征): pred=λ·pool侧+(1-λ)·臂侧; 在split0-val上用真池评估capC"""
import sys, os, json
sys.path.insert(0, '/home/ltz/Huawei-wireless-competition')
sys.path.insert(0, '/home/ltz/Huawei-wireless-competition/scripts')
os.chdir('/home/ltz/Huawei-wireless-competition')
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree
from wireless_twin.data.map_loader import load_point_cloud
from wireless_twin.models.raytrace2 import build_heightmap
from wireless_twin.evaluation.predictor import load_model_from_checkpoint, predict_test_channels
from score_holdout import reproduce_val_indices

dd = "Round1_Map(2)/"; st = json.load(open(dd + "Round1_Setup.json"))
MH, MV, MP, N, S = st["M_H"], st["M_V"], st["M_P"], st["N"], st["S"]
w = st["w"]; dev = "cuda"; tiny = torch.finfo(torch.float32).tiny; K = 16
pos = np.load(dd + "Round1_Train_Pos.npy").astype(np.float32)
ch = np.load(dd + "Round1_Train_Channel.npy")
v0 = np.array(sorted(reproduce_val_indices(len(pos), 0.1, 0)))
def PAS(x):
    a = x.reshape(-1, MH, MV, MP, N, S)
    return torch.fft.fft2(a, dim=(1, 2), norm="ortho").abs().square().sum(3).reshape(-1, MH*MV, N, S)
def PDP(x): return torch.fft.ifft(x, dim=-1, norm="ortho").abs().square()
def nrm(P, dim): return P / P.norm(dim=dim, keepdim=True).clamp_min(1e-30)
pts = load_point_cloud(dd + "Round1_Map.ply")
hm, x0, y0, res = build_heightmap(pts)
gx = ((pos[:, 0]-x0)/res).astype(int); gy = ((pos[:, 1]-y0)/res).astype(int)
indoor = (hm[gx, gy] > 2.0).astype(np.float32)
all_pas = torch.zeros(len(pos), MH*MV, N, S, device=dev)
all_pdp = torch.zeros(len(pos), MH*MV*MP, N, S, device=dev)
for c0 in range(0, len(pos), 200):
    cs = slice(c0, c0+200)
    Hc = torch.tensor(ch[cs], dtype=torch.complex64, device=dev)
    all_pas[cs] = nrm(PAS(Hc), 1); all_pdp[cs] = nrm(PDP(Hc), -1); del Hc
tree = cKDTree(pos[:, :2])
dA, jA0 = tree.query(pos[:, :2], k=K+1)
dA, jA = dA[:, 1:].astype(np.float32), jA0[:, 1:]

class SliceAttn(nn.Module):
    def __init__(self, nf=6):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(nf, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.idw_w = nn.Parameter(torch.tensor(2.0))
    def forward(self, feats, logd):
        return torch.softmax(self.mlp(feats).squeeze(-1) - self.idw_w*logd, dim=1)
def build_feats(dn, jn, ti, spec, dim):
    B = len(dn); nb = spec[jn]
    Y = nb.permute(0, 1, 3, 4, 2).reshape(B, K, -1, MH*MV) if dim == 1 else nb.reshape(B, K, -1, S)
    mean = Y.mean(1, keepdim=True); agree = F.cosine_similarity(Y, mean, dim=-1)
    d_ = torch.tensor(dn, device=dev, dtype=torch.float32)[:, :, None].expand(-1, -1, Y.shape[2])
    inb = torch.tensor(indoor[jn], device=dev)[:, :, None].expand(-1, -1, Y.shape[2])
    it = torch.tensor(ti, device=dev)[:, None, None].expand(-1, K, Y.shape[2])
    feats = torch.stack([d_/3.0, (inb == it).float(), agree, agree*agree,
                         d_*0+1, (d_ < 2.5).float()], -1)
    logd = torch.log(torch.tensor(dn, device=dev, dtype=torch.float32).clamp_min(0.3))[:, :, None]
    return feats, Y, logd, agree
def arm_out(idx, tag, dim, spec):
    acc = None
    for sd in [0, 1]:
        m = SliceAttn().to(dev)
        m.load_state_dict(torch.load(f"checkpoints/nbrattn3_{tag}_k16f_s{sd}.pt")); m.eval()
        outs, ags = [], []
        with torch.no_grad():
            for i in range(0, len(idx), 100):
                b = idx[i:i+100]
                feats, Y, logd, agree = build_feats(dA[b], jA[b], indoor[b], spec, dim)
                outs.append((m(feats, logd)[..., None]*Y).sum(1)); ags.append(agree.mean(1))
        o = torch.cat(outs)
        acc = o if acc is None else acc + o
    return acc/2, torch.cat(ags)
# ---- OOF训练对 ----
Xs, Ys, Gs = [], [], []
for k in range(1, 9):
    vk = np.array(sorted(reproduce_val_indices(len(pos), 0.1, k)))
    m, mt = load_model_from_checkpoint(f"checkpoints/bag{k}.pt", device=dev)
    p = predict_test_channels(m, pos[vk], mt, device=dev)
    p = p/np.sqrt(np.mean(np.abs(p)**2)); X = torch.tensor(p, dtype=torch.complex64, device=dev); del m
    bag_pas = nrm(PAS(X), 1)                                     # 池侧OOF代理
    arm_pas, agree = arm_out(vk, "pas", 1, all_pas)
    arm_pas = nrm(arm_pas.reshape(len(vk), N, S, MH*MV).permute(0, 3, 1, 2).contiguous(), 1)
    gtp = nrm(PAS(torch.tensor(ch[vk].reshape(len(vk), MH*MV*MP, N, S), dtype=torch.complex64, device=dev)), 1)
    # 特征per切片: 池臂一致性, 邻居一致性, 是否室内
    c_pa = F.cosine_similarity(bag_pas, arm_pas, 1, eps=tiny)     # (B,N,S)
    agr = agree.reshape(len(vk), N, S)
    ind = torch.tensor(indoor[vk], device=dev)[:, None, None].expand(-1, N, S)
    Xs.append(torch.stack([c_pa, agr, ind], -1).reshape(-1, 3))
    Ys.append((bag_pas.permute(0, 2, 3, 1).reshape(-1, MH*MV),
               arm_pas.permute(0, 2, 3, 1).reshape(-1, MH*MV)))
    Gs.append(gtp.permute(0, 2, 3, 1).reshape(-1, MH*MV))
Xf = torch.cat(Xs); Gf = torch.cat(Gs)
Pf = torch.cat([y[0] for y in Ys]); Af = torch.cat([y[1] for y in Ys])
print("OOF切片样本: %d" % len(Xf), flush=True)
lam_net = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 1)).to(dev)
opt = torch.optim.Adam(lam_net.parameters(), lr=1e-3)
for ep in range(8):
    perm = torch.randperm(len(Xf), device=dev)
    for i in range(0, len(Xf), 8192):
        b = perm[i:i+8192]; opt.zero_grad()
        lam = torch.sigmoid(lam_net(Xf[b]))
        pred = lam*Pf[b] + (1-lam)*Af[b]
        loss = 1 - F.cosine_similarity(pred, Gf[b], dim=-1).mean()
        loss.backward(); opt.step()
    with torch.no_grad():
        lam = torch.sigmoid(lam_net(Xf))
        c = float(F.cosine_similarity(lam*Pf+(1-lam)*Af, Gf, dim=-1).mean())
        c_fix = float(F.cosine_similarity(0.6*Pf+0.4*Af, Gf, dim=-1).mean())
    print("ep%d 学习λ: OOF-cos=%.4f (固定0.6: %.4f) λ均值=%.2f" % (
        ep, c, c_fix, float(lam.mean())), flush=True)
torch.save(lam_net.state_dict(), "checkpoints/lam_stack_pas.pt")
print("OOFSTACK_DONE", flush=True)
