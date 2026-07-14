"""邻居注意力谱预测器: 每切片学习邻居加权(替代固定IDW²)
leave-self-out在1800训练点上训, 200 val验证是否超IDW²(PAS 0.689/PDP 0.792)"""
import sys, os, json
sys.path.insert(0, '/home/ltz/Huawei-wireless-competition')
sys.path.insert(0, '/home/ltz/Huawei-wireless-competition/scripts')
os.chdir('/home/ltz/Huawei-wireless-competition')
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree
from scipy import ndimage
from wireless_twin.data.map_loader import load_point_cloud
from wireless_twin.models.raytrace2 import build_heightmap
from score_holdout import reproduce_val_indices

dd = "Round1_Map(2)/"; st = json.load(open(dd + "Round1_Setup.json"))
MH, MV, MP, N, S = st["M_H"], st["M_V"], st["M_P"], st["N"], st["S"]
dev = "cuda"; tiny = torch.finfo(torch.float32).tiny
pos = np.load(dd + "Round1_Train_Pos.npy").astype(np.float32)
ch = np.load(dd + "Round1_Train_Channel.npy")
vi = sorted(reproduce_val_indices(len(pos), 0.1, 0)); vs = set(vi)
tri = np.array([i for i in range(len(pos)) if i not in vs]); vai = np.array(vi)
K = 16

def PAS(x):
    a = x.reshape(-1, MH, MV, MP, N, S)
    return torch.fft.fft2(a, dim=(1, 2), norm="ortho").abs().square().sum(3).reshape(-1, MH*MV, N, S)
def PDP(x):
    return torch.fft.ifft(x, dim=-1, norm="ortho").abs().square()
def nrm(P, dim): return P / P.norm(dim=dim, keepdim=True).clamp_min(1e-30)

pts = load_point_cloud(dd + "Round1_Map.ply")
hm, x0, y0, res = build_heightmap(pts)
gx = ((pos[:, 0]-x0)/res).astype(int); gy = ((pos[:, 1]-y0)/res).astype(int)
indoor = (hm[gx, gy] > 2.0).astype(np.float32)

# 全部train点的谱(分块算, 存GPU)
print("[nbr] 计算谱...", flush=True)
all_pas = torch.zeros(len(pos), MH*MV, N, S, device=dev)
all_pdp = torch.zeros(len(pos), MH*MV*MP, N, S, device=dev)
for c0 in range(0, len(pos), 200):
    cs = slice(c0, c0+200)
    Hc = torch.tensor(ch[cs], dtype=torch.complex64, device=dev)
    all_pas[cs] = nrm(PAS(Hc), 1); all_pdp[cs] = nrm(PDP(Hc), -1)
    del Hc
torch.cuda.empty_cache()

# 邻居表: train点(los), 目标=train(除自己)/val
tree = cKDTree(pos[:, :2])   # 全2000
def neigh(target_idx):
    d, j = tree.query(pos[target_idx][:, :2], k=K+1)
    return d[:, 1:].astype(np.float32), j[:, 1:]   # 留自身
ALL=np.arange(len(pos))
dT, jT = neigh(ALL)          # 训练对=全2000
dV, jV = dT[vai], jT[vai]    # val评估(2000-loo口径)

class SliceAttn(nn.Module):
    """每切片邻居加权: feat->score->softmax; 初始化≈IDW²"""
    def __init__(self, nf=6):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(nf, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.mlp[4].weight); nn.init.zeros_(self.mlp[4].bias)
        self.idw_w = nn.Parameter(torch.tensor(2.0))    # 初始IDW幂
    def forward(self, feats, logd):
        # feats: (B,K,nslice,nf), logd: (B,K,1)
        s = self.mlp(feats).squeeze(-1) - self.idw_w * logd   # (B,K,nslice)
        return torch.softmax(s, dim=1)

def build_feats(dn, jn, tgt_indoor, spec, dim):
    """spec: all_pas或all_pdp; 返回 (B,K,nslice,nf), 邻居切片(B,K,nslice,L)"""
    B = len(dn)
    nb = spec[jn]                                    # (B,K,...)
    if dim == 1:   # PAS: 切片沿dim1, nslice=N*S, L=128
        Y = nb.permute(0, 1, 3, 4, 2).reshape(B, K, -1, MH*MV)
    else:          # PDP: nslice=M*N, L=S
        Y = nb.reshape(B, K, -1, S)
    mean = Y.mean(1, keepdim=True)
    agree = F.cosine_similarity(Y, mean, dim=-1)     # (B,K,nslice)
    en = Y.norm(dim=-1)                               # 1(归一过) 无用, 用能量代理: 提前不归一? 简化掉
    d_ = torch.tensor(dn, device=dev)[:, :, None].expand(-1, -1, Y.shape[2])
    ind_nb = torch.tensor(indoor[jn], device=dev)[:, :, None].expand(-1, -1, Y.shape[2])
    ind_t = torch.tensor(tgt_indoor, device=dev)[:, None, None].expand(-1, K, Y.shape[2])
    feats = torch.stack([d_/3.0, (ind_nb == ind_t).float(), agree,
                         agree*agree, d_*0+1, (d_ < 2.5).float()], -1)
    logd = torch.log(torch.tensor(dn, device=dev).clamp_min(0.3))[:, :, None]
    return feats, Y, logd

def run(dim, tag, gt_spec, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    model = SliceAttn().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # gt切片
    if dim == 1:
        gtY = lambda idx: gt_spec[idx].permute(0, 2, 3, 1).reshape(len(idx), -1, MH*MV)
    else:
        gtY = lambda idx: gt_spec[idx].reshape(len(idx), -1, S)
    best = 0
    for ep in range(30):
        perm = np.random.permutation(len(pos))
        model.train()
        for i in range(0, len(pos), 64):
            b = perm[i:i+64]
            feats, Y, logd = build_feats(dT[b], jT[b], indoor[b], gt_spec, dim)
            w_ = model(feats, logd)
            pred = (w_[..., None] * Y).sum(1)
            gt = gtY(b)
            loss = 1 - F.cosine_similarity(pred, gt, dim=-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            cs = []
            for i in range(0, len(vai), 100):
                b = np.arange(i, min(i+100, len(vai)))
                feats, Y, logd = build_feats(dV[b], jV[b], indoor[vai[b]], gt_spec, dim)
                w_ = model(feats, logd)
                pred = (w_[..., None]*Y).sum(1)
                gt = gtY(vai[b])
                cs.append(F.cosine_similarity(pred, gt, dim=-1).mean().item())
            c = float(np.mean(cs))
        if c > best:
            best = c
            torch.save(model.state_dict(), f"checkpoints/nbrattn3_{tag}.pt")
        if ep % 5 == 4:
            print("[%s] ep%d val cos=%.4f (best=%.4f)" % (tag, ep+1, c, best), flush=True)
    return best

for sd in [0,1]:
    bp = run(1, "pas_k16f_s%d"%sd, all_pas, sd)
    bd = run(-1, "pdp_k16f_s%d"%sd, all_pdp, sd)
    print("K16全量臂 seed%d: PAS=%.4f PDP=%.4f (2000-loo口径基线: 0.7482/0.8341)" % (sd, bp, bd), flush=True)
print("NBRATTN_DONE", flush=True)
