"""GS9: eps微调成员入池 + 多尺度臂(K8v1+K16f+K32平均) + α重扫 -> val评估+test构建"""
import sys, os, glob, json
sys.path.insert(0, '/home/ltz/Huawei-wireless-competition')
sys.path.insert(0, '/home/ltz/Huawei-wireless-competition/scripts')
os.chdir('/home/ltz/Huawei-wireless-competition')
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree
from wireless_twin.data.map_loader import load_point_cloud
from wireless_twin.models.raytrace2 import build_heightmap
from wireless_twin.evaluation.predictor import (load_model_from_checkpoint,
                                                predict_test_channels, save_test_channels)
from score_holdout import reproduce_val_indices

dd = "Round1_Map(2)/"; st = json.load(open(dd + "Round1_Setup.json"))
MH, MV, MP, N, S = st["M_H"], st["M_V"], st["M_P"], st["N"], st["S"]
w = st["w"]; dev = "cuda"; tiny = torch.finfo(torch.float32).tiny
pos = np.load(dd + "Round1_Train_Pos.npy").astype(np.float32)
ch = np.load(dd + "Round1_Train_Channel.npy")
tpos = np.load(dd + "Round1_Test_Pos.npy").astype(np.float32)
vai = np.array(sorted(reproduce_val_indices(len(pos), 0.1, 0)))
def PAS(x):
    a = x.reshape(-1, MH, MV, MP, N, S)
    return torch.fft.fft2(a, dim=(1, 2), norm="ortho").abs().square().sum(3).reshape(-1, MH*MV, N, S)
def PDP(x): return torch.fft.ifft(x, dim=-1, norm="ortho").abs().square()
def nrm(P, dim): return P / P.norm(dim=dim, keepdim=True).clamp_min(1e-30)
pts = load_point_cloud(dd + "Round1_Map.ply")
hm, x0, y0, res = build_heightmap(pts)
def ind_of(p):
    gx = ((p[:, 0]-x0)/res).astype(int); gy = ((p[:, 1]-y0)/res).astype(int)
    return (hm[gx, gy] > 2.0).astype(np.float32)
indoor = ind_of(pos); indoor_te = ind_of(tpos)
all_pas = torch.zeros(len(pos), MH*MV, N, S, device=dev)
all_pdp = torch.zeros(len(pos), MH*MV*MP, N, S, device=dev)
for c0 in range(0, len(pos), 200):
    cs = slice(c0, c0+200)
    Hc = torch.tensor(ch[cs], dtype=torch.complex64, device=dev)
    all_pas[cs] = nrm(PAS(Hc), 1); all_pdp[cs] = nrm(PDP(Hc), -1); del Hc

class AttnSmall(nn.Module):
    def __init__(self, nf=6):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(nf, 32), nn.ReLU(), nn.Linear(32, 1))
        self.idw_w = nn.Parameter(torch.tensor(2.0))
    def forward(self, feats, logd):
        return torch.softmax(self.mlp(feats).squeeze(-1) - self.idw_w*logd, dim=1)
class AttnWide(nn.Module):
    def __init__(self, nf=6):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(nf, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.idw_w = nn.Parameter(torch.tensor(2.0))
    def forward(self, feats, logd):
        return torch.softmax(self.mlp(feats).squeeze(-1) - self.idw_w*logd, dim=1)

def build_feats(dn, jn, ti, spec, dim, K):
    B = len(dn); nb = spec[jn]
    Y = nb.permute(0, 1, 3, 4, 2).reshape(B, K, -1, MH*MV) if dim == 1 else nb.reshape(B, K, -1, S)
    mean = Y.mean(1, keepdim=True); agree = F.cosine_similarity(Y, mean, dim=-1)
    d_ = torch.tensor(dn, device=dev, dtype=torch.float32)[:, :, None].expand(-1, -1, Y.shape[2])
    inb = torch.tensor(indoor[jn], device=dev)[:, :, None].expand(-1, -1, Y.shape[2])
    it = torch.tensor(ti, device=dev)[:, None, None].expand(-1, K, Y.shape[2])
    feats = torch.stack([d_/3.0, (inb == it).float(), agree, agree*agree,
                         d_*0+1, (d_ < 2.5).float()], -1)
    logd = torch.log(torch.tensor(dn, device=dev, dtype=torch.float32).clamp_min(0.3))[:, :, None]
    return feats, Y, logd

ARMS = [("checkpoints/nbrattn_{t}.pt", AttnSmall, 8, [""]),
        ("checkpoints/nbrattn_{t}_k16s{s}.pt", AttnWide, 16, ["0", "1", "2"]),
        ("checkpoints/nbrattn3_{t}_k16f_s{s}.pt", AttnWide, 16, ["0", "1"]),
        ("checkpoints/nbrattn5_{t}_k32s{s}.pt", AttnWide, 32, ["0", "1"])]
def multi_arm(tgt_pos, ti):
    tree = cKDTree(pos[:, :2])
    outP, outD, cnt = None, None, 0
    for fmt, cls, K, seeds in ARMS:
        dq, jq = tree.query(tgt_pos[:, :2], k=K+1)
        # 目标若在训练集(val), 首列是自身 -> 剔除; test则取前K
        self_hit = (dq[:, 0] < 1e-6)
        dn = np.where(self_hit[:, None], dq[:, 1:K+1], dq[:, :K]).astype(np.float32)
        jn = np.where(self_hit[:, None], jq[:, 1:K+1], jq[:, :K])
        for sd in seeds:
            for tag, dim, spec in [("pas", 1, all_pas), ("pdp", -1, all_pdp)]:
                pth = fmt.format(t=tag, s=sd)
                if not os.path.exists(pth): continue
                m = cls().to(dev); m.load_state_dict(torch.load(pth)); m.eval()
                outs = []
                with torch.no_grad():
                    for i in range(0, len(tgt_pos), 100):
                        b = slice(i, min(i+100, len(tgt_pos)))
                        feats, Y, logd = build_feats(dn[b], jn[b], ti[b], spec, dim, K)
                        outs.append((m(feats, logd)[..., None]*Y).sum(1))
                o = torch.cat(outs)
                if tag == "pas": outP = o if outP is None else outP + o
                else: outD = o if outD is None else outD + o
        cnt += len(seeds)
    kp = nrm((outP/cnt).reshape(len(tgt_pos), N, S, MH*MV).permute(0, 3, 1, 2).contiguous(), 1)
    kd = nrm((outD/cnt).reshape(len(tgt_pos), MH*MV*MP, N, S), -1)
    return kp, kd

EXC = {"ens1", "base_all", "epscap_all", "attn4", "ep_wavg", "round1_real",
       "raymodel", "raymodel2", "raymodel3", "raymodel4", "raymodel5", "nerf2ch",
       "spec_in", "spec_out", "polar_spectra", "geo_anchor"}
BAGS = {f"bag{i}" for i in range(1, 9)}
paths = [p for p in sorted(glob.glob("checkpoints/*.pt"))
         if os.path.basename(p)[:-3] not in EXC and "nbrattn" not in p and "lam_stack" not in p]
G = torch.tensor(ch[vai].reshape(len(vai), MH*MV*MP, N, S), dtype=torch.complex64, device=dev)
gpas = PAS(G); gpdp = PDP(G); gpn = nrm(gpas, 1)
solos, names = [], []
for pth in paths:
    f = os.path.basename(pth)[:-3]
    if f in BAGS: continue
    try:
        m, mt = load_model_from_checkpoint(pth, device=dev)
        p = predict_test_channels(m, pos[vai], mt, device=dev)
        p = p/np.sqrt(np.mean(np.abs(p)**2))
        s = float(F.cosine_similarity(nrm(PAS(torch.tensor(p, dtype=torch.complex64, device=dev)), 1),
                                      gpn, 1, eps=tiny).mean()); del m
        if s > 0.76: continue
        solos.append(s); names.append(f)
    except Exception: pass
solo = torch.tensor(solos, device=dev)
wts = torch.softmax((solo-solo.mean())/0.02, 0)
wmap = dict(zip(names, wts.tolist()))
med = float(np.median(list(wmap.values())))
for b in BAGS:
    if os.path.exists(f"checkpoints/{b}.pt"): wmap[b] = med
print("池=%d(+bag8), 新e35成员在池: %s" % (len(names),
      [n for n in names if n.endswith("_e35")]), flush=True)
def pool_spectra(tgt_pos):
    apS = torch.zeros(len(tgt_pos), MH*MV, N, S, device=dev)
    adS = torch.zeros(len(tgt_pos), MH*MV*MP, N, S, device=dev)
    Href = None
    for pth in paths:
        f = os.path.basename(pth)[:-3]
        if f not in wmap: continue
        m, mt = load_model_from_checkpoint(pth, device=dev)
        p = predict_test_channels(m, tgt_pos, mt, device=dev)
        p = p/np.sqrt(np.mean(np.abs(p)**2)); X = torch.tensor(p, dtype=torch.complex64, device=dev)
        if f == "epscap": Href = X.clone()
        apS += wmap[f]*nrm(PAS(X), 1); adS += wmap[f]*nrm(PDP(X), -1)
        del m, X; torch.cuda.empty_cache()
    return nrm(apS, 1), nrm(adS, -1), Href
def gs_op(paT, pdT, Href):
    H = Href.clone()
    for _ in range(5):
        A = torch.fft.fft2(H.reshape(-1, MH, MV, MP, N, S), dim=(1, 2), norm="ortho")
        cur = A.abs().square().sum(3).reshape(-1, MH*MV, N, S)
        sn = cur.norm(dim=1, keepdim=True).clamp_min(1e-30)
        g = torch.sqrt((paT*sn).clamp_min(0)/cur.clamp_min(1e-38)).reshape(-1, MH, MV, 1, N, S)
        H = torch.fft.ifft2(A*g, dim=(1, 2), norm="ortho").reshape(H.shape)
        D = torch.fft.ifft(H, dim=-1, norm="ortho"); cur = D.abs().square()
        sn = cur.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        H = torch.fft.fft(torch.sqrt((pdT*sn).clamp_min(0))*(D/D.abs().clamp_min(1e-30)),
                          dim=-1, norm="ortho")
    Hs = H/H.abs().pow(2).mean().sqrt()*4e-6
    A = torch.fft.fft2(Hs.reshape(-1, MH, MV, MP, N, S), dim=(1, 2), norm="ortho")
    n_ = A.abs().square().sum(3).reshape(-1, MH*MV, N, S).norm(dim=1, keepdim=True)
    g = torch.sqrt((4.28e-9/n_.clamp_min(1e-38)).clamp_min(1.0)).reshape(-1, 1, 1, 1, N, S)
    Hs = torch.fft.ifft2(A*g, dim=(1, 2), norm="ortho").reshape(Hs.shape)
    D = torch.fft.ifft(Hs, dim=-1, norm="ortho")
    n2 = D.abs().square().norm(dim=-1, keepdim=True)
    return torch.fft.fft(D*torch.sqrt((4.28e-9/n2.clamp_min(1e-38)).clamp_min(1.0)),
                         dim=-1, norm="ortho")
def capC(P, eps=1e-9):
    p1, p2 = PAS(P), PDP(P)
    c1 = float(F.cosine_similarity(p1, gpas, 1, eps=eps).mean())
    c2 = float(F.cosine_similarity(p2, gpdp, -1, eps=eps).mean())
    nm = float((P-G).abs().square().sum()/G.abs().square().sum())
    return (w[0]*c1 + w[1]*c2 + w[2]/(1+nm))/sum(w)
# ---- val ----
kpV, kdV = multi_arm(pos[vai], indoor[vai])
print("多尺度臂val: PAS=%.4f PDP=%.4f" % (
    float(F.cosine_similarity(kpV, gpn, 1, eps=tiny).mean()),
    float(F.cosine_similarity(kdV, nrm(gpdp, -1), -1, eps=tiny).mean())), flush=True)
apV, adV, HrefV = pool_spectra(pos[vai])
best = (-1,)
for a in [0.4, 0.5, 0.6]:
    c = capC(gs_op(nrm(a*apV+(1-a)*kpV, 1), nrm(a*adV+(1-a)*kdV, -1), HrefV))
    if c > best[0]: best = (c, a)
    print("GS9 α=%.1f: capC=%.4f (GS8等效0.5026)" % (a, c), flush=True)
print(">>> GS9 val最优: %.4f @α=%.1f" % best, flush=True)
# ---- test构建 ----
ALPHA = best[1]
kpT, kdT = multi_arm(tpos, indoor_te)
apT, adT, HrefT = pool_spectra(tpos)
out = gs_op(nrm(ALPHA*apT+(1-ALPHA)*kpT, 1), nrm(ALPHA*adT+(1-ALPHA)*kdT, -1), HrefT)
out = out.cpu().numpy().astype(np.complex64)
os.makedirs("best_submit/BLEND_GS9", exist_ok=True)
save_test_channels(out, "best_submit/BLEND_GS9/Round1_Test_Channel.npy")
save_test_channels(out, "submission_round1/Round1_Test_Channel.npy")
print("已写 BLEND_GS9 (%d字节) RMS=%.2e α=%.1f" % (
    os.path.getsize("best_submit/BLEND_GS9/Round1_Test_Channel.npy"),
    float(np.sqrt(np.mean(np.abs(out)**2))), ALPHA), flush=True)
print("GS9DONE", flush=True)
