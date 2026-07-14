"""构建探索缓存: 温度池加权谱(val) + kNN目标 + Href + gt谱 + kNN能量(截断权重代理)"""
import sys, os, glob, json
sys.path.insert(0, '/home/ltz/Huawei-wireless-competition')
sys.path.insert(0, '/home/ltz/Huawei-wireless-competition/scripts')
os.chdir('/home/ltz/Huawei-wireless-competition')
import numpy as np, torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from wireless_twin.evaluation.predictor import load_model_from_checkpoint, predict_test_channels
from score_holdout import reproduce_val_indices

dd = "Round1_Map(2)/"; st = json.load(open(dd + "Round1_Setup.json"))
MH, MV, MP, N, S = st["M_H"], st["M_V"], st["M_P"], st["N"], st["S"]
dev = "cuda"; tiny = torch.finfo(torch.float32).tiny
pos = np.load(dd + "Round1_Train_Pos.npy").astype(np.float32)
ch = np.load(dd + "Round1_Train_Channel.npy")
vi = sorted(reproduce_val_indices(len(pos), 0.1, 0)); vs = set(vi)
tri = np.array([i for i in range(len(pos)) if i not in vs]); vai = np.array(vi)
def tt(h): return torch.tensor(h, dtype=torch.complex64, device=dev)
def PASt(x):
    a = x.reshape(-1, MH, MV, MP, N, S)
    return torch.fft.fft2(a, dim=(1, 2), norm="ortho").abs().square().sum(3).reshape(-1, MH*MV, N, S)
def PDPt(x): return torch.fft.ifft(x, dim=-1, norm="ortho").abs().square()
def nrm(P, dim): return P / P.norm(dim=dim, keepdim=True).clamp_min(1e-30)
EXC = {"ens1", "base_all", "epscap_all", "attn4", "ep_wavg", "round1_real",
       "raymodel", "raymodel2", "nerf2ch", "spec_in", "spec_out",
       "bag1", "bag2", "bag3", "bag4"}
paths = [p for p in sorted(glob.glob("checkpoints/*.pt"))
         if os.path.basename(p)[:-3] not in EXC and not p.endswith("spectra.pt")]
G = tt(ch[vai]); gpas = PASt(G); gpdp = PDPt(G)
gpn = nrm(gpas, 1)
solos, names = [], []
for pth in paths:
    f = os.path.basename(pth)[:-3]
    try:
        m, mt = load_model_from_checkpoint(pth, device=dev)
        p = predict_test_channels(m, pos[vai], mt, device=dev)
        p = p / np.sqrt(np.mean(np.abs(p) ** 2))
        s = float(F.cosine_similarity(nrm(PASt(tt(p)), 1), gpn, 1, eps=tiny).mean())
        del m
        if s > 0.76: continue
        solos.append(s); names.append(f)
    except Exception: pass
solo = torch.tensor(solos, device=dev)
wts = torch.softmax((solo - solo.mean()) / 0.02, 0)
wmap = dict(zip(names, wts.tolist()))
ap = torch.zeros(len(vai), MH*MV, N, S, device=dev)
ad = torch.zeros(len(vai), MH*MV*MP, N, S, device=dev)
Href = None
for pth in paths:
    f = os.path.basename(pth)[:-3]
    if f not in wmap: continue
    m, mt = load_model_from_checkpoint(pth, device=dev)
    p = predict_test_channels(m, pos[vai], mt, device=dev)
    p = p / np.sqrt(np.mean(np.abs(p) ** 2)); X = tt(p)
    if f == "epscap": Href = X.clone()
    ap += wmap[f] * nrm(PASt(X), 1); ad += wmap[f] * nrm(PDPt(X), -1)
    del m, X; torch.cuda.empty_cache()
# kNN (分块) + 原始能量(截断权重代理)
tr2 = cKDTree(pos[tri][:, :2]); d, idx = tr2.query(pos[vai][:, :2], k=3)
wk = 1.0 / np.clip(d, 0.3, None) ** 2; wk = wk / wk.sum(1, keepdims=True)
kp = torch.zeros_like(ap); kd = torch.zeros_like(ad)
kp_raw = torch.zeros_like(ap)   # 物理尺度能量代理
for j in range(3):
    sel = tri[idx[:, j]]
    for c0 in range(0, len(sel), 200):
        cs = slice(c0, c0 + 200)
        Hn = tt(ch[sel[cs]])
        w_ = torch.tensor(wk[cs, j], dtype=torch.float32, device=dev)[:, None, None, None]
        pp = PASt(Hn)
        kp[cs] += w_ * nrm(pp, 1); kp_raw[cs] += w_ * pp
        kd[cs] += w_ * nrm(PDPt(Hn), -1)
        del Hn, pp
torch.save(dict(ap=ap.cpu(), ad=ad.cpu(), kp=nrm(kp, 1).cpu(), kd=nrm(kd, -1).cpu(),
                kp_raw=kp_raw.cpu(), Href=Href.cpu(),
                gpas=gpas.cpu(), gpdp=gpdp.cpu(), G=G.cpu()),
           "./explore_cache.pt")
print("CACHEDONE", flush=True)
