import sys, os, json
sys.path.insert(0, '/home/ltz/Huawei-wireless-competition')
sys.path.insert(0, '/home/ltz/Huawei-wireless-competition/scripts')
os.chdir('/home/ltz/Huawei-wireless-competition')
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from score_holdout import reproduce_val_indices

dd = "Round1_Map(2)/"; st = json.load(open(dd + "Round1_Setup.json"))
MH, MV, MP, N, S = st["M_H"], st["M_V"], st["M_P"], st["N"], st["S"]; M = MH*MV*MP
pos = np.load(dd + "Round1_Train_Pos.npy").astype(np.float32)
ch = np.load(dd + "Round1_Train_Channel.npy")
vi = sorted(reproduce_val_indices(len(pos), 0.1, 0)); vs = set(vi)
tri = np.array([i for i in range(len(pos)) if i not in vs]); vai = np.array(vi)
dev = "cuda"; tiny = torch.finfo(torch.float32).tiny
sc = float(np.sqrt(np.mean(np.abs(ch) ** 2)))
gt = torch.tensor(ch.reshape(-1, M, N, S) / sc, dtype=torch.complex64, device=dev)
mag_db = 20*np.log10(np.abs(ch[::7][np.abs(ch[::7]) > 0]) / sc)
lo, hi = np.percentile(mag_db, [0.5, 99.5])
print("dB范围 [%.0f,%.0f]" % (lo, hi), flush=True)
def PAS(x):
    a = x.reshape(-1, MH, MV, MP, N, S)
    return torch.fft.fft2(a, dim=(1, 2), norm="ortho").abs().square().sum(3).reshape(-1, MH*MV, N, S)
def PDP(x): return torch.fft.ifft(x, dim=-1, norm="ortho").abs().square()
def ff(x, L=8):
    f = 2.0 ** torch.arange(L, device=dev) * np.pi
    s = (x[:, :, None] / 250.0 * f).reshape(x.shape[0], -1)
    return torch.cat([x / 250.0, torch.sin(s), torch.cos(s)], 1)
tp = torch.tensor(pos[tri], device=dev); vp = torch.tensor(pos[vai], device=dev)
Xt, Xv = ff(tp), ff(vp); din = Xt.shape[1]
vat = torch.tensor(vai, device=dev); trt = torch.tensor(tri, device=dev)
gpv = PAS(gt[vat]); gdv = PDP(gt[vat]); gtt = gt[trt]
D = M * N * S
for mode in ["reim", "polar"]:
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(din, 768), nn.ReLU(),
                        nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, D * 2)).to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    best = 0
    for ep in range(60):
        net.train(); perm = torch.randperm(len(tri), device=dev)
        for i in range(0, len(tri), 64):
            j = perm[i:i + 64]; opt.zero_grad()
            o = net(Xt[j]).reshape(-1, D, 2)
            if mode == "reim":
                H = torch.complex(o[..., 0], o[..., 1]).reshape(-1, M, N, S)
            else:
                db = lo + torch.sigmoid(o[..., 0]) * (hi - lo)
                H = (torch.pow(10.0, db / 20.0) *
                     torch.exp(1j * o[..., 1].to(torch.complex64))).reshape(-1, M, N, S)
            Hn = H / H.abs().pow(2).mean().clamp_min(1e-30).sqrt()
            c1 = F.cosine_similarity(PAS(Hn), PAS(gtt[j]), 1, eps=1e-12).mean()
            c2 = F.cosine_similarity(PDP(Hn), PDP(gtt[j]), -1, eps=1e-12).mean()
            (2 - c1 - c2).backward(); opt.step()
        if (ep + 1) % 15 == 0:
            net.eval()
            with torch.no_grad():
                o = net(Xv).reshape(-1, D, 2)
                if mode == "reim":
                    H = torch.complex(o[..., 0], o[..., 1]).reshape(-1, M, N, S)
                else:
                    db = lo + torch.sigmoid(o[..., 0]) * (hi - lo)
                    H = (torch.pow(10.0, db / 20.0) *
                         torch.exp(1j * o[..., 1].to(torch.complex64))).reshape(-1, M, N, S)
                v1 = float(F.cosine_similarity(PAS(H), gpv, 1, eps=tiny).mean())
                v2 = float(F.cosine_similarity(PDP(H), gdv, -1, eps=tiny).mean())
            best = max(best, v1 + v2)
            print("  [%s] ep%d val PAS=%.4f PDP=%.4f" % (mode, ep + 1, v1, v2), flush=True)
    print(">>> %s 最佳和=%.4f (散射模型基线1.524)" % (mode, best), flush=True)
print("POLARDONE", flush=True)
