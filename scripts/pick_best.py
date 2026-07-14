#!/usr/bin/env python3
"""评估 checkpoints 下所有候选(干净val, eps1e-9代理), 排名, 用最优构建测试集交付。
可反复调用(每个训练阶段后), OVERNIGHT_BEST 始终反映当前最优。"""
import sys, os, json, glob
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np, torch
import torch.nn.functional as F
from wireless_twin.evaluation.predictor import (
    load_model_from_checkpoint, predict_test_channels, save_test_channels)
from score_holdout import reproduce_val_indices

dd = "Round1_Map(2)/"; st = json.load(open(dd + "Round1_Setup.json"))
MH, MV, MP, N, S = st["M_H"], st["M_V"], st["M_P"], st["N"], st["S"]; w = st["w"]
pos = np.load(dd + "Round1_Train_Pos.npy").astype(np.float32)
ch = np.load(dd + "Round1_Train_Channel.npy")
tpos = np.load(dd + "Round1_Test_Pos.npy").astype(np.float32)
vi = reproduce_val_indices(len(pos), 0.1, 0); vp, vg = pos[vi], ch[vi]
G = torch.tensor(vg, dtype=torch.complex64, device="cuda")

def T(h):
    x = h.reshape(-1, MH, MV, MP, N, S)
    return (torch.fft.fft2(x, dim=(1, 2), norm="ortho").abs().square().sum(3).reshape(-1, MH*MV, N, S),
            torch.fft.ifft(h, dim=-1, norm="ortho").abs().square())
gp, gd = T(G); tiny = torch.finfo(torch.float32).tiny

def evalp(path):
    m, mt = load_model_from_checkpoint(path, device="cuda")
    p = predict_test_channels(m, vp, mt, device="cuda"); p = p / np.sqrt(np.mean(np.abs(p)**2))
    P = torch.tensor(p, dtype=torch.complex64, device="cuda"); p1, p2 = T(P)
    sh1 = float(F.cosine_similarity(p1, gp, 1, eps=tiny).mean())
    sh2 = float(F.cosine_similarity(p2, gd, -1, eps=tiny).mean())
    best, bsc = -1, 2e-5
    for r in [1.5e-5, 2e-5, 2.5e-5, 3e-5, 4e-5, 5e-5]:
        q1, q2 = T(P * r)
        c1 = float(F.cosine_similarity(q1, gp, 1, eps=1e-9).mean())
        c2 = float(F.cosine_similarity(q2, gd, -1, eps=1e-9).mean())
        nm = float((P*r - G).abs().square().sum() / G.abs().square().sum())
        C = (w[0]*c1 + w[1]*c2 + w[2]/(1+nm)) / sum(w)
        if C > best: best, bsc = C, r
    return sh1, sh2, best, bsc

paths = ["checkpoints/round1_graft.pt", "checkpoints/epscap.pt"] + sorted(glob.glob("checkpoints/ep_*.pt"))
res = []
print("\n===== 候选排名(干净val, %d个) =====" % len(paths))
for pth in paths:
    if not os.path.exists(pth): continue
    try:
        sh1, sh2, C, sc = evalp(pth); res.append((os.path.basename(pth)[:-3], C, sc, sh1, sh2))
    except Exception as e:
        print("  %s 出错: %s" % (pth, e))
res.sort(key=lambda x: -x[1])
for name, C, sc, sh1, sh2 in res:
    print("  %-20s shape=%.3f/%.3f | eps1e-9 C=%.4f @%.0e ~线上%.3f" % (name, sh1, sh2, C, sc, C-0.047))
if not res:
    print("无候选"); sys.exit(0)
best = res[0]
print("\n>>> 当前最优: %s  C=%.4f  估线上~%.3f  (epscap基线=0.4798)" % (best[0], best[1], best[1]-0.047))

# 构建最优交付
m, mt = load_model_from_checkpoint(f"checkpoints/{best[0]}.pt", device="cuda")
pt = predict_test_channels(m, tpos, mt, device="cuda")
pt = (pt / np.sqrt(np.mean(np.abs(pt)**2)) * best[2]).astype(np.complex64)
os.makedirs("best_submit/OVERNIGHT_BEST", exist_ok=True)
save_test_channels(pt, "best_submit/OVERNIGHT_BEST/Round1_Test_Channel.npy")
sz = os.path.getsize("best_submit/OVERNIGHT_BEST/Round1_Test_Channel.npy")
rank = "\n".join("  %2d. %-20s C=%.4f ~线上%.3f" % (i+1, r[0], r[1], r[1]-0.047)
                 for i, r in enumerate(res))
open("best_submit/OVERNIGHT_BEST/说明.txt", "w").write(
"""通宵最优候选(自动构建, 会随训练进展刷新)
==========================================
模型: %s
eps1e-9 代理 C = %.4f    估算线上 ~%.3f
提交尺度 RMS = %.0e     文件 = %d 字节(应为786432128)

【决策】若 C > 0.4798 → 本文件为新最优, 提交它; 否则提交 EPSCAP_2e-5(线上~0.433稳妥).

完整排名:
%s
""" % (best[0], best[1], best[1]-0.047, best[2], sz, rank))
print("已刷新 best_submit/OVERNIGHT_BEST/ (%d字节)" % sz)
