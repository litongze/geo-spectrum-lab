"""用GS3/GS4线上差反解真实eps, 再找新最优(scale,floor)"""
import sys, os, json
sys.path.insert(0, '/home/ltz/Huawei-wireless-competition')
sys.path.insert(0, '/home/ltz/Huawei-wireless-competition/scripts')
os.chdir('/home/ltz/Huawei-wireless-competition')
import numpy as np, torch
import torch.nn.functional as F
dd="Round1_Map(2)/";st=json.load(open(dd+"Round1_Setup.json"))
MH,MV,MP,N,S=st["M_H"],st["M_V"],st["M_P"],st["N"],st["S"];w=st["w"];dev="cuda"
SC="."
C_=torch.load(SC+"/explore_cache.pt")
ap,ad=C_["ap"].to(dev),C_["ad"].to(dev)
kp,kd=C_["kp"].to(dev),C_["kd"].to(dev)
Href,gpas,gpdp,G=C_["Href"].to(dev),C_["gpas"].to(dev),C_["gpdp"].to(dev),C_["G"].to(dev)
def PASt(x):
    a=x.reshape(-1,MH,MV,MP,N,S);return torch.fft.fft2(a,dim=(1,2),norm="ortho").abs().square().sum(3).reshape(-1,MH*MV,N,S)
def PDPt(x):return torch.fft.ifft(x,dim=-1,norm="ortho").abs().square()
def nrm(P,dim):return P/P.norm(dim=dim,keepdim=True).clamp_min(1e-30)
def capC(P,eps):
    p1,p2=PASt(P),PDPt(P)
    c1=float(F.cosine_similarity(p1,gpas,1,eps=eps).mean());c2=float(F.cosine_similarity(p2,gpdp,-1,eps=eps).mean())
    nm=float((P-G).abs().square().sum()/G.abs().square().sum())
    return (w[0]*c1+w[1]*c2+w[2]/(1+nm))/sum(w)
a=0.7
paT=nrm(a*ap+(1-a)*kp,1);pdT=nrm(a*ad+(1-a)*kd,-1)
H0=Href.clone()
for _ in range(5):
    A=torch.fft.fft2(H0.reshape(-1,MH,MV,MP,N,S),dim=(1,2),norm="ortho")
    cur=A.abs().square().sum(3).reshape(-1,MH*MV,N,S);sn=cur.norm(dim=1,keepdim=True).clamp_min(1e-30)
    g=torch.sqrt((paT*sn).clamp_min(0)/cur.clamp_min(1e-38)).reshape(-1,MH,MV,1,N,S)
    H0=torch.fft.ifft2(A*g,dim=(1,2),norm="ortho").reshape(H0.shape)
    D=torch.fft.ifft(H0,dim=-1,norm="ortho");cur=D.abs().square();sn=cur.norm(dim=-1,keepdim=True).clamp_min(1e-30)
    H0=torch.fft.fft(torch.sqrt((pdT*sn).clamp_min(0))*(D/D.abs().clamp_min(1e-30)),dim=-1,norm="ortho")
def floor_(H,sc,ef):
    Hs=H/H.abs().pow(2).mean().sqrt()*sc
    A=torch.fft.fft2(Hs.reshape(-1,MH,MV,MP,N,S),dim=(1,2),norm="ortho")
    n_=A.abs().square().sum(3).reshape(-1,MH*MV,N,S).norm(dim=1,keepdim=True)
    g=torch.sqrt((ef*1.02/n_.clamp_min(1e-38)).clamp_min(1.0)).reshape(-1,1,1,1,N,S)
    Hs=torch.fft.ifft2(A*g,dim=(1,2),norm="ortho").reshape(Hs.shape)
    D=torch.fft.ifft(Hs,dim=-1,norm="ortho");n2=D.abs().square().norm(dim=-1,keepdim=True)
    return torch.fft.fft(D*torch.sqrt((ef*1.02/n2.clamp_min(1e-38)).clamp_min(1.0)),dim=-1,norm="ortho")
H3=floor_(H0,1.8e-5,3e-9)   # ≈GS3配置
H4=floor_(H0,6e-6,3e-9)     # ≈GS4配置
print("eps假设  C(GS3cfg)  C(GS4cfg)  Δ(GS3-GS4)  [线上Δ=+0.0017]")
for eps in [1e-9,2.5e-9,4e-9,5e-9,7e-9,1e-8,1.5e-8,2e-8]:
    c3=capC(H3,eps);c4=capC(H4,eps)
    print("  %.1e: %.4f  %.4f  %+.4f"%(eps,c3,c4,c3-c4),flush=True)
print("EPSFIT_DONE",flush=True)
