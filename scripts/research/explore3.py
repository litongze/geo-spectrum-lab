import sys, os, json
sys.path.insert(0, '/home/ltz/Huawei-wireless-competition')
os.chdir('/home/ltz/Huawei-wireless-competition')
import numpy as np, torch
import torch.nn.functional as F
dd="Round1_Map(2)/"; st=json.load(open(dd+"Round1_Setup.json"))
MH,MV,MP,N,S=st["M_H"],st["M_V"],st["M_P"],st["N"],st["S"];w=st["w"];dev="cuda"
C_=torch.load("./explore_cache.pt")
ap,ad=C_["ap"].to(dev),C_["ad"].to(dev)
kp,kd=C_["kp"].to(dev),C_["kd"].to(dev)
Href,gpas,gpdp,G=C_["Href"].to(dev),C_["gpas"].to(dev),C_["gpdp"].to(dev),C_["G"].to(dev)
def PASt(x):
    a=x.reshape(-1,MH,MV,MP,N,S);return torch.fft.fft2(a,dim=(1,2),norm="ortho").abs().square().sum(3).reshape(-1,MH*MV,N,S)
def PDPt(x):return torch.fft.ifft(x,dim=-1,norm="ortho").abs().square()
def nrm(P,dim):return P/P.norm(dim=dim,keepdim=True).clamp_min(1e-30)
def capC3(P,eps):
    p1,p2=PASt(P),PDPt(P)
    c1=float(F.cosine_similarity(p1,gpas,1,eps=eps).mean());c2=float(F.cosine_similarity(p2,gpdp,-1,eps=eps).mean())
    nm=float((P-G).abs().square().sum()/G.abs().square().sum())
    return (w[0]*c1+w[1]*c2+w[2]/(1+nm))/sum(w),c1,c2,0.2/(1+nm)
a=0.7
paT=nrm(a*ap+(1-a)*kp,1);pdT=nrm(a*ad+(1-a)*kd,-1)
H=Href.clone()
for _ in range(5):
    A=torch.fft.fft2(H.reshape(-1,MH,MV,MP,N,S),dim=(1,2),norm="ortho")
    cur=A.abs().square().sum(3).reshape(-1,MH*MV,N,S);sn=cur.norm(dim=1,keepdim=True).clamp_min(1e-30)
    g=torch.sqrt((paT*sn).clamp_min(0)/cur.clamp_min(1e-38)).reshape(-1,MH,MV,1,N,S)
    H=torch.fft.ifft2(A*g,dim=(1,2),norm="ortho").reshape(H.shape)
    D=torch.fft.ifft(H,dim=-1,norm="ortho");cur=D.abs().square();sn=cur.norm(dim=-1,keepdim=True).clamp_min(1e-30)
    H=torch.fft.fft(torch.sqrt((pdT*sn).clamp_min(0))*(D/D.abs().clamp_min(1e-30)),dim=-1,norm="ortho")
def floor_scale(H,sc,ef):
    Hs=H/H.abs().pow(2).mean().sqrt()*sc
    A=torch.fft.fft2(Hs.reshape(-1,MH,MV,MP,N,S),dim=(1,2),norm="ortho")
    n_=A.abs().square().sum(3).reshape(-1,MH*MV,N,S).norm(dim=1,keepdim=True)
    g=torch.sqrt((ef*1.02/n_.clamp_min(1e-38)).clamp_min(1.0)).reshape(-1,1,1,1,N,S)
    Hs=torch.fft.ifft2(A*g,dim=(1,2),norm="ortho").reshape(Hs.shape)
    D=torch.fft.ifft(Hs,dim=-1,norm="ortho");n2=D.abs().square().norm(dim=-1,keepdim=True)
    return torch.fft.fft(D*torch.sqrt((ef*1.02/n2.clamp_min(1e-38)).clamp_min(1.0)),dim=-1,norm="ortho")
print("scale x floor 联合网格 (capC@1e-9 / @2.5e-9, C1,C2,C3项@1e-9):",flush=True)
for sc in [1.2e-5,1.0e-5,0.8e-5,0.6e-5,0.4e-5]:
    for ef in [3e-9,4e-9]:
        Hf=floor_scale(H,sc,ef)
        t1,c1,c2,c3=capC3(Hf,1e-9);t25,_,_,_=capC3(Hf,2.5e-9)
        print("  sc=%.1e ef=%.0e: %.4f/%.4f (C1=%.3f C2=%.3f C3项=%.4f)"%(sc,ef,t1,t25,c1,c2,c3),flush=True)
print("EXP3DONE",flush=True)
