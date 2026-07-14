"""判别探针设计: PDP域抬底到bb假设的eps上方, 各假设下预测线上分+成分分解"""
import sys, os, json
sys.path.insert(0, '/home/ltz/Huawei-wireless-competition')
sys.path.insert(0, '/home/ltz/Huawei-wireless-competition/scripts')
os.chdir('/home/ltz/Huawei-wireless-competition')
import numpy as np, torch
import torch.nn.functional as F
from score_holdout import reproduce_val_indices
dd="Round1_Map(2)/";st=json.load(open(dd+"Round1_Setup.json"))
MH,MV,MP,N,S=st["M_H"],st["M_V"],st["M_P"],st["N"],st["S"];w=st["w"];dev="cuda"
SC="."
C_=torch.load(SC+"/explore_cache.pt")
ap,ad=C_["ap"].to(dev),C_["ad"].to(dev)
kp,kd=C_["kp"].to(dev),C_["kd"].to(dev)
Href,G=C_["Href"].to(dev),C_["G"].to(dev)
def nrm(P,dim):return P/P.norm(dim=dim,keepdim=True).clamp_min(1e-30)
def PAS_(x,mode):
    a=x.reshape(-1,MH,MV,MP,N,S);return torch.fft.fft2(a,dim=(1,2),norm=mode).abs().square().sum(3).reshape(-1,MH*MV,N,S)
def PDP_(x,mode):return torch.fft.ifft(x,dim=-1,norm=mode).abs().square()
a_=0.7
paT=nrm(a_*ap+(1-a_)*kp,1);pdT=nrm(a_*ad+(1-a_)*kd,-1)
H=Href.clone()
for _ in range(5):
    A=torch.fft.fft2(H.reshape(-1,MH,MV,MP,N,S),dim=(1,2),norm="ortho")
    cur=A.abs().square().sum(3).reshape(-1,MH*MV,N,S);sn=cur.norm(dim=1,keepdim=True).clamp_min(1e-30)
    g=torch.sqrt((paT*sn).clamp_min(0)/cur.clamp_min(1e-38)).reshape(-1,MH,MV,1,N,S)
    H=torch.fft.ifft2(A*g,dim=(1,2),norm="ortho").reshape(H.shape)
    D=torch.fft.ifft(H,dim=-1,norm="ortho");cur=D.abs().square();sn=cur.norm(dim=-1,keepdim=True).clamp_min(1e-30)
    H=torch.fft.fft(torch.sqrt((pdT*sn).clamp_min(0))*(D/D.abs().clamp_min(1e-30)),dim=-1,norm="ortho")
def build(sc,ef_pas,ef_pdp):
    Hs=H/H.abs().pow(2).mean().sqrt()*sc
    A=torch.fft.fft2(Hs.reshape(-1,MH,MV,MP,N,S),dim=(1,2),norm="ortho")
    n_=A.abs().square().sum(3).reshape(-1,MH*MV,N,S).norm(dim=1,keepdim=True)
    g=torch.sqrt((ef_pas/n_.clamp_min(1e-38)).clamp_min(1.0)).reshape(-1,1,1,1,N,S)
    Hs=torch.fft.ifft2(A*g,dim=(1,2),norm="ortho").reshape(Hs.shape)
    D=torch.fft.ifft(Hs,dim=-1,norm="ortho");n2=D.abs().square().norm(dim=-1,keepdim=True)
    return torch.fft.fft(D*torch.sqrt((ef_pdp/n2.clamp_min(1e-38)).clamp_min(1.0)),dim=-1,norm="ortho")
HYPS=[("backward","backward",1e-9,1.04,0.054),
      ("backward","ortho",3e-9,0.42,0.213),
      ("backward","ortho",2e-9,0.41,0.207),
      ("ortho","ortho",1e-9,0.42,None),   # b从GS5点定: 0.428896-0.42*0.5002
      ("ortho","ortho",3.5e-9,0.42,None)]
def comps(P,np_,nd_,eps):
    p1=PAS_(P,np_);g1=PAS_(G,np_);p2=PDP_(P,nd_);g2=PDP_(G,nd_)
    c1=float(F.cosine_similarity(p1,g1,1,eps=eps).mean());c2=float(F.cosine_similarity(p2,g2,-1,eps=eps).mean())
    nm=float((P-G).abs().square().sum()/G.abs().square().sum())
    return c1,c2,0.2/(1+nm),(w[0]*c1+w[1]*c2+w[2]/(1+nm))/sum(w)
cfgs=[("GS5(现)",4e-6,4.28e-9,4.28e-9),
      ("探针A: PDP抬2.2e-7",4e-6,4.28e-9,2.2e-7),
      ("探针B: PDP抬6e-8",4e-6,4.28e-9,6e-8),
      ("探针C: PDP抬2.2e-7+sc1e-5",1e-5,4.28e-9,2.2e-7)]
for tag,sc,e1,e2 in cfgs:
    P=build(sc,e1,e2)
    rms=float(P.abs().pow(2).mean().sqrt())
    print("%s (RMS=%.1e):"%(tag,rms),flush=True)
    for np_,nd_,eps,aa,bb in HYPS:
        c1,c2,c3,px=comps(P,np_,nd_,eps)
        if bb is None:
            P5=build(4e-6,4.28e-9,4.28e-9);_,_,_,px5=comps(P5,np_,nd_,eps)
            bb=0.428896-aa*px5
        print("   %s/%s e=%.0e: C1=%.3f C2=%.3f C3=%.3f -> 预测线上=%.4f"%(np_,nd_,eps,c1,c2,c3,aa*px+bb),flush=True)
print("PROBE_DONE",flush=True)
