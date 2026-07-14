"""bb下C1/C2上限 + PDP强切片下压测试"""
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
def PAS_(x,m_):
    a=x.reshape(-1,MH,MV,MP,N,S);return torch.fft.fft2(a,dim=(1,2),norm=m_).abs().square().sum(3).reshape(-1,MH*MV,N,S)
def PDP_(x,m_):return torch.fft.ifft(x,dim=-1,norm=m_).abs().square()
tiny=torch.finfo(torch.float32).tiny
# gt bb域norm
g1=PAS_(G,"backward");g2=PDP_(G,"backward")
n1=g1.norm(dim=1);n2=g2.norm(dim=-1)
eps=1e-9
print("bb-PAS gt-norm分位[10,50,90]:",["%.1e"%v for v in np.percentile(n1.cpu().numpy(),[10,50,90])],flush=True)
print("bb-PDP gt-norm分位[10,50,90]:",["%.1e"%v for v in np.percentile(n2.cpu().numpy(),[10,50,90])],flush=True)
print("C1绝对上限(cos=1) = %.4f"%float(torch.minimum(torch.ones_like(n1),n1/eps).mean()),flush=True)
print("C2绝对上限(cos=1) = %.4f"%float(torch.minimum(torch.ones_like(n2),n2/eps).mean()),flush=True)
# 我们目标shape的cos-per-slice(ortho域target vs gt)
a_=0.7
paT=nrm(a_*ap+(1-a_)*kp,1);pdT=nrm(a_*ad+(1-a_)*kd,-1)
cos1=F.cosine_similarity(paT,nrm(PAS_(G,"ortho"),1),1,eps=tiny)     # (B,N,S)
cos2=F.cosine_similarity(pdT,nrm(PDP_(G,"ortho"),-1),-1,eps=tiny)   # (B,M,N)
print("C1可达(现shape) = %.4f"%float((cos1*torch.minimum(torch.ones_like(n1),n1/eps)).mean()),flush=True)
print("C2可达(现shape) = %.4f"%float((cos2*torch.minimum(torch.ones_like(n2),n2/eps)).mean()),flush=True)
# bb-max总分估计: C3 at floor-能量
print("(BB2实测: C1=0.630 C2=0.192 C3=0.058)",flush=True)
# ==== 强切片下压 ====
H=Href.clone()
for _ in range(5):
    A=torch.fft.fft2(H.reshape(-1,MH,MV,MP,N,S),dim=(1,2),norm="ortho")
    cur=A.abs().square().sum(3).reshape(-1,MH*MV,N,S);sn=cur.norm(dim=1,keepdim=True).clamp_min(1e-30)
    g=torch.sqrt((paT*sn).clamp_min(0)/cur.clamp_min(1e-38)).reshape(-1,MH,MV,1,N,S)
    H=torch.fft.ifft2(A*g,dim=(1,2),norm="ortho").reshape(H.shape)
    D=torch.fft.ifft(H,dim=-1,norm="ortho");cur=D.abs().square();sn=cur.norm(dim=-1,keepdim=True).clamp_min(1e-30)
    H=torch.fft.fft(torch.sqrt((pdT*sn).clamp_min(0))*(D/D.abs().clamp_min(1e-30)),dim=-1,norm="ortho")
aa,bb_=0.91,0.428896-0.91*0.3602
def bbproxy(P):
    c1=float(F.cosine_similarity(PAS_(P,"backward"),g1,1,eps=eps).mean())
    c2=float(F.cosine_similarity(PDP_(P,"backward"),g2,-1,eps=eps).mean())
    nm=float((P-G).abs().square().sum()/G.abs().square().sum())
    return (w[0]*c1+w[1]*c2+w[2]/(1+nm))/sum(w),c1,c2,0.2/(1+nm)
def build(sc,capk):   # k=1稀疏抬底 + 强切片下压到capk*2.2e-7
    Hs=H/H.abs().pow(2).mean().sqrt()*sc
    A=torch.fft.fft2(Hs.reshape(-1,MH,MV,MP,N,S),dim=(1,2),norm="ortho")
    n_=A.abs().square().sum(3).reshape(-1,MH*MV,N,S).norm(dim=1,keepdim=True)
    g=torch.sqrt((4.28e-9/n_.clamp_min(1e-38)).clamp_min(1.0)).reshape(-1,1,1,1,N,S)
    Hs=torch.fft.ifft2(A*g,dim=(1,2),norm="ortho").reshape(Hs.shape)
    D=torch.fft.ifft(Hs,dim=-1,norm="ortho")
    cur=D.abs().square();nn=cur.norm(dim=-1,keepdim=True)
    if capk is not None:  # 下压
        dc=(capk*2.2e-7/nn.clamp_min(1e-38)).clamp_max(1.0)
        D=D*torch.sqrt(dc);cur=D.abs().square();nn=cur.norm(dim=-1,keepdim=True)
    need=(nn<2.2e-7)
    topi=pdT.argmax(dim=-1,keepdim=True)
    mask=torch.zeros_like(pdT);mask.scatter_(-1,topi,1.0)
    x=torch.sqrt((2.2e-7**2-nn.square()).clamp_min(0))
    ph=D/D.abs().clamp_min(1e-30)
    return torch.fft.fft(torch.sqrt(cur+x*mask*need.float())*ph,dim=-1,norm="ortho")
for capk in [None,4.0,2.0,1.0]:
    P=build(6e-6,capk)
    px,c1,c2,c3=bbproxy(P)
    print("下压cap=%s: C1=%.3f C2=%.3f C3=%.3f bb预测=%.4f RMS=%.1e"%(str(capk),c1,c2,c3,aa*px+bb_,float(P.abs().pow(2).mean().sqrt())),flush=True)
print("CEIL_DONE",flush=True)
