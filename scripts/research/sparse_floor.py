"""bb下PDP稀疏抬底: floor方向=目标top-k时延bin, 省L1能量保L2 norm"""
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
def bbproxy(P):
    c1=float(F.cosine_similarity(PAS_(P,"backward"),PAS_(G,"backward"),1,eps=1e-9).mean())
    c2=float(F.cosine_similarity(PDP_(P,"backward"),PDP_(G,"backward"),-1,eps=1e-9).mean())
    nm=float((P-G).abs().square().sum()/G.abs().square().sum())
    return (w[0]*c1+w[1]*c2+w[2]/(1+nm))/sum(w),c1,c2,0.2/(1+nm)
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
aa,bb_=0.91,0.428896-0.91*0.3602
# 基线: 满方向floor(=BLEND_BB)
def floor_full(H,sc,efp,efd):
    Hs=H/H.abs().pow(2).mean().sqrt()*sc
    A=torch.fft.fft2(Hs.reshape(-1,MH,MV,MP,N,S),dim=(1,2),norm="ortho")
    n_=A.abs().square().sum(3).reshape(-1,MH*MV,N,S).norm(dim=1,keepdim=True)
    g=torch.sqrt((efp/n_.clamp_min(1e-38)).clamp_min(1.0)).reshape(-1,1,1,1,N,S)
    Hs=torch.fft.ifft2(A*g,dim=(1,2),norm="ortho").reshape(Hs.shape)
    D=torch.fft.ifft(Hs,dim=-1,norm="ortho");n2=D.abs().square().norm(dim=-1,keepdim=True)
    return torch.fft.fft(D*torch.sqrt((efd/n2.clamp_min(1e-38)).clamp_min(1.0)),dim=-1,norm="ortho")
P0=floor_full(H,6e-6,4.28e-9,2.2e-7)
px,c1,c2,c3=bbproxy(P0)
print("满方向floor(BB基线): C1=%.3f C2=%.3f C3=%.3f bb预测=%.4f RMS=%.2e"%(c1,c2,c3,aa*px+bb_,float(P0.abs().pow(2).mean().sqrt())),flush=True)
# 稀疏floor: 在delay域把不足eps的切片, 加top-k目标bin的delta能量(而不是等比放大整形)
def floor_sparse(H,sc,efp,efd,k):
    Hs=H/H.abs().pow(2).mean().sqrt()*sc
    A=torch.fft.fft2(Hs.reshape(-1,MH,MV,MP,N,S),dim=(1,2),norm="ortho")
    n_=A.abs().square().sum(3).reshape(-1,MH*MV,N,S).norm(dim=1,keepdim=True)
    g=torch.sqrt((efp/n_.clamp_min(1e-38)).clamp_min(1.0)).reshape(-1,1,1,1,N,S)
    Hs=torch.fft.ifft2(A*g,dim=(1,2),norm="ortho").reshape(Hs.shape)
    D=torch.fft.ifft(Hs,dim=-1,norm="ortho")           # (B,M,N,S) delay域复数
    cur=D.abs().square();n2=cur.norm(dim=-1,keepdim=True)   # 当前PDP norm
    need=(n2<efd)                                       # 需要抬的切片
    # top-k mask(用目标pdT的top-k bin)
    topv,topi=pdT.topk(k,dim=-1)
    mask=torch.zeros_like(pdT);mask.scatter_(-1,topi,1.0)
    # 需要的额外power L2: 加x·sparse_shape 使新norm=efd
    # 新PDP = cur + x*shape (近似, 相位对齐加): norm≈sqrt(n2²+x²·(shape norm)²) -> x=sqrt(efd²-n2²)
    shp=nrm(pdT*mask,-1)                                # 单位L2稀疏形状
    x=torch.sqrt((efd**2-n2.square()).clamp_min(0))     # (B,M,N,1)
    addpow=x*shp*need.float()                            # 要加的power profile
    # 转成delay域复数叠加: 与D同相位加(在已有bin), 空bin用随机固定相位
    ph=D/D.abs().clamp_min(1e-30)
    Dnew=torch.sqrt(cur+addpow)*ph
    return torch.fft.fft(Dnew,dim=-1,norm="ortho")
for k in [2,4,8,16,192]:
    P=floor_sparse(H,6e-6,4.28e-9,2.2e-7,k)
    px,c1,c2,c3=bbproxy(P)
    print("稀疏k=%d: C1=%.3f C2=%.3f C3=%.3f bb预测=%.4f RMS=%.2e"%(k,c1,c2,c3,aa*px+bb_,float(P.abs().pow(2).mean().sqrt())),flush=True)
print("SPARSE_DONE",flush=True)
