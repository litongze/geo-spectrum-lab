"""量化邻居密度效应: val臂用1800邻居 vs 用全2000(留自身)邻居的质量差"""
import sys, os, json
sys.path.insert(0, '/home/ltz/Huawei-wireless-competition')
sys.path.insert(0, '/home/ltz/Huawei-wireless-competition/scripts')
os.chdir('/home/ltz/Huawei-wireless-competition')
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree
from wireless_twin.data.map_loader import load_point_cloud
from wireless_twin.models.raytrace2 import build_heightmap
from score_holdout import reproduce_val_indices
dd="Round1_Map(2)/";st=json.load(open(dd+"Round1_Setup.json"))
MH,MV,MP,N,S=st["M_H"],st["M_V"],st["M_P"],st["N"],st["S"];dev="cuda"
pos=np.load(dd+"Round1_Train_Pos.npy").astype(np.float32);ch=np.load(dd+"Round1_Train_Channel.npy")
vi=sorted(reproduce_val_indices(len(pos),0.1,0));vs=set(vi)
tri=np.array([i for i in range(len(pos)) if i not in vs]);vai=np.array(vi)
tiny=torch.finfo(torch.float32).tiny;K=16
def PAS(x):
    a=x.reshape(-1,MH,MV,MP,N,S);return torch.fft.fft2(a,dim=(1,2),norm="ortho").abs().square().sum(3).reshape(-1,MH*MV,N,S)
def PDP(x):return torch.fft.ifft(x,dim=-1,norm="ortho").abs().square()
def nrm(P,dim):return P/P.norm(dim=dim,keepdim=True).clamp_min(1e-30)
pts=load_point_cloud(dd+"Round1_Map.ply");hm,x0,y0,res=build_heightmap(pts)
gx=((pos[:,0]-x0)/res).astype(int);gy=((pos[:,1]-y0)/res).astype(int)
indoor=(hm[gx,gy]>2.0).astype(np.float32)
all_pas=torch.zeros(len(pos),MH*MV,N,S,device=dev);all_pdp=torch.zeros(len(pos),MH*MV*MP,N,S,device=dev)
for c0 in range(0,len(pos),200):
    cs=slice(c0,c0+200);Hc=torch.tensor(ch[cs],dtype=torch.complex64,device=dev)
    all_pas[cs]=nrm(PAS(Hc),1);all_pdp[cs]=nrm(PDP(Hc),-1);del Hc
class SliceAttn(nn.Module):
    def __init__(self,nf=6):
        super().__init__()
        self.mlp=nn.Sequential(nn.Linear(nf,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
        self.idw_w=nn.Parameter(torch.tensor(2.0))
    def forward(self,feats,logd):return torch.softmax(self.mlp(feats).squeeze(-1)-self.idw_w*logd,dim=1)
def build_feats(dn,jn,ti,spec,dim):
    B=len(dn);nb=spec[jn]
    Y=nb.permute(0,1,3,4,2).reshape(B,K,-1,MH*MV) if dim==1 else nb.reshape(B,K,-1,S)
    mean=Y.mean(1,keepdim=True);agree=F.cosine_similarity(Y,mean,dim=-1)
    d_=torch.tensor(dn,device=dev,dtype=torch.float32)[:,:,None].expand(-1,-1,Y.shape[2])
    inb=torch.tensor(indoor[jn],device=dev)[:,:,None].expand(-1,-1,Y.shape[2])
    it=torch.tensor(ti,device=dev)[:,None,None].expand(-1,K,Y.shape[2])
    feats=torch.stack([d_/3.0,(inb==it).float(),agree,agree*agree,d_*0+1,(d_<2.5).float()],-1)
    logd=torch.log(torch.tensor(dn,device=dev,dtype=torch.float32).clamp_min(0.3))[:,:,None]
    return feats,Y,logd
def arm_eval(tree_pts_idx,label):
    tree=cKDTree(pos[tree_pts_idx][:,:2])
    kq=K+1 if len(np.intersect1d(tree_pts_idx,vai))>0 else K
    d,j0=tree.query(pos[vai][:,:2],k=kq)
    if kq>K: d,j0=d[:,1:],j0[:,1:]   # 留自身
    jn=tree_pts_idx[j0]
    accP=None;accD=None
    for sd in [0,1,2]:
        for tag,dim,spec in [("pas",1,all_pas),("pdp",-1,all_pdp)]:
            m=SliceAttn().to(dev);m.load_state_dict(torch.load(f"checkpoints/nbrattn_{tag}_k16s{sd}.pt"));m.eval()
            outs=[]
            with torch.no_grad():
                for i in range(0,len(vai),100):
                    b=slice(i,min(i+100,len(vai)))
                    feats,Y,logd=build_feats(d[b].astype(np.float32),jn[b],indoor[vai[b]],spec,dim)
                    outs.append((m(feats,logd)[...,None]*Y).sum(1))
            o=torch.cat(outs)
            if tag=="pas":accP=o if accP is None else accP+o
            else:accD=o if accD is None else accD+o
    gp=nrm(PAS(torch.tensor(ch[vai].reshape(len(vai),MH*MV*MP,N,S),dtype=torch.complex64,device=dev)),1)
    gd=nrm(PDP(torch.tensor(ch[vai].reshape(len(vai),MH*MV*MP,N,S),dtype=torch.complex64,device=dev)),-1)
    kp=nrm((accP/3).reshape(len(vai),N,S,MH*MV).permute(0,3,1,2).contiguous(),1)
    kd=nrm((accD/3).reshape(len(vai),MH*MV*MP,N,S),-1)
    print("%s: 邻距中位%.2fm PAS=%.4f PDP=%.4f"%(label,np.median(d[:,0]),
        float(F.cosine_similarity(kp,gp,1,eps=tiny).mean()),
        float(F.cosine_similarity(kd,gd,-1,eps=tiny).mean())),flush=True)
arm_eval(tri,"1800邻居(现val口径)")
arm_eval(np.arange(len(pos)),"2000邻居留自身(=test口径)")
print("DENSITY_DONE",flush=True)
