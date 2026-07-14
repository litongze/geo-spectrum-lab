"""ScatterField@几何锚点: 立面中点(多高度)+角点+屋顶沿, 替代随机点云采样"""
import sys, os, json
sys.path.insert(0, '/home/ltz/Huawei-wireless-competition')
sys.path.insert(0, '/home/ltz/Huawei-wireless-competition/scripts')
os.chdir('/home/ltz/Huawei-wireless-competition')
import numpy as np, torch
import torch.nn.functional as F
from wireless_twin.data.map_loader import load_point_cloud
from wireless_twin.models.raytrace2 import build_heightmap, extract_facades
from wireless_twin.models import build_model
from wireless_twin.data.setup_config import ChannelSpec
from score_holdout import reproduce_val_indices
dd="Round1_Map(2)/";st=json.load(open(dd+"Round1_Setup.json"))
MH,MV,MP,N,S=st["M_H"],st["M_V"],st["M_P"],st["N"],st["S"];dev="cuda"
pos=np.load(dd+"Round1_Train_Pos.npy").astype(np.float32);ch=np.load(dd+"Round1_Train_Channel.npy")
vi=sorted(reproduce_val_indices(len(pos),0.1,0));vs=set(vi)
tri=np.array([i for i in range(len(pos)) if i not in vs]);vai=np.array(vi)
pts=load_point_cloud(dd+"Round1_Map.ply")
hm,x0,y0,res=build_heightmap(pts)
fac=extract_facades(hm,x0,y0,res)
# 几何锚点集
anchors=[]
for f in fac:
    p,t,e,h=f["p"],f["t"],f["e"],f["h"]
    # 立面: 沿切向3点 x 高度4层
    for u in [-0.6,0.0,0.6]:
        for z in np.linspace(1.5,max(h-1,2),4):
            anchors.append([p[0]+u*e*t[0],p[1]+u*e*t[1],z])
    # 角点(段端) x 2高度
    for sgn in [-1,1]:
        for z in [h*0.5,h-0.5]:
            anchors.append([p[0]+sgn*e*t[0],p[1]+sgn*e*t[1],z])
    # 屋顶沿
    anchors.append([p[0],p[1],h+0.3])
anchors=np.array(anchors,dtype=np.float32)
print("几何锚点: %d个"%len(anchors),flush=True)
gp0=torch.load("checkpoints/round1_graft.pt",map_location="cpu",weights_only=False)
spec=ChannelSpec(**gp0["meta"]["spec"])
scale=float(np.asarray(gp0["meta"]["scaler"]["scale"]).flatten()[0])
def PAS(x):
    a=x.reshape(-1,MH,MV,MP,N,S);return torch.fft.fft2(a,dim=(1,2),norm="ortho").abs().square().sum(3).reshape(-1,MH*MV,N,S)
def PDP(x):return torch.fft.ifft(x,dim=-1,norm="ortho").abs().square()
tiny=torch.finfo(torch.float32).tiny
K=gp0["meta"]["model_kwargs"]["n_scatterers"]  # 2048
rng=np.random.default_rng(0)
if len(anchors)>=K: sel=anchors[rng.choice(len(anchors),K,replace=False)]
else: sel=np.concatenate([anchors,pts[rng.choice(len(pts),K-len(anchors),replace=False)]])
torch.manual_seed(0)
m=build_model("scatter_field",spec,**gp0["meta"]["model_kwargs"])
m.set_scatterers(sel);m=m.to(dev)
tp=torch.tensor(pos[tri],device=dev)
tg=torch.tensor(ch[tri].reshape(len(tri),spec.m,N,S)/scale,dtype=torch.complex64,device=dev)
vp=torch.tensor(pos[vai],device=dev)
vg=torch.tensor(ch[vai].reshape(len(vai),spec.m,N,S),dtype=torch.complex64,device=dev)
gpv=PAS(vg);gdv=PDP(vg)
opt=torch.optim.Adam(m.parameters(),lr=3e-3)
best=0
for ep in range(170):
    perm=torch.randperm(len(tri),device=dev)
    m.train()
    for i in range(0,len(tri),64):
        j=perm[i:i+64];opt.zero_grad()
        h=m(tp[j]);h=h/h.abs().pow(2).mean().clamp_min(1e-30).sqrt()
        c1=F.cosine_similarity(PAS(h),PAS(tg[j]),1,eps=1e-12).mean()
        c2=F.cosine_similarity(PDP(h),PDP(tg[j]),-1,eps=1e-12).mean()
        (2-c1-c2).backward();opt.step()
    if (ep+1)%20==0:
        m.eval()
        with torch.no_grad():
            hv=m(vp);hv=hv/hv.abs().pow(2).mean().clamp_min(1e-30).sqrt()
            v1=float(F.cosine_similarity(PAS(hv),gpv,1,eps=tiny).mean())
            v2=float(F.cosine_similarity(PDP(hv),gdv,-1,eps=tiny).mean())
        if v1+v2>best:
            best=v1+v2
            pl=dict(gp0);pl["model_state"]={k:v.cpu() for k,v in m.state_dict().items()}
            torch.save(pl,"checkpoints/geo_anchor.pt")
        print("[geo] ep%d val PAS=%.4f PDP=%.4f (graft基线0.711/0.813)"%(ep+1,v1,v2),flush=True)
print("GEO_DONE best=%.4f"%best,flush=True)
