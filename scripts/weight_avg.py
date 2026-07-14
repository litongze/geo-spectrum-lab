#!/usr/bin/env python3
"""Average weights of same-base fine-tunes (aligned basin -> no phase cancel)."""
import sys, json, argparse
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np, torch
import torch.nn.functional as F
from wireless_twin.evaluation.predictor import load_model_from_checkpoint, predict_test_channels
from score_holdout import reproduce_val_indices
ap=argparse.ArgumentParser(); ap.add_argument("--members",nargs="+",required=True)
ap.add_argument("--out",default=None); ap.add_argument("--datadir",default="Round1_Map(2)")
a=ap.parse_args()
dd=a.datadir+"/"; st=json.load(open(dd+"Round1_Setup.json"))
MH,MV,MP,N,S=st["M_H"],st["M_V"],st["M_P"],st["N"],st["S"];w=st["w"]
pos=np.load(dd+"Round1_Train_Pos.npy").astype(np.float32);ch=np.load(dd+"Round1_Train_Channel.npy")
vi=reproduce_val_indices(len(pos),0.1,0);vp,vg=pos[vi],ch[vi]
G=torch.tensor(vg,dtype=torch.complex64,device="cuda")
def T(h):
    x=h.reshape(-1,MH,MV,MP,N,S);return torch.fft.fft2(x,dim=(1,2),norm="ortho").abs().square().sum(3).reshape(-1,MH*MV,N,S),torch.fft.ifft(h,dim=-1,norm="ortho").abs().square()
gp,gd=T(G)
def C(P,eps=1e-9):
    p1,p2=T(P);c1=float(F.cosine_similarity(p1,gp,1,eps=eps).mean());c2=float(F.cosine_similarity(p2,gd,-1,eps=eps).mean())
    nm=float((P-G).abs().square().sum()/G.abs().square().sum());return (w[0]*c1+w[1]*c2+w[2]/(1+nm))/sum(w)
def evalckpt(path=None,state=None,meta=None):
    m,mt=load_model_from_checkpoint(a.members[0],device="cuda")
    if state is not None: m.load_state_dict({k:v.to("cuda") for k,v in state.items()})
    p=predict_test_channels(m,vp,mt,device="cuda");p=p/np.sqrt(np.mean(np.abs(p)**2))
    P=torch.tensor(p,dtype=torch.complex64,device="cuda")
    return max((C(P*r),r) for r in [1.5e-5,2e-5,2.5e-5,3e-5,4e-5,5e-5])
# 加载各成员state
payloads=[torch.load(m,map_location="cpu",weights_only=False) for m in a.members]
states=[p["model_state"] for p in payloads]
# 平均float参数(buffer取第一个)
avg={}
for k in states[0]:
    if states[0][k].is_floating_point() or states[0][k].is_complex():
        avg[k]=sum(s[k].float() for s in states)/len(states) if not states[0][k].is_complex() else sum(s[k] for s in states)/len(states)
    else: avg[k]=states[0][k]
b,r=evalckpt(state=avg);print("权重平均(%d成员) C=%.4f @%.0e ~线上%.3f"%(len(states),b,r,b-0.047))
if a.out:
    payloads[0]["model_state"]={k:v.cpu() for k,v in avg.items()};torch.save(payloads[0],a.out);print("saved",a.out)
