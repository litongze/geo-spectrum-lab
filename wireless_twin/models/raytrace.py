"""Image-method ray tracer over an oriented-box city + learnable channel assembly.

The map is a set of flat-roofed buildings (oriented boxes) extracted from the
point cloud, plus a ground plane.  For each UE we enumerate geometrically-exact
specular paths (LoS, ground bounce, first-order wall bounces) with the image
method, validate reflection points and occlusion against all boxes, and output
per-path (delay tau, BS direction, UE direction, metadata).  A small learnable
head then assembles H from the paths; only *structural* parameters are learned
(reflection coefficients per building, array constants, per-type phases), so
the geometry does the generalising - not memorisation.
"""
from __future__ import annotations

import numpy as np
import torch


# --------------------------------------------------------------------------
# Scene
# --------------------------------------------------------------------------
class BoxScene:
    """Oriented boxes: centers (K,2), angles (K,), half-extents (K,2), heights (K,)."""

    def __init__(self, boxes: list[dict], device="cuda"):
        self.dev = device
        self.K = len(boxes)
        self.c = torch.tensor([[b["cx"], b["cy"]] for b in boxes], dtype=torch.float32, device=device)
        self.th = torch.tensor([b["th"] for b in boxes], dtype=torch.float32, device=device)
        self.he = torch.tensor([[b["lx"] / 2, b["ly"] / 2] for b in boxes], dtype=torch.float32, device=device)
        self.h = torch.tensor([b["h"] for b in boxes], dtype=torch.float32, device=device)
        cs, sn = torch.cos(self.th), torch.sin(self.th)
        # rows of R map world->box frame: [cos,sin],[-sin,cos]
        self.R = torch.stack([torch.stack([cs, sn], -1), torch.stack([-sn, cs], -1)], 1)  # (K,2,2)
        # wall planes: 4 per box; plane i has outward normal n, point p on plane,
        # in-plane tangent t (horizontal) with half-extent e_t, vertical extent [0,h]
        ns, ps, ts, et, bid = [], [], [], [], []
        for k in range(self.K):
            Rk = self.R[k]  # world->box
            for (axis, sgn) in ((0, 1), (0, -1), (1, 1), (1, -1)):
                n_box = torch.zeros(2, device=device); n_box[axis] = sgn
                n_w = Rk.T @ n_box                       # box->world
                t_w = Rk.T @ torch.tensor([-n_box[1], n_box[0]], device=device)
                p_w = self.c[k] + n_w * self.he[k][axis]
                ns.append(n_w); ps.append(p_w); ts.append(t_w)
                et.append(self.he[k][1 - axis]); bid.append(k)
        self.wn = torch.stack(ns)      # (W,2) outward normals (horizontal)
        self.wp = torch.stack(ps)      # (W,2) points on wall planes
        self.wt = torch.stack(ts)      # (W,2) tangents
        self.we = torch.stack(et)      # (W,) tangent half extents
        self.wb = torch.tensor(bid, device=device)  # (W,) building id
        self.Wn = self.wn.shape[0]

    def seg_blocked(self, a: torch.Tensor, b: torch.Tensor, skip_eps=1e-3) -> torch.Tensor:
        """Segments a->b: (B,3) x (B,3) -> (B,) bool blocked by any box (slab test)."""
        B = a.shape[0]
        rel_a = a[:, None, :2] - self.c[None]                # (B,K,2)
        rel_b = b[:, None, :2] - self.c[None]
        a2 = torch.einsum("kij,bkj->bki", self.R, rel_a)
        b2 = torch.einsum("kij,bkj->bki", self.R, rel_b)
        d2 = b2 - a2
        lo = torch.cat([-self.he, torch.zeros(self.K, 1, device=self.dev)], 1)   # (K,3) box bounds
        hi = torch.cat([self.he, self.h[:, None]], 1)
        az = a[:, None, 2:3].expand(-1, self.K, -1)
        dz = (b - a)[:, None, 2:3].expand(-1, self.K, -1)
        A = torch.cat([a2, az], -1)                          # (B,K,3)
        D = torch.cat([d2, dz], -1)
        Dsafe = torch.where(D.abs() < 1e-8, torch.full_like(D, 1e-8), D)
        t1 = (lo[None] - A) / Dsafe
        t2 = (hi[None] - A) / Dsafe
        tmin = torch.minimum(t1, t2).amax(-1)
        tmax = torch.maximum(t1, t2).amin(-1)
        hit = (tmax > tmin) & (tmax > skip_eps) & (tmin < 1 - skip_eps)
        return hit.any(1)


# --------------------------------------------------------------------------
# Path enumeration (image method)
# --------------------------------------------------------------------------
def trace_paths(scene: BoxScene, bs: torch.Tensor, ue: torch.Tensor,
                max_paths: int = 48):
    """bs:(3,) ue:(B,3) -> padded path tensors.

    Returns dict of tensors (B, P): tau (m), valid mask, and (B,P,3) unit
    directions at BS (departure) and path type / building ids (B,P).
    Types: 0=LoS, 1=ground, 2=wall.
    """
    dev = ue.device
    B = ue.shape[0]
    taus, dirs, typs, bids, oks = [], [], [], [], []

    def add(tau, dep, typ, bid, ok):
        taus.append(tau); dirs.append(dep); typs.append(typ); bids.append(bid); oks.append(ok)

    # ---- LoS
    v = ue - bs[None]
    tau = v.norm(dim=-1)
    dep = v / tau[:, None].clamp_min(1e-6)
    ok = ~scene.seg_blocked(bs[None].expand(B, -1), ue)
    add(tau, dep, torch.zeros(B, dtype=torch.long, device=dev),
        torch.full((B,), -1, dtype=torch.long, device=dev), ok)

    # ---- ground bounce (image of BS at -z)
    bsg = bs.clone(); bsg = torch.tensor([bs[0], bs[1], -bs[2]], device=dev)
    v = ue - bsg[None]
    tau = v.norm(dim=-1)
    t_hit = bs[2] / (bs[2] + ue[:, 2]).clamp_min(1e-6)       # param along image->ue where z=0
    pref = bsg[None] + v * t_hit[:, None]                    # reflection point (z=0)
    ok = (~scene.seg_blocked(bs[None].expand(B, -1), pref)) & (~scene.seg_blocked(pref, ue))
    dep = (pref - bs[None]); dep = dep / dep.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    add(tau, dep, torch.ones(B, dtype=torch.long, device=dev),
        torch.full((B,), -2, dtype=torch.long, device=dev), ok)

    # ---- first-order wall bounces
    n3 = torch.cat([scene.wn, torch.zeros(scene.Wn, 1, device=dev)], 1)   # (W,3)
    p3 = torch.cat([scene.wp, torch.zeros(scene.Wn, 1, device=dev)], 1)
    d_bs = ((bs[None] - p3) * n3).sum(-1)                    # (W,) signed dist of BS
    img = bs[None] - 2 * d_bs[:, None] * n3                  # (W,3) BS images
    for wi in range(scene.Wn):
        if d_bs[wi] <= 0.5:      # BS behind or on wall -> no reflection
            continue
        iw = img[wi]
        v = ue - iw[None]
        denom = (v * n3[wi][None]).sum(-1)                  # (B,)
        t_hit = ((p3[wi] - iw)[None] * n3[wi][None]).sum(-1) / torch.where(denom.abs() < 1e-8, torch.full_like(denom, 1e-8), denom)
        good = (t_hit > 1e-4) & (t_hit < 1 - 1e-4)
        pref = iw[None] + v * t_hit[:, None]                 # (B,3) reflection point
        # inside wall rectangle?
        loc_t = ((pref[:, :2] - scene.wp[wi][None]) * scene.wt[wi][None]).sum(-1)
        inside = good & (loc_t.abs() <= scene.we[wi]) & (pref[:, 2] >= 0) & (pref[:, 2] <= scene.h[scene.wb[wi]])
        # UE on the outside of the wall
        side = ((ue[:, :2] - scene.wp[wi][None]) * scene.wn[wi][None]).sum(-1) > 0
        inside = inside & side
        if not inside.any():
            continue
        ok = inside.clone()
        idx = inside.nonzero(as_tuple=True)[0]
        blocked1 = scene.seg_blocked(bs[None].expand(len(idx), -1), pref[idx])
        blocked2 = scene.seg_blocked(pref[idx], ue[idx])
        ok[idx] = ~(blocked1 | blocked2)
        tau = v.norm(dim=-1)
        dep = pref - bs[None]; dep = dep / dep.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        add(tau, dep, torch.full((B,), 2, dtype=torch.long, device=dev),
            torch.full((B,), int(scene.wb[wi]), dtype=torch.long, device=dev), ok)

    tau = torch.stack(taus, 1)         # (B,P)
    dep = torch.stack(dirs, 1)         # (B,P,3)
    typ = torch.stack(typs, 1)
    bid = torch.stack(bids, 1)
    ok = torch.stack(oks, 1)
    # keep the max_paths shortest valid paths per UE
    key = torch.where(ok, tau, torch.full_like(tau, 1e9))
    order = key.argsort(1)[:, :max_paths]
    g = lambda x: torch.gather(x, 1, order) if x.dim() == 2 else torch.gather(x, 1, order[..., None].expand(-1, -1, 3))
    return dict(tau=g(tau), dep=g(dep), typ=g(typ), bid=g(bid),
                ok=torch.gather(ok, 1, order))
