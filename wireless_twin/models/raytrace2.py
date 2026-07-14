"""Image-method ray tracer v2: heightmap occlusion + facade-segment reflections.

v1 failed because connected building clusters are non-convex: an oriented box
around a whole block swallows the streets (70% of UEs ended up "inside").
v2 splits the two roles:
  * occlusion  -> exact 2.5D heightmap ray-marching (no geometric simplification)
  * reflections -> straight facade segments extracted from cluster boundaries
"""
from __future__ import annotations

import numpy as np
import torch


# --------------------------------------------------------------------------
def build_heightmap(pts: np.ndarray, res: float = 1.0):
    x0, y0 = pts[:, 0].min(), pts[:, 1].min()
    W = int((pts[:, 0].max() - x0) / res) + 2
    H = int((pts[:, 1].max() - y0) / res) + 2
    hm = np.zeros((W, H), dtype=np.float32)
    bld = pts[pts[:, 2] > 0.5]
    gx = ((bld[:, 0] - x0) / res).astype(int)
    gy = ((bld[:, 1] - y0) / res).astype(int)
    np.maximum.at(hm, (gx, gy), bld[:, 2])
    return hm, float(x0), float(y0), res


def extract_facades(hm: np.ndarray, x0: float, y0: float, res: float,
                    min_len: float = 6.0):
    """Extract straight vertical facade segments from the occupancy boundary.

    For each connected cluster: rotate its mask to the dominant angle, walk the
    rotated-grid boundary rows/cols and emit maximal straight runs.
    Returns list of dicts: p (2,) midpoint, t (2,) tangent, e half-length,
    n (2,) outward normal, h height.
    """
    from scipy import ndimage
    occ = hm > 0.5
    lab, nb = ndimage.label(occ)
    segs = []
    for k in range(1, nb + 1):
        m = lab == k
        if m.sum() < 20:
            continue
        ys, xs = np.nonzero(m)
        pts = np.stack([ys, xs], 1).astype(np.float64)      # grid coords (ix,iy)
        h_k = float(np.percentile(hm[m], 90))
        c = pts.mean(0)
        xy = pts - c
        C = np.cov(xy.T); ev, V = np.linalg.eigh(C)
        th = float(np.arctan2(V[1, 1], V[0, 1]))            # dominant axis angle
        R = np.array([[np.cos(-th), -np.sin(-th)], [np.sin(-th), np.cos(-th)]])
        q = xy @ R.T
        # rasterise rotated mask at res
        q0 = q.min(0) - 1.0
        qi = np.round(q - q0).astype(int)
        Wr, Hr = qi[:, 0].max() + 2, qi[:, 1].max() + 2
        mr = np.zeros((Wr, Hr), bool); mr[qi[:, 0], qi[:, 1]] = True
        mr = ndimage.binary_closing(mr, np.ones((2, 2)))
        Rinv = R.T                                          # rotated -> cluster frame

        def emit(run_axis, line_idx, a, b, outward):
            # run along `run_axis` at fixed line_idx, cells [a,b) -> segment
            L = (b - a) * res
            if L < min_len:
                return
            if run_axis == 0:   # run over rows (i), wall along +i, at col j=line_idx
                p_rot = np.array([(a + b) / 2 - 0.5, line_idx + (0.5 if outward > 0 else -0.5)])
                t_rot = np.array([1.0, 0.0]); n_rot = np.array([0.0, float(outward)])
            else:
                p_rot = np.array([line_idx + (0.5 if outward > 0 else -0.5), (a + b) / 2 - 0.5])
                t_rot = np.array([0.0, 1.0]); n_rot = np.array([float(outward), 0.0])
            p = (p_rot + q0) @ Rinv.T + c
            t = t_rot @ Rinv.T; n = n_rot @ Rinv.T
            # grid(i=x-idx, j=y-idx) -> world
            pw = np.array([y0 + p[1] * res, 0.0]);  # placeholder replaced below
            segs.append(dict(
                p=np.array([x0 + p[0] * res, y0 + p[1] * res]),
                t=np.array([t[0], t[1]]) / np.linalg.norm(t),
                n=np.array([n[0], n[1]]) / np.linalg.norm(n),
                e=L / 2, h=h_k))

        # vertical walls: for each column j, runs of "occupied with free at j+1 / j-1"
        for axis in (0, 1):
            M = mr if axis == 0 else mr.T
            for j in range(M.shape[1]):
                col = M[:, j]
                nxt = np.zeros_like(col); prv = np.zeros_like(col)
                if j + 1 < M.shape[1]: nxt = ~M[:, j + 1]
                else: nxt = np.ones_like(col)
                if j - 1 >= 0: prv = ~M[:, j - 1]
                else: prv = np.ones_like(col)
                for outward, freem in ((1, col & nxt), (-1, col & prv)):
                    i = 0
                    while i < len(freem):
                        if freem[i]:
                            a = i
                            while i < len(freem) and freem[i]:
                                i += 1
                            if axis == 0:
                                emit(0, j, a, i, outward)
                            else:
                                emit(1, j, a, i, outward)
                        else:
                            i += 1
    return segs


# --------------------------------------------------------------------------
class HeightmapScene:
    def __init__(self, hm: np.ndarray, x0: float, y0: float, res: float,
                 facades: list, device="cuda", n_samples: int = 256):
        self.dev = device
        self.hm = torch.tensor(hm, device=device)
        self.x0, self.y0, self.res = x0, y0, res
        self.T = n_samples
        self.fp = torch.tensor(np.stack([f["p"] for f in facades]), dtype=torch.float32, device=device)
        self.ft = torch.tensor(np.stack([f["t"] for f in facades]), dtype=torch.float32, device=device)
        self.fn = torch.tensor(np.stack([f["n"] for f in facades]), dtype=torch.float32, device=device)
        self.fe = torch.tensor([f["e"] for f in facades], dtype=torch.float32, device=device)
        self.fh = torch.tensor([f["h"] for f in facades], dtype=torch.float32, device=device)
        self.F = len(facades)

    def blocked(self, a: torch.Tensor, b: torch.Tensor, margin: float = 0.5,
                trim: float = 1.5) -> torch.Tensor:
        """(B,3),(B,3) -> (B,) bool: segment intersects the heightmap volume.
        `trim` metres are excluded at both endpoints (so reflection points on a
        facade do not self-block)."""
        B = a.shape[0]
        L = (b - a).norm(dim=-1).clamp_min(1e-6)
        t0 = (trim / L).clamp(max=0.45)
        ts = torch.linspace(0, 1, self.T, device=self.dev)[None] * (1 - 2 * t0[:, None]) + t0[:, None]
        p = a[:, None] + (b - a)[:, None] * ts[..., None]     # (B,T,3)
        gx = ((p[..., 0] - self.x0) / self.res).clamp(0, self.hm.shape[0] - 1)
        gy = ((p[..., 1] - self.y0) / self.res).clamp(0, self.hm.shape[1] - 1)
        hv = self.hm[gx.long(), gy.long()]                    # (B,T)
        return (hv > p[..., 2] + margin).any(1)


def trace_paths2(scene: HeightmapScene, bs: torch.Tensor, ue: torch.Tensor,
                 max_paths: int = 32, chunk: int = 64):
    """LoS + ground + first-order facade reflections. Returns padded tensors."""
    dev = ue.device
    B = ue.shape[0]
    taus, deps, arrs, typs, oks = [], [], [], [], []

    def add(tau, dep, arr, typ, ok):
        taus.append(tau); deps.append(dep); arrs.append(arr); typs.append(typ); oks.append(ok)

    # LoS
    v = ue - bs[None]; tau = v.norm(dim=-1)
    dep = v / tau[:, None]
    ok = ~scene.blocked(bs[None].expand(B, -1), ue)
    add(tau, dep, -dep, torch.zeros(B, dtype=torch.long, device=dev), ok)

    # ground bounce
    bsg = torch.tensor([bs[0], bs[1], -bs[2]], device=dev)
    v = ue - bsg[None]; tau = v.norm(dim=-1)
    th = bs[2] / (bs[2] + ue[:, 2]).clamp_min(1e-6)
    pref = bsg[None] + v * th[:, None]
    ok = (~scene.blocked(bs[None].expand(B, -1), pref)) & (~scene.blocked(pref, ue))
    dep = pref - bs[None]; dep = dep / dep.norm(dim=-1, keepdim=True)
    arr = ue - pref; arr = -(arr / arr.norm(dim=-1, keepdim=True))
    add(tau, dep, arr, torch.ones(B, dtype=torch.long, device=dev), ok)

    # facade reflections (chunk over facades)
    n3 = torch.cat([scene.fn, torch.zeros(scene.F, 1, device=dev)], 1)
    p3 = torch.cat([scene.fp, torch.zeros(scene.F, 1, device=dev)], 1)
    d_bs = ((bs[None] - p3) * n3).sum(-1)                    # (F,)
    img = bs[None] - 2 * d_bs[:, None] * n3                  # (F,3)
    valid_f = (d_bs > 0.5).nonzero(as_tuple=True)[0]
    for c0 in range(0, len(valid_f), chunk):
        fi = valid_f[c0:c0 + chunk]; Fc = len(fi)
        iw = img[fi]                                          # (Fc,3)
        v = ue[None] - iw[:, None]                            # (Fc,B,3)
        den = (v * n3[fi][:, None]).sum(-1)
        den = torch.where(den.abs() < 1e-8, torch.full_like(den, 1e-8), den)
        th = (((p3[fi] - iw)[:, None] * n3[fi][:, None]).sum(-1)) / den   # (Fc,B)
        pref = iw[:, None] + v * th[..., None]                # (Fc,B,3)
        loc = ((pref[..., :2] - scene.fp[fi][:, None]) * scene.ft[fi][:, None]).sum(-1)
        side = ((ue[None, :, :2] - scene.fp[fi][:, None]) * scene.fn[fi][:, None]).sum(-1) > 0
        inside = (th > 1e-4) & (th < 1 - 1e-4) & (loc.abs() <= scene.fe[fi][:, None]) \
                 & (pref[..., 2] >= 0) & (pref[..., 2] <= scene.fh[fi][:, None]) & side
        if not inside.any():
            continue
        tau = v.norm(dim=-1)                                  # (Fc,B)
        ok = torch.zeros_like(inside)
        idx = inside.nonzero(as_tuple=False)
        aseg = bs[None].expand(len(idx), -1)
        prf = pref[idx[:, 0], idx[:, 1]]
        ueseg = ue[idx[:, 1]]
        blk = scene.blocked(aseg, prf) | scene.blocked(prf, ueseg)
        ok[idx[:, 0], idx[:, 1]] = ~blk
        dep = pref - bs[None, None]; dep = dep / dep.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        arr = ue[None] - pref; arr = -(arr / arr.norm(dim=-1, keepdim=True).clamp_min(1e-6))
        for j in range(Fc):
            if not ok[j].any():
                continue
            add(tau[j], dep[j], arr[j],
                torch.full((B,), 2, dtype=torch.long, device=dev), ok[j])

    tau = torch.stack(taus, 1); dep = torch.stack(deps, 1)
    arr = torch.stack(arrs, 1); typ = torch.stack(typs, 1); ok = torch.stack(oks, 1)
    key = torch.where(ok, tau, torch.full_like(tau, 1e9))
    order = key.argsort(1)[:, :max_paths]
    g2 = lambda x: torch.gather(x, 1, order)
    g3 = lambda x: torch.gather(x, 1, order[..., None].expand(-1, -1, 3))
    return dict(tau=g2(tau), dep=g3(dep), arr=g3(arr), typ=g2(typ), ok=g2(ok))


# --------------------------------------------------------------------------
# v3: transmission-aware tracing (no binary occlusion; wall-crossing features)
# --------------------------------------------------------------------------
def _cross_feats(scene, a, b, margin=0.5, trim=1.5):
    """(B,3)->(B,) wall-crossing count and indoor length along segments."""
    B = a.shape[0]
    L = (b - a).norm(dim=-1).clamp_min(1e-6)
    t0 = (trim / L).clamp(max=0.45)
    ts = torch.linspace(0, 1, scene.T, device=scene.dev)[None] * (1 - 2 * t0[:, None]) + t0[:, None]
    p = a[:, None] + (b - a)[:, None] * ts[..., None]
    gx = ((p[..., 0] - scene.x0) / scene.res).clamp(0, scene.hm.shape[0] - 1)
    gy = ((p[..., 1] - scene.y0) / scene.res).clamp(0, scene.hm.shape[1] - 1)
    occ = scene.hm[gx.long(), gy.long()] > p[..., 2] + margin          # (B,T)
    enter = (occ[:, 1:] & ~occ[:, :-1]).sum(1).float() + occ[:, 0].float()
    lin = occ.float().mean(1) * L
    return enter, lin


def trace_paths3(scene, bs, ue, max_paths=24, chunk=64):
    """Transmission-aware: all paths kept, blocked -> (n_cross, L_indoor) feats."""
    dev = ue.device
    B = ue.shape[0]
    out = {k: [] for k in ("tau", "dep", "arr", "typ", "ncx", "lin")}

    def add(tau, dep, arr, typ, ncx, lin):
        out["tau"].append(tau); out["dep"].append(dep); out["arr"].append(arr)
        out["typ"].append(typ); out["ncx"].append(ncx); out["lin"].append(lin)

    v = ue - bs[None]; tau = v.norm(dim=-1); dep = v / tau[:, None]
    ncx, lin = _cross_feats(scene, bs[None].expand(B, -1), ue)
    add(tau, dep, -dep, torch.zeros(B, dtype=torch.long, device=dev), ncx, lin)

    bsg = torch.tensor([bs[0], bs[1], -bs[2]], device=dev)
    v = ue - bsg[None]; tau = v.norm(dim=-1)
    th = bs[2] / (bs[2] + ue[:, 2]).clamp_min(1e-6)
    pref = bsg[None] + v * th[:, None]
    n1, l1 = _cross_feats(scene, bs[None].expand(B, -1), pref)
    n2, l2 = _cross_feats(scene, pref, ue)
    dep = pref - bs[None]; dep = dep / dep.norm(dim=-1, keepdim=True)
    arr = ue - pref; arr = -(arr / arr.norm(dim=-1, keepdim=True))
    add(tau, dep, arr, torch.ones(B, dtype=torch.long, device=dev), n1 + n2, l1 + l2)

    n3 = torch.cat([scene.fn, torch.zeros(scene.F, 1, device=dev)], 1)
    p3 = torch.cat([scene.fp, torch.zeros(scene.F, 1, device=dev)], 1)
    d_bs = ((bs[None] - p3) * n3).sum(-1)
    img = bs[None] - 2 * d_bs[:, None] * n3
    valid_f = (d_bs > 0.5).nonzero(as_tuple=True)[0]
    for c0 in range(0, len(valid_f), chunk):
        fi = valid_f[c0:c0 + chunk]
        iw = img[fi]
        v = ue[None] - iw[:, None]
        den = (v * n3[fi][:, None]).sum(-1)
        den = torch.where(den.abs() < 1e-8, torch.full_like(den, 1e-8), den)
        th = (((p3[fi] - iw)[:, None] * n3[fi][:, None]).sum(-1)) / den
        pref = iw[:, None] + v * th[..., None]
        loc = ((pref[..., :2] - scene.fp[fi][:, None]) * scene.ft[fi][:, None]).sum(-1)
        side = ((ue[None, :, :2] - scene.fp[fi][:, None]) * scene.fn[fi][:, None]).sum(-1) > 0
        okg = (th > 1e-4) & (th < 1 - 1e-4) & (loc.abs() <= scene.fe[fi][:, None]) \
              & (pref[..., 2] >= 0.2) & (pref[..., 2] <= scene.fh[fi][:, None]) & side
        if not okg.any():
            continue
        tau = v.norm(dim=-1)
        idx = okg.nonzero(as_tuple=False)
        prf = pref[idx[:, 0], idx[:, 1]]
        n1 = torch.zeros_like(tau); l1 = torch.zeros_like(tau)
        n2 = torch.zeros_like(tau); l2 = torch.zeros_like(tau)
        a1, b1 = _cross_feats(scene, bs[None].expand(len(idx), -1), prf)
        a2, b2 = _cross_feats(scene, prf, ue[idx[:, 1]])
        n1[idx[:, 0], idx[:, 1]] = a1; l1[idx[:, 0], idx[:, 1]] = b1
        n2[idx[:, 0], idx[:, 1]] = a2; l2[idx[:, 0], idx[:, 1]] = b2
        dep = pref - bs[None, None]; dep = dep / dep.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        arr = ue[None] - pref; arr = -(arr / arr.norm(dim=-1, keepdim=True).clamp_min(1e-6))
        for j in range(len(fi)):
            if not okg[j].any():
                continue
            tt = torch.where(okg[j], tau[j], torch.full_like(tau[j], 1e9))
            add(tt, dep[j], arr[j], torch.full((B,), 2, dtype=torch.long, device=dev),
                n1[j] + n2[j], l1[j] + l2[j])

    tau = torch.stack(out["tau"], 1)
    dep = torch.stack(out["dep"], 1); arr = torch.stack(out["arr"], 1)
    typ = torch.stack(out["typ"], 1)
    ncx = torch.stack(out["ncx"], 1); lin = torch.stack(out["lin"], 1)
    score = tau + 60.0 * ncx           # rank: delay + heavy penalty per wall
    order = score.argsort(1)[:, :max_paths]
    g2 = lambda x: torch.gather(x, 1, order)
    g3 = lambda x: torch.gather(x, 1, order[..., None].expand(-1, -1, 3))
    return dict(tau=g2(tau), dep=g3(dep), arr=g3(arr), typ=g2(typ),
                ncx=g2(ncx), lin=g2(lin))


def trace_paths4(scene, bs, ue, max_paths=32, max_diff=24, chunk=64):
    """v4: trace_paths3 + 垂直角点绕射路径(角点=立面段端点, 街面高+屋顶高两层)。
    返回额外字段 dang: 水平绕射角(非绕射径=0)。"""
    base = trace_paths3(scene, bs, ue, max_paths=max_paths, chunk=chunk)
    dev = ue.device
    B = ue.shape[0]
    # 角点集合: 每立面段两端 x 两高度
    ends = torch.cat([scene.fp + scene.fe[:, None]*scene.ft,
                      scene.fp - scene.fe[:, None]*scene.ft], 0)      # (2F,2)
    hh = torch.cat([scene.fh, scene.fh], 0)                            # (2F,)
    czs = [torch.full_like(hh, 2.0), hh - 0.5]                         # 街面高/屋顶高
    taus, deps, arrs, ncxs, lins, dangs = [], [], [], [], [], []
    for cz in czs:
        C3 = torch.cat([ends, cz[:, None]], 1)                         # (2F,3)
        # 分块处理角点
        for c0 in range(0, len(C3), 256):
            cc = C3[c0:c0+256]                                          # (Cc,3)
            Cc = len(cc)
            d1 = (cc - bs[None]).norm(dim=-1)                           # (Cc,)
            v2 = ue[None, :, :] - cc[:, None, :]                        # (Cc,B,3)
            d2 = v2.norm(dim=-1)                                        # (Cc,B)
            tau = d1[:, None] + d2
            # 绕射角: BS->c 与 c->UE 的水平夹角偏离直线的程度
            u1 = (cc[:, :2] - bs[None, :2]); u1 = u1/u1.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            u2 = v2[..., :2]/v2[..., :2].norm(dim=-1, keepdim=True).clamp_min(1e-6)
            cosb = (u1[:, None, :]*u2).sum(-1).clamp(-1, 1)
            dang = torch.acos(cosb)                                     # (Cc,B) 0=直线
            # 段crossings
            n1, l1 = _cross_feats(scene, bs[None].expand(Cc, -1), cc)   # (Cc,)
            # 第二段: 每(角点,UE)对 -> 展平算
            a2 = cc[:, None, :].expand(Cc, B, 3).reshape(-1, 3)
            b2 = ue[None, :, :].expand(Cc, B, 3).reshape(-1, 3)
            n2 = torch.zeros(Cc*B, device=dev); l2 = torch.zeros(Cc*B, device=dev)
            for s0 in range(0, Cc*B, 8192):
                ss = slice(s0, min(s0+8192, Cc*B))
                n2[ss], l2[ss] = _cross_feats(scene, a2[ss], b2[ss])
            n2 = n2.reshape(Cc, B); l2 = l2.reshape(Cc, B)
            taus.append(tau.T); dangs.append(dang.T)                    # (B,Cc)
            dep = (cc[:, None, :] - bs[None, None, :]).expand(Cc, B, 3)
            dep = dep/dep.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            deps.append(dep.transpose(0, 1))                            # (B,Cc,3)
            arr = -v2/v2.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            arrs.append(arr.transpose(0, 1))
            ncxs.append((n1[:, None]+n2).T); lins.append((l1[:, None]+l2).T)
    tau = torch.cat(taus, 1); dang = torch.cat(dangs, 1)
    dep = torch.cat(deps, 1); arr = torch.cat(arrs, 1)
    ncx = torch.cat(ncxs, 1); lin = torch.cat(lins, 1)
    # 排序: 时延 + 穿墙惩罚 + 绕射角惩罚
    score = tau + 60.0*ncx + 80.0*dang
    order = score.argsort(1)[:, :max_diff]
    g2 = lambda x: torch.gather(x, 1, order)
    g3 = lambda x: torch.gather(x, 1, order[..., None].expand(-1, -1, 3))
    P0 = base["tau"].shape[1]
    out = dict(
        tau=torch.cat([base["tau"], g2(tau)], 1),
        dep=torch.cat([base["dep"], g3(dep)], 1),
        arr=torch.cat([base["arr"], g3(arr)], 1),
        typ=torch.cat([base["typ"], torch.full((B, max_diff), 3, dtype=torch.long, device=dev)], 1),
        ncx=torch.cat([base["ncx"], g2(ncx)], 1),
        lin=torch.cat([base["lin"], g2(lin)], 1),
        dang=torch.cat([torch.zeros(B, P0, device=dev), g2(dang)], 1))
    return out
