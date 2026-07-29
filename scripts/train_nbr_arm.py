"""邻居注意力谱预测器: 每切片学习邻居加权(替代固定IDW²)
leave-self-out在1800训练点上训, 200 val验证是否超IDW²(PAS 0.689/PDP 0.792)"""
import argparse, json
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401

import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree
from scipy import ndimage
from wireless_twin.data.map_loader import load_point_cloud
from wireless_twin.models.raytrace2 import build_heightmap
from score_holdout import reproduce_val_indices
from sweep_mapaware_spectrum_knn import map_features, robust_standardize

ap = argparse.ArgumentParser()
ap.add_argument("--datadir", default="Round1_Map(2)")
ap.add_argument("--device", default="cuda")
ap.add_argument("--K", type=int, default=16)
ap.add_argument("--prefix", default="nbrattn3")
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--eval-batch", type=int, default=100)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--only", choices=["both", "pas", "pdp"], default="both")
ap.add_argument(
    "--feature-set",
    choices=["basic", "geometry", "rich", "pas_content"],
    default="basic",
)
ap.add_argument(
    "--pas-layout",
    choices=["legacy_hvp", "pvh", "phv"],
    default="legacy_hvp",
    help=(
        "BS antenna flattening: legacy (MH,MV,MP), (MP,MV,MH), "
        "or polarization-first (MP,MH,MV)"
    ),
)
ap.add_argument(
    "--neighbor-metric",
    choices=["euclidean", "los"],
    default="euclidean",
)
ap.add_argument("--neighbor-lambda", type=float, default=4.0)
ap.add_argument(
    "--rank-loss",
    type=float,
    default=0.0,
    help="weight for per-slice soft neighbor-ranking supervision",
)
ap.add_argument(
    "--rank-temperature",
    type=float,
    default=0.05,
    help="temperature for oracle neighbor-similarity targets",
)
ap.add_argument(
    "--moment-correction",
    type=float,
    default=0.0,
    help=(
        "fraction of the attention spatial-centroid bias removed by an "
        "exponential moment projection"
    ),
)
ap.add_argument("--moment-iterations", type=int, default=8)
ap.add_argument("--moment-residual-tolerance", type=float, default=0.03)
ap.add_argument(
    "--init-checkpoint",
    default=None,
    help="optional compatible clean-arm checkpoint to fine-tune",
)
ap.add_argument("--split-seed", type=int, default=0)
ap.add_argument(
    "--val-indices",
    default=None,
    help="optional external validation index file instead of split_seed",
)
ap.add_argument("--val-fraction", type=float, default=0.1)
ap.add_argument("--clean-holdout", action="store_true",
                help="train/evaluate using only the train side of split_seed as labeled neighbors")
args = ap.parse_args()
dd = Path(args.datadir)
st = json.loads((dd / "Round1_Setup.json").read_text(encoding="utf-8"))
MH, MV, MP, N, S = st["M_H"], st["M_V"], st["M_P"], st["N"], st["S"]
dev = args.device
tiny = torch.finfo(torch.float32).tiny
pos = np.load(dd / "Round1_Train_Pos.npy").astype(np.float32)
ch = np.load(dd / "Round1_Train_Channel.npy")
if args.val_indices:
    vi = sorted(np.load(args.val_indices).astype(np.int64).tolist())
else:
    vi = sorted(
        reproduce_val_indices(
            len(pos), args.val_fraction, args.split_seed
        )
    )
vs = set(vi)
tri = np.array([i for i in range(len(pos)) if i not in vs]); vai = np.array(vi)
K = args.K

def PAS(x):
    if args.pas_layout in {"pvh", "phv"}:
        spatial = (MV, MH) if args.pas_layout == "pvh" else (MH, MV)
        a = x.reshape(-1, MP, *spatial, N, S)
        return torch.fft.fft2(
            a, dim=(2, 3), norm="ortho"
        ).abs().square().sum(1).reshape(-1, MH*MV, N, S)
    a = x.reshape(-1, MH, MV, MP, N, S)
    return torch.fft.fft2(
        a, dim=(1, 2), norm="ortho"
    ).abs().square().sum(3).reshape(-1, MH*MV, N, S)
def PDP(x):
    return torch.fft.ifft(x, dim=-1, norm="ortho").abs().square()
def nrm(P, dim): return P / P.norm(dim=dim, keepdim=True).clamp_min(1e-30)

pts = load_point_cloud(dd / "Round1_Map.ply")
hm, x0, y0, res = build_heightmap(pts)
gx = np.clip(np.floor((pos[:, 0]-x0)/res).astype(int), 0, hm.shape[0]-1)
gy = np.clip(np.floor((pos[:, 1]-y0)/res).astype(int), 0, hm.shape[1]-1)
indoor = (hm[gx, gy] > 2.0).astype(np.float32)

# 全部train点的谱(分块算, 存GPU)
print("[nbr] 计算谱...", flush=True)
all_pas = (
    torch.zeros(len(pos), MH*MV, N, S, device=dev)
    if args.only in {"both", "pas"}
    else None
)
all_pdp = (
    torch.zeros(len(pos), MH*MV*MP, N, S, device=dev)
    if args.only in {"both", "pdp"}
    else None
)
for c0 in range(0, len(pos), 200):
    cs = slice(c0, c0+200)
    Hc = torch.tensor(ch[cs], dtype=torch.complex64, device=dev)
    if all_pas is not None:
        all_pas[cs] = nrm(PAS(Hc), 1)
    if all_pdp is not None:
        all_pdp[cs] = nrm(PDP(Hc), -1)
    del Hc
torch.cuda.empty_cache()

# 邻居表:
# - default preserves the historical all-2000 leave-self-out protocol.
# - clean-holdout removes the current validation split from both training
#   examples and the labeled neighbor bank, so validation labels cannot leak.
if args.neighbor_metric == "los":
    raw_neighbor_features, _ = map_features(
        pos,
        np.asarray(st["X"], dtype=np.float32),
        hm,
        x0,
        y0,
        res,
        ray_samples=128,
    )
else:
    raw_neighbor_features = None


def metric_neighbors(query_idx, pool_idx, count, exclude_self=False):
    if args.neighbor_metric == "euclidean":
        tree = cKDTree(pos[pool_idx, :2])
        extra = 1 if exclude_self else 0
        distance, local = tree.query(pos[query_idx, :2], k=count + extra)
        indices = pool_idx[local]
    else:
        standardized, _, _ = robust_standardize(
            raw_neighbor_features[pool_idx], raw_neighbor_features
        )
        xy_delta = (
            pos[query_idx, None, :2] - pos[pool_idx][None, :, :2]
        )
        feature_delta = (
            standardized[query_idx, None, :6]
            - standardized[pool_idx][None, :, :6]
        )
        distance2 = np.square(xy_delta).sum(axis=-1)
        distance2 += (
            args.neighbor_lambda ** 2
            * np.square(feature_delta).mean(axis=-1)
        )
        extra = 1 if exclude_self else 0
        local = np.argpartition(
            distance2, kth=count + extra - 1, axis=1
        )[:, : count + extra]
        selected = np.take_along_axis(distance2, local, axis=1)
        order = np.argsort(selected, axis=1)
        local = np.take_along_axis(local, order, axis=1)
        selected = np.take_along_axis(selected, order, axis=1)
        indices = pool_idx[local]
        distance = np.sqrt(np.maximum(selected, 1e-6))
    if exclude_self:
        kept_indices = np.empty((len(query_idx), count), dtype=np.int64)
        kept_distance = np.empty((len(query_idx), count), dtype=np.float32)
        for row, source_idx in enumerate(query_idx):
            keep = indices[row] != source_idx
            kept_indices[row] = indices[row][keep][:count]
            kept_distance[row] = distance[row][keep][:count]
        return kept_distance, kept_indices
    return distance.astype(np.float32), indices


if args.clean_holdout:
    target_train = tri
    tree_idx = tri
    dT, jT = metric_neighbors(
        target_train, tree_idx, K, exclude_self=True
    )
    dV, jV = metric_neighbors(vai, tree_idx, K)
    print("[nbr] clean_holdout split_seed=%d train=%d val=%d" % (
          args.split_seed, len(target_train), len(vai)), flush=True)
else:
    target_train = np.arange(len(pos))
    tree_idx = np.arange(len(pos))
    dT, jT = metric_neighbors(
        target_train, tree_idx, K, exclude_self=True
    )
    dV, jV = dT[vai], jT[vai]    # historical 2000-loo val口径

class SliceAttn(nn.Module):
    """每切片邻居加权: feat->score->softmax; 初始化≈IDW²"""
    def __init__(self, nf=6, content_dim=0):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(nf, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.mlp[4].weight); nn.init.zeros_(self.mlp[4].bias)
        self.content_dim = content_dim
        if content_dim:
            self.content_mlp = nn.Sequential(
                nn.Linear(content_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )
            nn.init.zeros_(self.content_mlp[4].weight)
            nn.init.zeros_(self.content_mlp[4].bias)
        self.idw_w = nn.Parameter(torch.tensor(2.0))    # 初始IDW幂
    def logits(self, feats, logd):
        # feats: (B,K,nslice,nf), logd: (B,K,1)
        if self.content_dim:
            base = self.mlp(feats[..., :-self.content_dim])
            content = self.content_mlp(
                feats[..., -self.content_dim:]
            )
            score = base + content
        else:
            score = self.mlp(feats)
        return score.squeeze(-1) - self.idw_w * logd

    def forward(self, feats, logd):
        return torch.softmax(self.logits(feats, logd), dim=1)


def moment_weights(logits, dn, jn, target_pos):
    """Apply a low-memory spatial moment projection to attention logits.

    Newton multipliers are estimated without autograd. Gradients still flow
    through the final constrained softmax with the multiplier held fixed,
    which is a useful straight-through approximation to implicit
    differentiation of the constrained optimum.
    """
    if args.moment_correction <= 0:
        return torch.softmax(logits, dim=1)
    neighbor_xy = torch.as_tensor(
        pos[jn, :2], dtype=torch.float32, device=dev
    )
    target_xy = torch.as_tensor(
        target_pos[:, :2], dtype=torch.float32, device=dev
    )
    delta = neighbor_xy - target_xy[:, None]
    scale = torch.as_tensor(
        np.median(dn, axis=1), dtype=torch.float32, device=dev
    ).clamp_min(0.3)
    delta = delta / scale[:, None, None]
    with torch.no_grad():
        detached = logits.detach()
        prior = torch.softmax(detached, dim=1)
        prior_mean = torch.einsum("bkq,bkd->bqd", prior, delta)
        target = (1.0 - args.moment_correction) * prior_mean
        multiplier = torch.zeros_like(target)
        for _ in range(args.moment_iterations):
            tilted = detached + torch.einsum(
                "bqd,bkd->bkq", multiplier, delta
            )
            weight = torch.softmax(tilted, dim=1)
            mean = torch.einsum("bkq,bkd->bqd", weight, delta)
            second = torch.einsum(
                "bkq,bki,bkj->bqij", weight, delta, delta
            )
            covariance = second - torch.einsum(
                "bqi,bqj->bqij", mean, mean
            )
            error = mean - target
            a = covariance[..., 0, 0] + 1e-5
            b = covariance[..., 0, 1]
            c = covariance[..., 1, 1] + 1e-5
            determinant = (a * c - b.square()).clamp_min(1e-8)
            step_x = (
                c * error[..., 0] - b * error[..., 1]
            ) / determinant
            step_y = (
                a * error[..., 1] - b * error[..., 0]
            ) / determinant
            step = torch.stack([step_x, step_y], dim=-1)
            step_norm = step.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            step *= torch.clamp(5.0 / step_norm, max=1.0)
            multiplier = (multiplier - step).clamp(-60.0, 60.0)
        tilted = detached + torch.einsum(
            "bqd,bkd->bkq", multiplier, delta
        )
        projected_detached = torch.softmax(tilted, dim=1)
        projected_mean = torch.einsum(
            "bkq,bkd->bqd", projected_detached, delta
        )
        residual = (projected_mean - target).norm(dim=-1)
        invalid = (~torch.isfinite(residual)) | (
            residual > args.moment_residual_tolerance
        )
    projected = torch.softmax(
        logits
        + torch.einsum("bqd,bkd->bkq", multiplier.detach(), delta),
        dim=1,
    )
    prior_with_grad = torch.softmax(logits, dim=1)
    return torch.where(
        invalid[:, None, :], prior_with_grad, projected
    )

def build_feats(dn, jn, target_pos, tgt_indoor, spec, dim):
    """spec: all_pas或all_pdp; 返回 (B,K,nslice,nf), 邻居切片(B,K,nslice,L)"""
    B = len(dn)
    nb = spec[jn]                                    # (B,K,...)
    if dim == 1:   # PAS: 切片沿dim1, nslice=N*S, L=128
        Y = nb.permute(0, 1, 3, 4, 2).reshape(B, K, -1, MH*MV)
    else:          # PDP: nslice=M*N, L=S
        Y = nb.reshape(B, K, -1, S)
    mean = Y.mean(1, keepdim=True)
    agree = F.cosine_similarity(Y, mean, dim=-1)     # (B,K,nslice)
    en = Y.norm(dim=-1)                               # 1(归一过) 无用, 用能量代理: 提前不归一? 简化掉
    d_ = torch.tensor(dn, device=dev)[:, :, None].expand(-1, -1, Y.shape[2])
    ind_nb = torch.tensor(indoor[jn], device=dev)[:, :, None].expand(-1, -1, Y.shape[2])
    ind_t = torch.tensor(tgt_indoor, device=dev)[:, None, None].expand(-1, K, Y.shape[2])
    columns = [
        d_ / 3.0,
        (ind_nb == ind_t).float(),
        agree,
        agree * agree,
        torch.ones_like(d_),
        (d_ < 2.5).float(),
    ]
    if args.feature_set in {"geometry", "rich", "pas_content"}:
        target_xy = torch.as_tensor(
            target_pos[:, :2], dtype=torch.float32, device=dev
        )
        neighbor_xy = torch.as_tensor(
            pos[jn, :2], dtype=torch.float32, device=dev
        )
        delta = neighbor_xy - target_xy[:, None, :]
        bs_xy = torch.tensor(
            st["X"][:2], dtype=torch.float32, device=dev
        )
        radial = target_xy - bs_xy
        radius = radial.norm(dim=-1).clamp_min(1e-3)
        radial_unit = radial / radius[:, None]
        tangent_unit = torch.stack(
            [-radial_unit[:, 1], radial_unit[:, 0]], dim=-1
        )
        radial_delta = (delta * radial_unit[:, None, :]).sum(dim=-1)
        tangent_delta = (delta * tangent_unit[:, None, :]).sum(dim=-1)

        def expand(value):
            return value[:, :, None].expand(-1, -1, Y.shape[2])

        angle = torch.atan2(radial[:, 1], radial[:, 0])
        columns.extend(
            [
                expand(delta[..., 0] / 5.0),
                expand(delta[..., 1] / 5.0),
                expand(radial_delta / 5.0),
                expand(tangent_delta / 5.0),
                (radius / 200.0)[:, None, None].expand(
                    -1, K, Y.shape[2]
                ),
                angle.sin()[:, None, None].expand(-1, K, Y.shape[2]),
                angle.cos()[:, None, None].expand(-1, K, Y.shape[2]),
                ind_nb,
                ind_t,
            ]
        )
    if args.feature_set == "rich":
        # Candidate-to-candidate agreement exposes local spectral modes.  It
        # remains usable at test time because every candidate is labeled.
        gram = torch.einsum("bksl,bjsl->bksj", Y, Y)
        diagonal = torch.eye(K, dtype=torch.bool, device=dev)[
            None, :, None, :
        ]
        other = gram.masked_fill(diagonal, 0.0)
        denominator = max(K - 1, 1)
        mean_other = other.sum(dim=-1) / denominator
        squared_other = other.square().sum(dim=-1) / denominator
        std_other = (
            squared_other - mean_other.square()
        ).clamp_min(0.0).sqrt()
        max_other = gram.masked_fill(diagonal, -1.0).amax(dim=-1)
        rank = torch.linspace(
            0.0, 1.0, K, device=dev
        )[None, :, None].expand(B, -1, Y.shape[2])
        nearest_similarity = gram[..., 0]

        consensus_columns = []
        distance_t = torch.as_tensor(
            dn, dtype=torch.float32, device=dev
        ).clamp_min(0.3)
        for power in (1.0, 2.0, 4.0):
            weight = distance_t.pow(-power)
            weight = weight / weight.sum(dim=1, keepdim=True)
            consensus = torch.einsum("bk,bksl->bsl", weight, Y)
            consensus_columns.append(
                F.cosine_similarity(
                    Y, consensus[:, None], dim=-1
                )
            )
        for count in (2, 4, 8):
            keep = min(count, K)
            consensus = Y[:, :keep].mean(dim=1)
            consensus_columns.append(
                F.cosine_similarity(
                    Y, consensus[:, None], dim=-1
                )
            )

        probability = Y / Y.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        entropy = -(
            probability
            * probability.clamp_min(1e-12).log()
        ).sum(dim=-1) / np.log(Y.shape[-1])
        columns.extend(
            [
                rank,
                nearest_similarity,
                mean_other,
                max_other,
                std_other,
                *consensus_columns,
                Y.amax(dim=-1),
                entropy,
            ]
        )
    if args.feature_set == "pas_content":
        if dim != 1:
            raise ValueError("pas_content features are only valid for PAS")
        grid = Y.reshape(B, K, Y.shape[2], MH, MV)
        probability = grid / grid.sum(
            dim=(-2, -1), keepdim=True
        ).clamp_min(1e-12)
        marginal_h = probability.sum(dim=-1)
        marginal_v = probability.sum(dim=-2)
        h_angle = (
            torch.arange(MH, dtype=torch.float32, device=dev)
            * (2.0 * np.pi / MH)
        )
        v_angle = (
            torch.arange(MV, dtype=torch.float32, device=dev)
            * (2.0 * np.pi / MV)
        )
        h_cos = (marginal_h * h_angle.cos()).sum(dim=-1)
        h_sin = (marginal_h * h_angle.sin()).sum(dim=-1)
        v_cos = (marginal_v * v_angle.cos()).sum(dim=-1)
        v_sin = (marginal_v * v_angle.sin()).sum(dim=-1)
        h_concentration = torch.sqrt(
            h_cos.square() + h_sin.square()
        )
        v_concentration = torch.sqrt(
            v_cos.square() + v_sin.square()
        )

        distance_t = torch.as_tensor(
            dn, dtype=torch.float32, device=dev
        ).clamp_min(0.3)
        anchor_weight = distance_t.pow(-2)
        anchor_weight /= anchor_weight.sum(dim=1, keepdim=True)

        def circular_relative(
            cosine: torch.Tensor, sine: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            anchor_cosine = (
                anchor_weight[:, :, None] * cosine
            ).sum(dim=1)
            anchor_sine = (
                anchor_weight[:, :, None] * sine
            ).sum(dim=1)
            anchor_norm = torch.sqrt(
                anchor_cosine.square() + anchor_sine.square()
            ).clamp_min(1e-6)
            anchor_cosine = anchor_cosine / anchor_norm
            anchor_sine = anchor_sine / anchor_norm
            dot = (
                cosine * anchor_cosine[:, None]
                + sine * anchor_sine[:, None]
            )
            cross = (
                sine * anchor_cosine[:, None]
                - cosine * anchor_sine[:, None]
            )
            return dot, cross

        h_dot, h_cross = circular_relative(h_cos, h_sin)
        v_dot, v_cross = circular_relative(v_cos, v_sin)
        peak = Y.argmax(dim=-1)
        peak_h = torch.div(peak, MV, rounding_mode="floor")
        peak_v = torch.remainder(peak, MV)
        peak_h_angle = peak_h.float() * (2.0 * np.pi / MH)
        peak_v_angle = peak_v.float() * (2.0 * np.pi / MV)
        distribution = Y / Y.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)
        entropy = -(
            distribution
            * distribution.clamp_min(1e-12).log()
        ).sum(dim=-1) / np.log(Y.shape[-1])

        slice_index = torch.arange(Y.shape[2], device=dev)
        ue_index = torch.div(slice_index, S, rounding_mode="floor")
        subcarrier = torch.remainder(slice_index, S).float()

        def shared_slice(value: torch.Tensor) -> torch.Tensor:
            return value[None, None].expand(B, K, -1)

        ue_one_hot = F.one_hot(
            ue_index, num_classes=N
        ).float()
        columns.extend(
            [
                h_cos,
                h_sin,
                h_concentration,
                v_cos,
                v_sin,
                v_concentration,
                h_dot,
                h_cross,
                v_dot,
                v_cross,
                peak_h_angle.cos(),
                peak_h_angle.sin(),
                peak_v_angle.cos(),
                peak_v_angle.sin(),
                Y.amax(dim=-1),
                entropy,
                shared_slice(
                    torch.sin(2.0 * np.pi * subcarrier / S)
                ),
                shared_slice(
                    torch.cos(2.0 * np.pi * subcarrier / S)
                ),
                shared_slice(
                    2.0 * subcarrier / max(S - 1, 1) - 1.0
                ),
                *[
                    shared_slice(ue_one_hot[:, index])
                    for index in range(N)
                ],
            ]
        )
    feats = torch.stack(columns, -1)
    logd = torch.log(torch.tensor(dn, device=dev).clamp_min(0.3))[:, :, None]
    return feats, Y, logd

def run(dim, tag, gt_spec, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    feature_dim = {
        "basic": 6,
        "geometry": 15,
        "rich": 28,
        "pas_content": 38,
    }[args.feature_set]
    model = (
        SliceAttn(15, content_dim=23).to(dev)
        if args.feature_set == "pas_content"
        else SliceAttn(feature_dim).to(dev)
    )
    if args.init_checkpoint:
        initial = torch.load(
            args.init_checkpoint, map_location=dev, weights_only=False
        )
        state = initial.get("model_state", initial)
        incompatible = model.load_state_dict(
            state, strict=args.feature_set != "pas_content"
        )
        if args.feature_set == "pas_content":
            unexpected = list(incompatible.unexpected_keys)
            missing = [
                key
                for key in incompatible.missing_keys
                if not key.startswith("content_mlp.")
            ]
            if unexpected or missing:
                raise ValueError(
                    f"incompatible content initialization: "
                    f"missing={missing} unexpected={unexpected}"
                )
        print(
            f"[{tag}] initialized from {args.init_checkpoint}",
            flush=True,
        )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # gt切片
    if dim == 1:
        gtY = lambda idx: gt_spec[idx].permute(0, 2, 3, 1).reshape(len(idx), -1, MH*MV)
    else:
        gtY = lambda idx: gt_spec[idx].reshape(len(idx), -1, S)
    best = 0
    for ep in range(args.epochs):
        perm = np.random.permutation(len(target_train))
        model.train()
        for i in range(0, len(target_train), args.batch):
            b = perm[i:i+args.batch]
            tgt_idx = target_train[b]
            feats, Y, logd = build_feats(
                dT[b], jT[b], pos[tgt_idx], indoor[tgt_idx], gt_spec, dim
            )
            logits = model.logits(feats, logd)
            w_ = moment_weights(
                logits, dT[b], jT[b], pos[tgt_idx]
            )
            pred = (w_[..., None] * Y).sum(1)
            gt = gtY(tgt_idx)
            loss = 1 - F.cosine_similarity(pred, gt, dim=-1).mean()
            if args.rank_loss > 0:
                similarity = F.cosine_similarity(
                    Y, gt[:, None], dim=-1
                )
                target_weight = torch.softmax(
                    similarity / args.rank_temperature, dim=1
                )
                ranking = -(
                    target_weight
                    * torch.log_softmax(logits, dim=1)
                ).sum(dim=1).mean()
                loss = loss + args.rank_loss * ranking
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            cs = []
            for i in range(0, len(vai), args.eval_batch):
                b = np.arange(i, min(i+args.eval_batch, len(vai)))
                feats, Y, logd = build_feats(
                    dV[b], jV[b], pos[vai[b]], indoor[vai[b]], gt_spec, dim
                )
                logits = model.logits(feats, logd)
                w_ = moment_weights(
                    logits, dV[b], jV[b], pos[vai[b]]
                )
                pred = (w_[..., None]*Y).sum(1)
                gt = gtY(vai[b])
                cs.append(F.cosine_similarity(pred, gt, dim=-1).mean().item())
            c = float(np.mean(cs))
        if c > best:
            best = c
            out_path = Path("checkpoints") / f"{args.prefix}_{tag}.pt"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if args.clean_holdout:
                torch.save({
                    "model_state": model.state_dict(),
                    "meta": {
                        "K": K,
                        "tag": tag,
                        "dim": dim,
                        "seed": seed,
                        "split_seed": args.split_seed,
                        "val_fraction": args.val_fraction,
                        "clean_holdout": True,
                        "class": "AttnWide",
                        "best_cos": best,
                        "pas_layout": args.pas_layout,
                        "feature_set": args.feature_set,
                        "feature_dim": feature_dim,
                        "neighbor_metric": args.neighbor_metric,
                        "neighbor_lambda": args.neighbor_lambda,
                        "rank_loss": args.rank_loss,
                        "rank_temperature": args.rank_temperature,
                        "moment_correction": args.moment_correction,
                        "moment_iterations": args.moment_iterations,
                        "moment_residual_tolerance": (
                            args.moment_residual_tolerance
                        ),
                        "init_checkpoint": args.init_checkpoint,
                        "learning_rate": args.lr,
                    },
                }, out_path)
            else:
                torch.save(model.state_dict(), out_path)
        if ep % 5 == 4:
            print("[%s] ep%d val cos=%.4f (best=%.4f)" % (tag, ep+1, c, best), flush=True)
    return best

for sd in args.seeds:
    if K == 16 and args.prefix == "nbrattn3":
        pas_tag = "pas_k16f_s%d" % sd
        pdp_tag = "pdp_k16f_s%d" % sd
    else:
        pas_tag = "pas_k%ds%d" % (K, sd)
        pdp_tag = "pdp_k%ds%d" % (K, sd)
    bp = run(1, pas_tag, all_pas, sd) if args.only in {"both", "pas"} else float("nan")
    bd = run(-1, pdp_tag, all_pdp, sd) if args.only in {"both", "pdp"} else float("nan")
    print("K%d臂 seed%d: PAS=%.4f PDP=%.4f" % (K, sd, bp, bd), flush=True)
print("NBRATTN_DONE", flush=True)
