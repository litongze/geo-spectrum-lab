"""Point-cloud preprocessing and Set-Transformer scene encoder.

The competition map is fixed within one round.  We therefore split scene
handling into two stages:

1. ``preprocess_point_cloud`` performs deterministic, non-learned geometry
   processing once (subsample -> FPS -> local neighbourhood descriptors).
2. ``SceneEncoder`` is a trainable Set Transformer operating on only a few
   hundred cached geometry tokens.

This keeps the map in the model while avoiding an expensive 16K-point forward
for every receiver position.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


SCENE_FEATURE_DIM = 15


@dataclass(frozen=True)
class ScenePreprocessConfig:
    """Configuration for deterministic point-cloud tokenisation."""

    n_input_points: int = 16_384
    n_scene_tokens: int = 512
    knn: int = 16
    seed: int = 0


def _farthest_point_sample(points: np.ndarray, n_samples: int) -> np.ndarray:
    """Deterministic FPS indices for a moderate point set.

    Complexity is O(N * n_samples), which is practical after the initial
    16K-point cap and is paid only once per round.
    """

    n = len(points)
    if n_samples >= n:
        return np.arange(n, dtype=np.int64)

    selected = np.empty(n_samples, dtype=np.int64)
    centroid = points.mean(axis=0, keepdims=True)
    selected[0] = int(np.argmax(np.sum((points - centroid) ** 2, axis=1)))
    min_dist = np.full(n, np.inf, dtype=np.float64)

    for i in range(1, n_samples):
        p = points[selected[i - 1]]
        dist = np.sum((points - p) ** 2, axis=1)
        min_dist = np.minimum(min_dist, dist)
        selected[i] = int(np.argmax(min_dist))
    return selected


def preprocess_point_cloud(
    points: np.ndarray,
    pos_mean: Sequence[float],
    pos_std: Sequence[float],
    bs_position: Sequence[float],
    config: Optional[ScenePreprocessConfig] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a raw ``(K,3)`` map into cached geometry tokens.

    Returns
    -------
    features:
        ``(T, 15)`` float32 descriptors.  They contain the sampled centre,
        centre relative to the BS, local mean/std offsets, local radius,
        density and height/range cues.
    bs_position_norm:
        BS coordinate in the same standardised coordinate system as model
        inputs.
    """

    cfg = config or ScenePreprocessConfig()
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"point cloud must be (K,3), got {pts.shape}")
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) == 0:
        raise ValueError("point cloud contains no finite xyz vertices")

    mean = np.asarray(pos_mean, dtype=np.float32)
    std = np.asarray(pos_std, dtype=np.float32)
    std = np.maximum(std, 1e-6)
    bs = np.asarray(bs_position, dtype=np.float32)
    bs_norm = (bs - mean) / std
    pts_norm = (pts - mean) / std

    rng = np.random.default_rng(cfg.seed)
    if len(pts_norm) > cfg.n_input_points:
        idx = rng.choice(len(pts_norm), cfg.n_input_points, replace=False)
        pts_norm = pts_norm[idx]

    fps_idx = _farthest_point_sample(
        pts_norm, min(cfg.n_scene_tokens, len(pts_norm)))
    centres = pts_norm[fps_idx]

    # Pairwise centre-to-input distances using the quadratic identity rather
    # than a (T,N,3) difference tensor.  At 768 x 16K this stays near 50 MB.
    centre_power = np.sum(centres * centres, axis=1, keepdims=True)
    point_power = np.sum(pts_norm * pts_norm, axis=1, keepdims=True).T
    dist2 = centre_power + point_power - 2.0 * (centres @ pts_norm.T)
    np.maximum(dist2, 0.0, out=dist2)
    k = max(1, min(cfg.knn, pts_norm.shape[0]))
    nn_idx = np.argpartition(dist2, kth=k - 1, axis=1)[:, :k]
    neighbours = pts_norm[nn_idx]
    offsets = neighbours - centres[:, None, :]

    local_mean = offsets.mean(axis=1)
    local_std = offsets.std(axis=1)
    local_radius = np.sqrt(
        np.take_along_axis(dist2, nn_idx, axis=1).max(axis=1) + 1e-8)
    log_radius = np.log1p(local_radius)[:, None]
    log_density = (-3.0 * np.log(local_radius + 1e-4))[:, None]
    rel_bs = centres - bs_norm[None, :]
    radial_range = np.linalg.norm(rel_bs, axis=1, keepdims=True)
    height = centres[:, 2:3]

    features = np.concatenate(
        [
            centres,          # 3
            rel_bs,           # 3
            local_mean,       # 3
            local_std,        # 3
            log_radius,       # 1
            log_density,      # 1
            radial_range,     # 1
        ],
        axis=1,
    ).astype(np.float32)
    assert features.shape[1] == SCENE_FEATURE_DIM
    # Keep the first FP16 projection numerically safe. The clipping is only a
    # guardrail for abnormal map extents; ordinary normalised geometry is
    # unchanged.
    features = np.nan_to_num(
        features, nan=0.0, posinf=32.0, neginf=-32.0)
    features = np.clip(features, -32.0, 32.0).astype(np.float32)
    bs_norm = np.nan_to_num(
        bs_norm, nan=0.0, posinf=32.0, neginf=-32.0)
    bs_norm = np.clip(bs_norm, -32.0, 32.0).astype(np.float32)
    # Height is already represented by centres[:, 2]; keeping the descriptor
    # at 15 dims avoids redundant input while retaining explicit range.
    _ = height
    return features, bs_norm.astype(np.float32)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiheadAttentionBlock(nn.Module):
    """Pre-norm residual MAB used by Set Transformer and Perceiver blocks."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(query)
        kv = self.norm_kv(key_value)
        attended, _ = self.attn(q, kv, kv, need_weights=False)
        x = query + self.dropout(attended)
        return x + self.ff(self.norm_ff(x))


class InducedSetAttentionBlock(nn.Module):
    """ISAB: inducing points attend to the set, then the set attends back."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_inducing: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.inducing = nn.Parameter(
            torch.randn(1, n_inducing, d_model) / d_model**0.5)
        self.induce = MultiheadAttentionBlock(
            d_model, n_heads, d_ff, dropout)
        self.project_back = MultiheadAttentionBlock(
            d_model, n_heads, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inducing = self.inducing.expand(x.shape[0], -1, -1)
        summary = self.induce(inducing, x)
        return self.project_back(x, summary)


class PoolingByMultiheadAttention(nn.Module):
    """PMA compresses an unordered set into a fixed number of scene tokens."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_seeds: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.seeds = nn.Parameter(
            torch.randn(1, n_seeds, d_model) / d_model**0.5)
        self.mab = MultiheadAttentionBlock(
            d_model, n_heads, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seeds = self.seeds.expand(x.shape[0], -1, -1)
        return self.mab(seeds, x)


class SceneEncoder(nn.Module):
    """Trainable Set Transformer over cached map-geometry descriptors."""

    def __init__(
        self,
        input_dim: int = SCENE_FEATURE_DIM,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        n_inducing: int = 128,
        n_output_tokens: int = 256,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [
                InducedSetAttentionBlock(
                    d_model, n_heads, n_inducing, d_ff, dropout)
                for _ in range(n_layers)
            ]
        )
        self.pool = PoolingByMultiheadAttention(
            d_model, n_heads, n_output_tokens, d_ff, dropout)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Encode ``(B,T,F)`` descriptors into ``(B,K,d_model)`` tokens."""
        x = self.input_proj(features)
        for block in self.blocks:
            x = block(x)
        return self.output_norm(self.pool(x))
