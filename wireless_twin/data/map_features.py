"""Map-aware geometric features from the environment point cloud.

A position-only model cannot know *why* the angular power spectrum at a given
location looks the way it does — that is set by the buildings around the user
and whether the direct path to the base station is blocked.  These features
inject that geometry so the network can predict LoS/NLoS and the dominant
reflector directions, which is the first-order driver of the PAS.

We compute, per user position, cheap ray/neighbourhood statistics against the
``RoundX_Map.ply`` point cloud (no ray tracing of the channel — purely
geometric context features for the AI model):

* **LoS-to-BS blockage** — sample points along the user->BS segment and measure
  the clearance to the nearest cloud point; a blocked segment => NLoS.
* **Directional openness** — in a few azimuth sectors around the user, the
  distance to the nearest tall obstacle; encodes which directions are walled
  (and therefore which BS angles are reachable only via reflection).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

try:
    from scipy.spatial import cKDTree
    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    _HAVE_SCIPY = False

N_SECTORS = 8
MAP_FEATURE_DIM = 3 + N_SECTORS  # LoS(3) + directional openness(8) = 11


def _clip_scale(x: np.ndarray, cap: float) -> np.ndarray:
    return np.clip(x, 0.0, cap) / cap


def compute_map_features(
    positions: np.ndarray,
    bs_position: Sequence[float],
    points: np.ndarray,
    n_ray_samples: int = 24,
    block_thresh: float = 2.5,
    sector_radius: float = 80.0,
    max_points: int = 400_000,
) -> np.ndarray:
    """Return ``(P, MAP_FEATURE_DIM)`` map-context features (all in [0, 1]-ish)."""
    if not _HAVE_SCIPY:
        raise ImportError("scipy is required for map features (pip install scipy)")

    positions = np.asarray(positions, dtype=np.float64)
    bs = np.asarray(bs_position, dtype=np.float64).reshape(3)
    pts = np.asarray(points, dtype=np.float64)

    # Downsample the (multi-million-point) cloud for tractable neighbour queries.
    if len(pts) > max_points:
        step = len(pts) // max_points
        pts = pts[::step]

    tree3d = cKDTree(pts)

    # --- LoS-to-BS blockage -------------------------------------------
    ts = np.linspace(0.05, 0.95, n_ray_samples)
    seg = bs[None, None, :] - positions[:, None, :]
    samples = positions[:, None, :] + ts[None, :, None] * seg      # (P, T, 3)
    dist, _ = tree3d.query(samples.reshape(-1, 3), k=1)
    dist = dist.reshape(len(positions), n_ray_samples)
    los = np.stack([
        _clip_scale(dist.min(axis=1), 15.0),                       # min clearance
        _clip_scale(dist.mean(axis=1), 15.0),                      # mean clearance
        (dist < block_thresh).mean(axis=1),                        # blocked fraction
    ], axis=1)

    # --- directional openness (horizontal) ----------------------------
    tree2d = cKDTree(pts[:, :2])
    sector = np.full((len(positions), N_SECTORS), sector_radius)
    for i, p in enumerate(positions):
        idx = tree2d.query_ball_point(p[:2], sector_radius)
        if not idx:
            continue
        near = pts[idx]
        rel = near[:, :2] - p[:2]
        rng = np.linalg.norm(rel, axis=1)
        # keep obstacles taller than the user (potential blockers/reflectors)
        tall = near[:, 2] > p[2] + 2.0
        rel, rng = rel[tall], rng[tall]
        if len(rng) == 0:
            continue
        az = (np.arctan2(rel[:, 1], rel[:, 0]) + np.pi)            # [0, 2pi)
        bins = np.minimum((az / (2 * np.pi) * N_SECTORS).astype(int), N_SECTORS - 1)
        for b in range(N_SECTORS):
            m = bins == b
            if m.any():
                sector[i, b] = rng[m].min()
    sector = _clip_scale(sector, sector_radius)

    return np.concatenate([los, sector], axis=1).astype(np.float32)
