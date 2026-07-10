"""Physics-informed position augmentation.

The dominant angle-of-departure at the base station is a geometric function of
the sample position relative to the BS.  Feeding that geometry explicitly (not
just raw ``x, y, z``) gives the network a strong, physically-grounded cue for
the large-scale structure of the PAS.  This is the single most useful non-map
feature for a position-only model.

``augment_positions`` maps ``(P, 3)`` raw coordinates to ``(P, 9)``:

    [ x, y, z,  dx, dy, dz,  r,  azimuth,  elevation ]

where ``(dx,dy,dz) = pos - BS`` and ``(r, azimuth, elevation)`` are the spherical
coordinates of that BS->sample vector.  The transform is deterministic so the
exact same features are reproduced at inference time.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

AUGMENTED_DIM = 9


def augment_positions(pos: np.ndarray, bs_position: Sequence[float]) -> np.ndarray:
    """Return ``(P, 9)`` geometry-augmented features for raw ``(P, 3)`` coords."""
    pos = np.asarray(pos, dtype=np.float32)
    bs = np.asarray(bs_position, dtype=np.float32).reshape(1, 3)
    rel = pos - bs                                          # BS -> sample
    r = np.linalg.norm(rel, axis=1, keepdims=True)
    rho = np.linalg.norm(rel[:, :2], axis=1)               # horizontal range
    az = np.arctan2(rel[:, 1], rel[:, 0])[:, None]         # azimuth
    el = np.arctan2(rel[:, 2], rho + 1e-9)[:, None]        # elevation
    return np.concatenate([pos, rel, r, az, el], axis=1).astype(np.float32)
