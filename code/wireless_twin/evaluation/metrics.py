"""Competition metrics (task book §2.2).

    C1 = PAS cosine similarity   (higher is better, in [0, 1])
    C2 = PDP cosine similarity   (higher is better, in [0, 1])
    C3 = NMSE                    (lower is better,  >= 0)
    C  = w1*C1 + w2*C2 + w3 * 1/(1 + C3)

These operate on numpy complex arrays ``(P, M, N, S)`` (predicted vs ground
truth) so they can score a saved ``RoundX_Test_Channel.npy`` offline.  The
underlying PAS/PDP transforms are the same ones used by the training loss
(:mod:`wireless_twin.signal`), guaranteeing train/eval consistency.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch

from ..data.setup_config import ChannelSpec
from ..signal import cosine_similarity_along_last, pas_spectrum, pdp_spectrum


def _to_complex_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(arr)).to(torch.complex64)


def channel_nmse(pred: np.ndarray, gt: np.ndarray, chunk: int = 16) -> float:
    """C3: ``||pred - gt||^2 / ||gt||^2`` over the whole test set."""
    num = 0.0
    den = 0.0
    for i in range(0, len(gt), chunk):
        diff = pred[i:i + chunk] - gt[i:i + chunk]
        target = gt[i:i + chunk]
        num += float(
            np.square(diff.real, dtype=np.float64).sum()
            + np.square(diff.imag, dtype=np.float64).sum()
        )
        den += float(
            np.square(target.real, dtype=np.float64).sum()
            + np.square(target.imag, dtype=np.float64).sum()
        )
    return num / max(den, 1e-300)


def pas_accuracy(pred: np.ndarray, gt: np.ndarray, spec: ChannelSpec,
                 chunk: int = 64) -> float:
    """C1: mean PAS cosine similarity, computed in position chunks."""
    return _chunked_cosine(pred, gt, spec, pas_spectrum, chunk)


def pdp_accuracy(pred: np.ndarray, gt: np.ndarray, spec: ChannelSpec,
                 chunk: int = 64) -> float:
    """C2: mean PDP cosine similarity, computed in position chunks."""
    return _chunked_cosine(pred, gt, spec, pdp_spectrum, chunk)


def _chunked_cosine(pred, gt, spec, transform, chunk) -> float:
    p = pred.shape[0]
    total, seen = 0.0, 0
    for i in range(0, p, chunk):
        pc = _to_complex_tensor(pred[i:i + chunk])
        gc = _to_complex_tensor(gt[i:i + chunk])
        with torch.no_grad():
            sim = cosine_similarity_along_last(
                transform(pc, spec), transform(gc, spec))
        bs = pc.shape[0]
        total += float(sim) * bs
        seen += bs
    return total / max(seen, 1)


def competition_score(c1: float, c2: float, c3: float,
                      weights: Sequence[float]) -> float:
    """Combined ranking metric ``C``."""
    w1, w2, w3 = weights
    return float(w1 * c1 + w2 * c2 + w3 * (1.0 / (1.0 + c3)))


def evaluate_channels(pred: np.ndarray, gt: np.ndarray, spec: ChannelSpec,
                      weights: Optional[Sequence[float]] = None) -> Dict[str, float]:
    """Return all three accuracies plus the combined score C."""
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs gt {gt.shape}")
    weights = list(weights) if weights is not None else list(spec.metric_weights)
    c1 = pas_accuracy(pred, gt, spec)
    c2 = pdp_accuracy(pred, gt, spec)
    c3 = channel_nmse(pred, gt)
    return {
        "C1_PAS": c1,
        "C2_PDP": c2,
        "C3_NMSE": c3,
        "C": competition_score(c1, c2, c3, weights),
    }
