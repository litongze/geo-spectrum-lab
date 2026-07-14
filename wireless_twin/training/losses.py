"""Loss functions aligned with the competition score.

The ranking metric is ``C = w1*C1 + w2*C2 + w3/(1+C3)`` where C1/C2 are PAS/PDP
cosine similarities and C3 is NMSE (task book §2.2).  We optimise a loss that
mirrors it directly:

    L = nmse  +  lambda_pas * (1 - C1)  +  lambda_pdp * (1 - C2)

so lowering the loss raises the leaderboard score.  All terms are differentiable
and scale-invariant, so they behave well on normalised channels.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..data.setup_config import ChannelSpec
from ..signal import cosine_similarity_along_last, pas_spectrum, pdp_spectrum


def nmse(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalised MSE ``||pred-gt||^2 / ||gt||^2`` (complex or real tensors)."""
    diff = (pred - gt).abs() ** 2
    denom = (gt.abs() ** 2).sum().clamp_min(eps)
    return diff.sum() / denom


class ChannelLoss(nn.Module):
    """Composite magnitude + PAS/PDP-consistency loss over complex channels.

    The PAS/PDP terms are cosine similarities (scale-invariant), so on their own
    they let the model shrink its output magnitude toward zero (NMSE -> 1) while
    still matching the *shape* of the spectra.  To also learn the magnitude we
    add a direct MSE term (``lambda_mag``); because the cosine terms produce much
    larger gradients when the prediction is small, ``lambda_mag`` usually needs
    to be well above 1 to compete under Adam.
    """

    def __init__(self, spec: ChannelSpec,
                 lambda_pas: float = 1.0,
                 lambda_pdp: float = 1.0,
                 lambda_mag: float = 1.0,
                 lambda_magabs: float = 0.0,
                 lambda_magdb: float = 0.0,
                 db_floor: float = 1e-9) -> None:
        super().__init__()
        self.spec = spec
        self.lambda_pas = lambda_pas
        self.lambda_pdp = lambda_pdp
        self.lambda_mag = lambda_mag
        self.lambda_magabs = lambda_magabs
        self.lambda_magdb = lambda_magdb
        self.db_floor = db_floor

    def forward(self, pred_h: torch.Tensor, gt_h: torch.Tensor) -> dict:
        """``pred_h`` / ``gt_h`` are complex ``(B, M, N, S)`` tensors."""
        loss_nmse = nmse(pred_h, gt_h)
        # complex MSE: phase-sensitive -> collapses to 0 when phase can't match
        loss_mag = ((pred_h - gt_h).abs() ** 2).mean()
        # phase-FREE magnitude MSE: pulls |pred| to the true scale without
        # fighting the phase wall -> lets the model output the label scale.
        loss_magabs = ((pred_h.abs() - gt_h.abs()) ** 2).mean()
        # dB (log) magnitude MSE: matches the huge-dynamic-range gain in log
        # space so small values are learned too (link-budget in dB).
        lp = torch.log10(pred_h.abs().clamp_min(self.db_floor))
        lg = torch.log10(gt_h.abs().clamp_min(self.db_floor))
        loss_magdb = ((20.0 * (lp - lg)) ** 2).mean()

        # Normalise pred to unit RMS before the PAS/PDP terms.  The two losses
        # are scale-invariant, so magnitude is a *free direction* the optimiser
        # can drift to ~1e-14 -> |FFT|^2 underflows in float32 and the cosine
        # becomes a spurious ~0.98 artefact (a collapse trap that attention
        # gates fall into).  Pinning pred to O(1) here removes the free
        # direction and keeps the spectra numerically healthy.
        rms = pred_h.abs().pow(2).mean().clamp_min(1e-30).sqrt()
        pred_n = pred_h / rms
        c1 = cosine_similarity_along_last(
            pas_spectrum(pred_n, self.spec), pas_spectrum(gt_h, self.spec))
        c2 = cosine_similarity_along_last(
            pdp_spectrum(pred_n, self.spec), pdp_spectrum(gt_h, self.spec))

        total = (self.lambda_mag * loss_mag
                 + self.lambda_magabs * loss_magabs
                 + self.lambda_magdb * loss_magdb
                 + self.lambda_pas * (1.0 - c1)
                 + self.lambda_pdp * (1.0 - c2))
        return {
            "loss": total,
            "nmse": loss_nmse.detach(),
            "pas": c1.detach(),
            "pdp": c2.detach(),
        }


def _diagnostics(pred_h: torch.Tensor, gt_h: torch.Tensor,
                 spec: ChannelSpec) -> dict:
    """The three competition metrics, detached (for logging any loss)."""
    with torch.no_grad():
        return {
            "nmse": nmse(pred_h, gt_h).detach(),
            "pas": cosine_similarity_along_last(
                pas_spectrum(pred_h, spec), pas_spectrum(gt_h, spec)).detach(),
            "pdp": cosine_similarity_along_last(
                pdp_spectrum(pred_h, spec), pdp_spectrum(gt_h, spec)).detach(),
        }


class MseLoss(nn.Module):
    """Pure complex MSE — the NeRF2 objective (``sig2mse``) applied to H."""

    def __init__(self, spec: ChannelSpec, **_) -> None:
        super().__init__()
        self.spec = spec

    def forward(self, pred_h, gt_h):
        loss = ((pred_h - gt_h).abs() ** 2).mean()
        return {"loss": loss, **_diagnostics(pred_h, gt_h, self.spec)}


class L1Loss(nn.Module):
    """Complex L1 — analogue of the WRF-GS reconstruction term (sans SSIM)."""

    def __init__(self, spec: ChannelSpec, **_) -> None:
        super().__init__()
        self.spec = spec

    def forward(self, pred_h, gt_h):
        loss = (pred_h - gt_h).abs().mean()
        return {"loss": loss, **_diagnostics(pred_h, gt_h, self.spec)}


# --- loss registry (loss is swappable per "method", like the model) --------
_LOSS_REGISTRY = {
    "competition": ChannelLoss,   # PAS/PDP consistency + magnitude (aligned to C)
    "mse": MseLoss,               # NeRF2-style pure MSE
    "l1": L1Loss,                 # WRF-GS-style L1 reconstruction
}


def build_loss(name: str, spec: ChannelSpec, **kwargs) -> nn.Module:
    if name not in _LOSS_REGISTRY:
        raise KeyError(f"unknown loss '{name}'. Have: {list(_LOSS_REGISTRY)}")
    return _LOSS_REGISTRY[name](spec, **kwargs)


def available_losses():
    return sorted(_LOSS_REGISTRY)
