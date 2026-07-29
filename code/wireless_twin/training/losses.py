"""Competition-aligned differentiable losses.

The official NMSE is a *single ratio over the whole evaluation set*::

    sum |H_hat - H|^2 / sum |H|^2

With batch size 1, averaging a ratio computed independently for every sample is
not equivalent to that metric and becomes numerically dominated by very weak
channel locations.  The dataset loader globally RMS-normalises the channel, so
mean squared error on the normalised channel is an unbiased minibatch proxy for
the official global NMSE.  Exact epoch NMSE is still reported by aggregating the
error and target powers across all batches in :mod:`trainer`.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.setup_config import ChannelSpec
from ..signal import cosine_similarity_along_last, pas_spectrum, pdp_spectrum


def nmse_power_sums(
    pred: torch.Tensor,
    gt: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return numerator and denominator sums for exact global NMSE."""
    error_power = (pred - gt).abs().square().sum()
    target_power = gt.abs().square().sum()
    return error_power, target_power


def nmse(
    pred: torch.Tensor,
    gt: torch.Tensor,
    eps: float = 1e-12,
    mode: str = "global",
) -> torch.Tensor:
    """NMSE-related objective for complex or real tensors.

    Parameters
    ----------
    mode:
        ``"dataset"`` / ``"mse"``
            Mean squared error on globally RMS-normalised channels. This is the
            recommended training objective and is an unbiased minibatch proxy
            for the official whole-dataset NMSE.
        ``"global"``
            Ratio over the current batch. Correct only when the batch itself is
            representative or contains the whole evaluation set.
        ``"sample"``
            Mean of per-position NMSE ratios. Kept for ablations, but strongly
            discouraged for wireless data with large path-loss variation.
    """
    diff_power = (pred - gt).abs().square()
    gt_power = gt.abs().square()
    if mode in {"dataset", "mse", "dataset_mse"}:
        return diff_power.mean()
    if mode == "global":
        return diff_power.sum() / gt_power.sum().clamp_min(eps)
    if mode == "sample":
        dims = tuple(range(1, pred.ndim))
        ratio = diff_power.sum(dims) / gt_power.sum(dims).clamp_min(eps)
        return ratio.mean()
    raise ValueError(f"unknown NMSE mode: {mode}")


class ChannelLoss(nn.Module):
    """Stable NMSE proxy + PAS/PDP consistency + angle-delay sparsity."""

    def __init__(
        self,
        spec: ChannelSpec,
        lambda_nmse: float = 20.0,
        lambda_pas: float = 1.0,
        lambda_pdp: float = 1.0,
        lambda_sparse: float = 0.0,
        lambda_log_power: float = 0.0,
        nmse_mode: str = "dataset",
    ) -> None:
        super().__init__()
        self.spec = spec
        self.lambda_nmse = float(lambda_nmse)
        self.lambda_pas = float(lambda_pas)
        self.lambda_pdp = float(lambda_pdp)
        self.lambda_sparse = float(lambda_sparse)
        self.lambda_log_power = float(lambda_log_power)
        self.nmse_mode = nmse_mode

    def forward(
        self,
        pred_h: torch.Tensor,
        gt_h: torch.Tensor,
        sparse_term: Optional[torch.Tensor] = None,
    ) -> dict:
        error_power, target_power = nmse_power_sums(pred_h, gt_h)
        pred_power = pred_h.abs().square().sum()
        cross_real = (pred_h.conj() * gt_h).real.sum()
        loss_nmse = nmse(pred_h, gt_h, mode=self.nmse_mode)
        dims = tuple(range(1, pred_h.ndim))
        pred_sample_power = pred_h.abs().square().sum(dim=dims).clamp_min(1e-30)
        gt_sample_power = gt_h.abs().square().sum(dim=dims).clamp_min(1e-30)
        log_power_ratio = torch.log(pred_sample_power) - torch.log(gt_sample_power)
        loss_log_power = F.huber_loss(
            log_power_ratio,
            torch.zeros_like(log_power_ratio),
            delta=2.0,
            reduction="mean",
        )

        c1 = cosine_similarity_along_last(
            pas_spectrum(pred_h, self.spec),
            pas_spectrum(gt_h, self.spec),
        )
        c2 = cosine_similarity_along_last(
            pdp_spectrum(pred_h, self.spec),
            pdp_spectrum(gt_h, self.spec),
        )

        if sparse_term is None:
            sparse_term = pred_h.real.new_zeros(())
        total = (
            self.lambda_nmse * loss_nmse
            + self.lambda_pas * (1.0 - c1)
            + self.lambda_pdp * (1.0 - c2)
            + self.lambda_sparse * sparse_term
            + self.lambda_log_power * loss_log_power
        )
        return {
            "loss": total,
            # The trainer sums these over every batch and forms one ratio. Do
            # not average batch NMSE ratios when batch size is small.
            "nmse_num": error_power.detach(),
            "nmse_den": target_power.detach(),
            "pred_power": pred_power.detach(),
            "cross_real": cross_real.detach(),
            "pas": c1.detach(),
            "pdp": c2.detach(),
            "sparse": sparse_term.detach(),
            "log_power": loss_log_power.detach(),
        }
