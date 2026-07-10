"""Differentiable PAS / PDP transforms shared by training and evaluation.

Both the training losses and the scoring metrics need the same two physical
transforms of a MIMO-OFDM channel, so they live here (torch, differentiable)
and are imported by :mod:`wireless_twin.training.losses` and
:mod:`wireless_twin.evaluation.metrics`.

Definitions (task book §2.2)
----------------------------
* **PAS** — Power Angle Spectrum on the BS side.  The BS array ``M = MH*MV*MP``
  is transformed to the angular domain with a 2-D spatial FFT over the
  (MH, MV) grid; power is summed over the ``MP`` polarisations.  One PAS vector
  (length ``MH*MV``) is produced per (position, UE-antenna ``n``, sub-carrier
  ``s``).
* **PDP** — Power Delay Profile on the BS side.  An inverse FFT over the ``S``
  sub-carriers maps each antenna pair to the delay domain; power gives one PDP
  vector (length ``S``) per (position, BS-antenna ``m``, UE-antenna ``n``).

The organiser's grader is authoritative; if their angular/delay convention
differs, this is the single file to adjust and every metric/loss follows.
"""

from __future__ import annotations

import torch

from .data.setup_config import ChannelSpec


def pas_spectrum(h: torch.Tensor, spec: ChannelSpec) -> torch.Tensor:
    """Power Angle Spectrum.

    Parameters
    ----------
    h : complex tensor ``(B, M, N, S)``.

    Returns
    -------
    Real, non-negative tensor ``(B, N, S, A)`` with ``A = MH*MV`` angle bins;
    the spectrum lives on the last axis.
    """
    b = h.shape[0]
    hm = h.reshape(b, spec.mh, spec.mv, spec.mp, spec.n, spec.s)
    # 2-D spatial FFT over the BS antenna grid (rows, cols)
    hf = torch.fft.fft2(hm, dim=(1, 2))
    power = hf.real ** 2 + hf.imag ** 2                 # |.|^2
    power = power.sum(dim=3)                            # sum polarisations -> (B,MH,MV,N,S)
    a = spec.mh * spec.mv
    pas = power.reshape(b, a, spec.n, spec.s)           # (B, A, N, S)
    return pas.permute(0, 2, 3, 1).contiguous()        # (B, N, S, A)


def pdp_spectrum(h: torch.Tensor, spec: ChannelSpec) -> torch.Tensor:
    """Power Delay Profile.

    Parameters
    ----------
    h : complex tensor ``(B, M, N, S)``.

    Returns
    -------
    Real, non-negative tensor ``(B, M, N, S)``; the delay spectrum lives on the
    last axis (``S`` taps).
    """
    hd = torch.fft.ifft(h, dim=-1)                      # subcarrier -> delay
    return hd.real ** 2 + hd.imag ** 2


def cosine_similarity_along_last(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Mean cosine similarity between vectors on the last axis.

    ``pred`` and ``gt`` share shape ``(..., L)``.  Cosine similarity is computed
    per length-``L`` vector and averaged over every leading axis, matching the
    task book's "average over all positions / antennas / sub-carriers".

    Note: the denominator is floored at the dtype's smallest positive value
    (not a fixed ``1e-12``) so the metric stays correct for *raw*-scale channels
    whose power spectra are ~1e-13; a fixed eps would swamp the denominator and
    collapse the cosine to zero.
    """
    num = (pred * gt).sum(dim=-1)
    den = pred.norm(dim=-1) * gt.norm(dim=-1)
    den = torch.clamp_min(den, torch.finfo(den.dtype).tiny)
    # Cauchy-Schwarz: a true cosine is in [-1, 1]; clamp guards float underflow
    # on near-zero spectra (e.g. a degenerate model) from exploding the ratio.
    return (num / den).clamp(-1.0, 1.0).mean()
