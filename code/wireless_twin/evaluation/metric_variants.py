"""Alternative PAS/PDP definitions for offline audit.

The task statement leaves enough room for implementation choices that a single
local metric can become misleading.  This module keeps the current definition
intact and adds side-by-side variants for model selection and reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Sequence

import numpy as np
import torch

from ..data.setup_config import ChannelSpec
from ..signal import cosine_similarity_along_last
from .metrics import channel_nmse, competition_score


@dataclass(frozen=True)
class RobustMetricResult:
    pas: Dict[str, float]
    pdp: Dict[str, float]
    nmse: float
    c1_robust: float
    c2_robust: float
    c_robust: float
    c_current: float


def pas_2d_sum_pol(h: torch.Tensor, spec: ChannelSpec) -> torch.Tensor:
    """P-A: 2-D BS FFT over MH/MV, sum polarisations before cosine."""
    b = h.shape[0]
    hm = h.reshape(b, spec.mh, spec.mv, spec.mp, spec.n, spec.s)
    hf = torch.fft.fft2(hm, dim=(1, 2))
    power = hf.abs().square().sum(dim=3)
    pas = power.reshape(b, spec.mh * spec.mv, spec.n, spec.s)
    return pas.permute(0, 2, 3, 1).contiguous()


def pas_2d_sep_pol(h: torch.Tensor, spec: ChannelSpec) -> torch.Tensor:
    """P-B: 2-D BS FFT, keep polarisations as separate cosine samples."""
    b = h.shape[0]
    hm = h.reshape(b, spec.mh, spec.mv, spec.mp, spec.n, spec.s)
    hf = torch.fft.fft2(hm, dim=(1, 2))
    power = hf.abs().square()
    pas = power.reshape(b, spec.mh * spec.mv, spec.mp, spec.n, spec.s)
    return pas.permute(0, 3, 4, 2, 1).reshape(
        b, spec.n, spec.s, spec.mp, spec.mh * spec.mv)


def pas_1d_flat_m(h: torch.Tensor, spec: ChannelSpec) -> torch.Tensor:
    """P-C: treat M=256 as one antenna sequence and run a 1-D FFT."""
    hf = torch.fft.fft(h, dim=1)
    power = hf.abs().square()
    return power.permute(0, 2, 3, 1).contiguous()


def pdp_per_mn(h: torch.Tensor, spec: ChannelSpec) -> torch.Tensor:
    """D-A: delay cosine per BS/UE antenna pair."""
    del spec
    return torch.fft.ifft(h, dim=-1).abs().square()


def pdp_sum_m(h: torch.Tensor, spec: ChannelSpec) -> torch.Tensor:
    """D-B: sum BS antenna power, keep UE antennas separate."""
    del spec
    return torch.fft.ifft(h, dim=-1).abs().square().sum(dim=1)


def pdp_sum_mn(h: torch.Tensor, spec: ChannelSpec) -> torch.Tensor:
    """D-C: sum all receive/transmit antenna power per position."""
    del spec
    return torch.fft.ifft(h, dim=-1).abs().square().sum(dim=(1, 2))


PAS_VARIANTS: Dict[str, Callable[[torch.Tensor, ChannelSpec], torch.Tensor]] = {
    "pas_2d_sum_pol": pas_2d_sum_pol,
    "pas_2d_sep_pol": pas_2d_sep_pol,
    "pas_1d_flat_m": pas_1d_flat_m,
}

PDP_VARIANTS: Dict[str, Callable[[torch.Tensor, ChannelSpec], torch.Tensor]] = {
    "pdp_per_mn": pdp_per_mn,
    "pdp_sum_m": pdp_sum_m,
    "pdp_sum_mn": pdp_sum_mn,
}


def _to_complex_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(arr)).to(torch.complex64)


def _chunked_variant_scores(
    pred: np.ndarray,
    gt: np.ndarray,
    spec: ChannelSpec,
    variants: Dict[str, Callable[[torch.Tensor, ChannelSpec], torch.Tensor]],
    chunk: int,
) -> Dict[str, float]:
    totals = {name: 0.0 for name in variants}
    seen = 0
    for i in range(0, len(gt), chunk):
        pc = _to_complex_tensor(pred[i:i + chunk])
        gc = _to_complex_tensor(gt[i:i + chunk])
        bs = pc.shape[0]
        with torch.no_grad():
            for name, fn in variants.items():
                totals[name] += float(
                    cosine_similarity_along_last(fn(pc, spec), fn(gc, spec))
                ) * bs
        seen += bs
    return {name: value / max(seen, 1) for name, value in totals.items()}


def robust_channel_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    spec: ChannelSpec,
    weights: Sequence[float] | None = None,
    chunk: int = 16,
) -> RobustMetricResult:
    """Evaluate all metric variants plus the robust aggregate."""
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs gt {gt.shape}")
    weights = list(weights) if weights is not None else list(spec.metric_weights)
    pas = _chunked_variant_scores(pred, gt, spec, PAS_VARIANTS, chunk)
    pdp = _chunked_variant_scores(pred, gt, spec, PDP_VARIANTS, chunk)
    nmse = channel_nmse(pred, gt, chunk=chunk)
    pas_values = np.asarray(list(pas.values()), dtype=np.float64)
    pdp_values = np.asarray(list(pdp.values()), dtype=np.float64)
    c1_robust = float(0.5 * pas_values.min() + 0.5 * pas_values.mean())
    c2_robust = float(0.5 * pdp_values.min() + 0.5 * pdp_values.mean())
    c_robust = competition_score(c1_robust, c2_robust, nmse, weights)
    c_current = competition_score(
        pas["pas_2d_sum_pol"], pdp["pdp_per_mn"], nmse, weights)
    return RobustMetricResult(
        pas=pas,
        pdp=pdp,
        nmse=nmse,
        c1_robust=c1_robust,
        c2_robust=c2_robust,
        c_robust=c_robust,
        c_current=c_current,
    )


def robust_metrics_dict(result: RobustMetricResult) -> Dict[str, float]:
    out: Dict[str, float] = {}
    out.update(result.pas)
    out.update(result.pdp)
    out.update(
        {
            "nmse": result.nmse,
            "c1_robust": result.c1_robust,
            "c2_robust": result.c2_robust,
            "c_robust": result.c_robust,
            "c_current": result.c_current,
        }
    )
    return out
