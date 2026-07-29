#!/usr/bin/env python3
"""Shared low-capacity gate for raw and physically transported PAS experts."""
from __future__ import annotations

import math

import torch
from torch import nn


GATE_CENTER = 0.25
GATE_RADIUS = 0.20

FEATURE_NAMES = (
    "raw_transport_cosine",
    "raw_l1_scaled",
    "transport_l1_scaled",
    "raw_max",
    "transport_max",
    "raw_entropy",
    "transport_entropy",
    "raw_h_cos",
    "raw_h_sin",
    "raw_v_cos",
    "raw_v_sin",
    "transport_h_cos",
    "transport_h_sin",
    "transport_v_cos",
    "transport_v_sin",
    "raw_h_concentration",
    "raw_v_concentration",
    "transport_h_concentration",
    "transport_v_concentration",
    "nearest_distance",
    "median_distance",
    "distance_spread",
    "query_radius",
    "query_angle_sin",
    "query_angle_cos",
    "query_elevation",
    "subcarrier_sin",
    "subcarrier_cos",
    "ue_antenna_0",
    "ue_antenna_1",
    "ue_antenna_2",
    "ue_antenna_3",
)


def _unit(value: torch.Tensor) -> torch.Tensor:
    return value / value.norm(dim=-1, keepdim=True).clamp_min(
        torch.finfo(value.dtype).tiny
    )


def build_gate_features(
    raw: torch.Tensor,
    transported: torch.Tensor,
    query_position: torch.Tensor,
    neighbor_distance: torch.Tensor,
    bs_position: torch.Tensor,
    mh: int,
    mv: int,
) -> torch.Tensor:
    """Return source-only features with shape ``(B, N, S, F)``."""
    if raw.shape != transported.shape:
        raise ValueError(
            f"PAS expert shape mismatch: {raw.shape} != {transported.shape}"
        )
    if raw.ndim != 4 or raw.shape[-1] != mh * mv:
        raise ValueError(f"unexpected PAS shape {raw.shape}")
    if raw.shape[1] != 4:
        raise ValueError("the current gate expects four UE antennas")

    raw = _unit(raw.float())
    transported = _unit(transported.float())
    dtype = raw.dtype
    device = raw.device
    batch, n_ue, subcarriers, angles = raw.shape
    tiny = torch.finfo(dtype).tiny
    log_angles = math.log(angles)

    raw_probability = raw / raw.sum(dim=-1, keepdim=True).clamp_min(tiny)
    transport_probability = transported / transported.sum(
        dim=-1, keepdim=True
    ).clamp_min(tiny)
    raw_entropy = -(
        raw_probability
        * raw_probability.clamp_min(tiny).log()
    ).sum(dim=-1) / log_angles
    transport_entropy = -(
        transport_probability
        * transport_probability.clamp_min(tiny).log()
    ).sum(dim=-1) / log_angles

    raw_grid = raw_probability.reshape(
        batch, n_ue, subcarriers, mh, mv
    )
    transport_grid = transport_probability.reshape(
        batch, n_ue, subcarriers, mh, mv
    )
    raw_h = raw_grid.sum(dim=-1)
    raw_v = raw_grid.sum(dim=-2)
    transport_h = transport_grid.sum(dim=-1)
    transport_v = transport_grid.sum(dim=-2)
    h_phase = (
        2.0
        * math.pi
        * torch.arange(mh, dtype=dtype, device=device)
        / mh
    )
    v_phase = (
        2.0
        * math.pi
        * torch.arange(mv, dtype=dtype, device=device)
        / mv
    )
    h_cos = h_phase.cos()
    h_sin = h_phase.sin()
    v_cos = v_phase.cos()
    v_sin = v_phase.sin()

    def moment(
        probability: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        real = (probability * cosine).sum(dim=-1)
        imag = (probability * sine).sum(dim=-1)
        concentration = torch.sqrt(
            real.square() + imag.square() + tiny
        )
        return real, imag, concentration

    raw_h_cos, raw_h_sin, raw_h_concentration = moment(
        raw_h, h_cos, h_sin
    )
    raw_v_cos, raw_v_sin, raw_v_concentration = moment(
        raw_v, v_cos, v_sin
    )
    trans_h_cos, trans_h_sin, trans_h_concentration = moment(
        transport_h, h_cos, h_sin
    )
    trans_v_cos, trans_v_sin, trans_v_concentration = moment(
        transport_v, v_cos, v_sin
    )

    def expand_query(value: torch.Tensor) -> torch.Tensor:
        return value[:, None, None].expand(-1, n_ue, subcarriers)

    neighbor_distance = neighbor_distance.to(
        device=device, dtype=dtype
    )
    nearest_distance = neighbor_distance[:, 0] / 3.0
    median_distance = neighbor_distance.median(dim=1).values / 5.0
    distance_spread = neighbor_distance.std(
        dim=1, correction=0
    ) / 3.0
    query_position = query_position.to(device=device, dtype=dtype)
    bs_position = bs_position.to(device=device, dtype=dtype)
    offset = query_position - bs_position[None]
    horizontal_radius = offset[:, :2].norm(dim=-1).clamp_min(1e-3)
    query_radius = offset.norm(dim=-1) / 200.0
    query_angle = torch.atan2(offset[:, 1], offset[:, 0])
    query_elevation = offset[:, 2] / horizontal_radius

    subcarrier_phase = (
        2.0
        * math.pi
        * torch.arange(
            subcarriers, dtype=dtype, device=device
        )
        / subcarriers
    )
    subcarrier_sin = subcarrier_phase.sin()[None, None].expand(
        batch, n_ue, -1
    )
    subcarrier_cos = subcarrier_phase.cos()[None, None].expand(
        batch, n_ue, -1
    )
    ue_one_hot = torch.eye(
        n_ue, dtype=dtype, device=device
    )[None, :, None].expand(batch, -1, subcarriers, -1)

    scalar_columns = [
        (raw * transported).sum(dim=-1),
        raw.sum(dim=-1) / math.sqrt(angles),
        transported.sum(dim=-1) / math.sqrt(angles),
        raw.amax(dim=-1),
        transported.amax(dim=-1),
        raw_entropy,
        transport_entropy,
        raw_h_cos,
        raw_h_sin,
        raw_v_cos,
        raw_v_sin,
        trans_h_cos,
        trans_h_sin,
        trans_v_cos,
        trans_v_sin,
        raw_h_concentration,
        raw_v_concentration,
        trans_h_concentration,
        trans_v_concentration,
        expand_query(nearest_distance),
        expand_query(median_distance),
        expand_query(distance_spread),
        expand_query(query_radius),
        expand_query(query_angle.sin()),
        expand_query(query_angle.cos()),
        expand_query(query_elevation),
        subcarrier_sin,
        subcarrier_cos,
    ]
    features = torch.cat(
        [torch.stack(scalar_columns, dim=-1), ue_one_hot],
        dim=-1,
    )
    if features.shape[-1] != len(FEATURE_NAMES):
        raise RuntimeError(
            f"feature count mismatch: {features.shape[-1]} "
            f"!= {len(FEATURE_NAMES)}"
        )
    return features


class PasTransportGate(nn.Module):
    """A deliberately small bounded PAS transport gate."""

    def __init__(self, feature_dim: int, architecture: str) -> None:
        super().__init__()
        self.architecture = architecture
        if architecture == "linear":
            self.network = nn.Linear(feature_dim, 1)
        elif architecture == "mlp8":
            self.network = nn.Sequential(
                nn.Linear(feature_dim, 8),
                nn.SiLU(),
                nn.Linear(8, 1),
            )
        else:
            raise ValueError(f"unsupported architecture={architecture}")
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        output = (
            self.network
            if isinstance(self.network, nn.Linear)
            else self.network[-1]
        )
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logit = self.network(features).squeeze(-1)
        return GATE_CENTER + GATE_RADIUS * torch.tanh(logit)


def mixed_cosine_from_dots(
    beta: torch.Tensor,
    raw_truth: torch.Tensor,
    transport_truth: torch.Tensor,
    raw_transport: torch.Tensor,
) -> torch.Tensor:
    """Cosine of a normalized linear mixture from three pairwise dots."""
    left = 1.0 - beta
    numerator = left * raw_truth + beta * transport_truth
    denominator = torch.sqrt(
        left.square()
        + beta.square()
        + 2.0 * left * beta * raw_transport
    ).clamp_min(torch.finfo(beta.dtype).tiny)
    return numerator / denominator
