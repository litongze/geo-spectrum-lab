#!/usr/bin/env python3
"""Shared low-capacity complex-neighbor weighting model."""
from __future__ import annotations

import torch
from torch import nn


FEATURE_NAMES = (
    "log_base_weight",
    "log_distance",
    "effective_distance",
    "radial_delta",
    "tangent_delta",
    "neighbor_rank",
    "log_relative_amplitude",
    "base_agreement_real",
    "base_agreement_imag",
    "base_agreement_abs",
    "nearest_agreement_real",
    "nearest_agreement_imag",
    "nearest_agreement_abs",
    "mean_pair_coherence",
    "query_radius",
    "query_angle_sin",
    "query_angle_cos",
)
ARCHITECTURES = ("mlp16", "set16")


def baseline_coefficients(
    base_weight: torch.Tensor,
    amplitude: torch.Tensor,
) -> torch.Tensor:
    target_amplitude = (
        base_weight * amplitude
    ).sum(dim=1, keepdim=True)
    return (
        base_weight
        * target_amplitude
        / amplitude.clamp_min(torch.finfo(amplitude.dtype).tiny)
    ).to(torch.complex64)


def build_complex_features(
    gram: torch.Tensor,
    base_weight: torch.Tensor,
    distance: torch.Tensor,
    effective_distance: torch.Tensor,
    radial_delta: torch.Tensor,
    tangent_delta: torch.Tensor,
    query_position: torch.Tensor,
    bs_position: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build source-only neighbor features from aligned-channel Gram data."""
    if gram.ndim != 3 or gram.shape[1] != gram.shape[2]:
        raise ValueError(f"unexpected Gram shape {gram.shape}")
    batch, k, _ = gram.shape
    real_dtype = gram.real.dtype
    tiny = torch.finfo(real_dtype).tiny
    amplitude = (
        gram.diagonal(dim1=1, dim2=2).real.clamp_min(tiny).sqrt()
    )
    normalized_gram = gram / (
        amplitude[:, :, None] * amplitude[:, None, :]
    ).clamp_min(tiny)
    base_coefficient = baseline_coefficients(
        base_weight, amplitude
    )
    base_energy = torch.einsum(
        "bi,bij,bj->b",
        base_coefficient.conj(),
        gram,
        base_coefficient,
    ).real.clamp_min(tiny)
    agreement = torch.einsum(
        "bij,bj->bi", gram, base_coefficient
    ) / (
        amplitude * base_energy.sqrt()[:, None]
    ).clamp_min(tiny)
    nearest_agreement = normalized_gram[:, 0]
    mean_pair_coherence = normalized_gram.abs().mean(dim=2)

    distance_scale = distance.median(dim=1).values.clamp_min(0.3)
    offset = query_position - bs_position[None]
    query_radius = offset[:, :2].norm(dim=1).clamp_min(1e-3)
    query_angle = torch.atan2(offset[:, 1], offset[:, 0])
    rank = torch.arange(
        k, dtype=real_dtype, device=gram.device
    )[None].expand(batch, -1) / max(k - 1, 1)
    log_amplitude = amplitude.clamp_min(tiny).log()
    log_relative_amplitude = (
        log_amplitude - log_amplitude.mean(dim=1, keepdim=True)
    )

    def expand(value: torch.Tensor) -> torch.Tensor:
        return value[:, None].expand(-1, k)

    columns = [
        base_weight.clamp_min(tiny).log(),
        distance.clamp_min(0.05).log(),
        effective_distance / distance_scale[:, None],
        radial_delta / distance_scale[:, None],
        tangent_delta / distance_scale[:, None],
        rank,
        log_relative_amplitude,
        agreement.real,
        agreement.imag,
        agreement.abs(),
        nearest_agreement.real,
        nearest_agreement.imag,
        nearest_agreement.abs(),
        mean_pair_coherence,
        expand(query_radius / 200.0),
        expand(query_angle.sin()),
        expand(query_angle.cos()),
    ]
    features = torch.stack(columns, dim=-1)
    if features.shape[-1] != len(FEATURE_NAMES):
        raise RuntimeError("complex-neighbor feature count mismatch")
    return features, amplitude


class ComplexNeighborGate(nn.Module):
    """Conservative shared MLP for complex interpolation coefficients."""

    def __init__(
        self, feature_dim: int, architecture: str = "mlp16"
    ) -> None:
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError(
                f"unsupported complex gate architecture {architecture}"
            )
        self.architecture = architecture
        if architecture == "mlp16":
            self.network = nn.Sequential(
                nn.Linear(feature_dim, 16),
                nn.SiLU(),
                nn.Linear(16, 3),
            )
            self.input_projection = None
            self.encoder = None
            self.output_projection = None
        else:
            self.network = None
            self.input_projection = nn.Linear(feature_dim, 16)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=16,
                nhead=4,
                dim_feedforward=32,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=1
            )
            self.output_projection = nn.Linear(16, 3)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        output = (
            self.network[-1]
            if self.network is not None
            else self.output_projection
        )
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def raw_parameters(
        self, normalized_features: torch.Tensor
    ) -> torch.Tensor:
        if self.network is not None:
            return self.network(normalized_features)
        hidden = self.input_projection(normalized_features)
        hidden = self.encoder(hidden)
        return self.output_projection(hidden)

    def coefficients(
        self,
        normalized_features: torch.Tensor,
        base_weight: torch.Tensor,
        amplitude: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raw = self.raw_parameters(normalized_features)
        logit_delta = 0.75 * torch.tanh(raw[..., 0])
        phase_delta = 0.20 * torch.tanh(raw[..., 1])
        weight = torch.softmax(
            base_weight.clamp_min(1e-30).log() + logit_delta,
            dim=1,
        )
        target_amplitude = (
            weight * amplitude
        ).sum(dim=1, keepdim=True)
        pooled_amplitude = (
            weight * raw[..., 2]
        ).sum(dim=1, keepdim=True)
        amplitude_factor = torch.exp(
            0.35 * torch.tanh(pooled_amplitude)
        )
        magnitude = (
            weight
            * target_amplitude
            * amplitude_factor
            / amplitude.clamp_min(
                torch.finfo(amplitude.dtype).tiny
            )
        )
        coefficient = magnitude.to(torch.complex64) * torch.exp(
            1j * phase_delta
        )
        return coefficient, {
            "logit_delta": logit_delta,
            "phase_delta": phase_delta,
            "amplitude_factor": amplitude_factor,
            "weight": weight,
        }
