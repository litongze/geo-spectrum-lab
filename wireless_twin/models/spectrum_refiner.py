"""Small geometry-conditioned residual network for local PAS interpolation."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PasSpectrumRefiner(nn.Module):
    def __init__(self, neighbors: int, hidden: int = 32) -> None:
        super().__init__()
        self.neighbors = neighbors
        input_channels = neighbors + 3
        geometry_dim = neighbors * 3 + 3
        self.geometry = nn.Sequential(
            nn.Linear(geometry_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.conv1 = nn.Conv2d(
            input_channels,
            hidden,
            3,
            padding=1,
            padding_mode="circular",
        )
        self.conv2 = nn.Conv2d(
            hidden,
            hidden,
            3,
            padding=1,
            padding_mode="circular",
        )
        self.output = nn.Conv2d(hidden, 1, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        neighbor_spectra: torch.Tensor,
        distance: torch.Tensor,
        geometry: torch.Tensor,
    ) -> torch.Tensor:
        """Predict unit-L2 PAS.

        ``neighbor_spectra`` has shape ``(B, K, MH, MV)`` and ``distance``
        has shape ``(B, K)``. ``geometry`` is one vector per spectrum slice.
        """
        weight = distance.clamp_min(0.3).pow(-2.0)
        weight = weight / weight.sum(dim=1, keepdim=True)
        baseline = (
            weight[:, :, None, None] * neighbor_spectra
        ).sum(dim=1, keepdim=True)
        mean = neighbor_spectra.mean(dim=1, keepdim=True)
        std = neighbor_spectra.std(dim=1, keepdim=True, unbiased=False)
        inputs = torch.cat(
            [neighbor_spectra, baseline, mean, std], dim=1
        )
        hidden = F.gelu(self.conv1(inputs))
        hidden = hidden + self.geometry(geometry)[:, :, None, None]
        hidden = F.gelu(self.conv2(hidden))
        prediction = F.relu(
            baseline + self.residual_scale * self.output(hidden)
        ).squeeze(1)
        return prediction / prediction.flatten(1).norm(
            dim=1, keepdim=True
        ).clamp_min(torch.finfo(prediction.dtype).tiny)[:, None]


class PdpSpectrumRefiner(nn.Module):
    """Geometry-conditioned PDP refiner over the BS array and delay axes."""

    def __init__(self, neighbors: int, hidden: int = 24) -> None:
        super().__init__()
        self.neighbors = neighbors
        input_channels = neighbors + 3
        geometry_dim = neighbors * 3 + 3
        self.geometry = nn.Sequential(
            nn.Linear(geometry_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.conv1 = nn.Conv3d(
            input_channels,
            hidden,
            kernel_size=(3, 3, 7),
            padding=(1, 1, 3),
            padding_mode="circular",
        )
        self.conv2 = nn.Conv3d(
            hidden,
            hidden,
            kernel_size=(3, 3, 7),
            padding=(1, 1, 3),
            padding_mode="circular",
        )
        self.output = nn.Conv3d(
            hidden, 1, kernel_size=(3, 3, 7), padding=(1, 1, 3)
        )
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        neighbor_spectra: torch.Tensor,
        distance: torch.Tensor,
        geometry: torch.Tensor,
    ) -> torch.Tensor:
        """Predict unit-L2 PDPs.

        ``neighbor_spectra`` is ``(B, K, MH, MV, S)``. The result has shape
        ``(B, MH, MV, S)`` and is normalized independently along delay.
        """
        weight = distance.clamp_min(0.3).pow(-2.0)
        weight = weight / weight.sum(dim=1, keepdim=True)
        baseline = (
            weight[:, :, None, None, None] * neighbor_spectra
        ).sum(dim=1, keepdim=True)
        mean = neighbor_spectra.mean(dim=1, keepdim=True)
        std = neighbor_spectra.std(dim=1, keepdim=True, unbiased=False)
        inputs = torch.cat(
            [neighbor_spectra, baseline, mean, std], dim=1
        )
        hidden = F.gelu(self.conv1(inputs))
        hidden = hidden + self.geometry(geometry)[:, :, None, None, None]
        hidden = F.gelu(self.conv2(hidden))
        prediction = F.relu(
            baseline + self.residual_scale * self.output(hidden)
        ).squeeze(1)
        return prediction / prediction.norm(
            dim=-1, keepdim=True
        ).clamp_min(torch.finfo(prediction.dtype).tiny)


class PdpContentAttention(nn.Module):
    """Select PDP neighbors from their delay shape and relative geometry."""

    def __init__(
        self,
        content_dim: int,
        geometry_dim: int,
        slice_dim: int,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        token_dim = 48
        condition_dim = 32
        self.content = nn.Sequential(
            nn.Linear(content_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, token_dim),
        )
        self.geometry = nn.Sequential(
            nn.Linear(geometry_dim, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )
        self.slice = nn.Sequential(
            nn.Linear(slice_dim, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )
        self.score = nn.Sequential(
            nn.Linear(
                token_dim * 3 + condition_dim * 2,
                hidden,
            ),
            nn.SiLU(),
            nn.Linear(hidden, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)
        self.idw_power = nn.Parameter(torch.tensor(2.0))

    def forward(
        self,
        content: torch.Tensor,
        geometry: torch.Tensor,
        slice_features: torch.Tensor,
        log_distance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return convex neighbor weights and their unnormalized scores."""
        token = self.content(content)
        distance_weight = torch.softmax(
            -self.idw_power.clamp(0.25, 8.0) * log_distance,
            dim=1,
        )
        context = (distance_weight[..., None] * token).sum(
            dim=1, keepdim=True
        )
        context = context.expand_as(token)
        geometry_token = self.geometry(geometry)
        slice_token = self.slice(slice_features)[:, None].expand(
            -1, token.shape[1], -1
        )
        features = torch.cat(
            [
                token,
                context,
                (token - context).abs(),
                geometry_token,
                slice_token,
            ],
            dim=-1,
        )
        score = self.score(features).squeeze(-1)
        score = score - self.idw_power.clamp(0.25, 8.0) * log_distance
        return torch.softmax(score, dim=1), score
