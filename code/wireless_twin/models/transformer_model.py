"""Perceiver-style map-conditioned Transformer for MIMO-OFDM generation.

The model predicts a complex channel in the BS-angle / delay domain and maps it
back to the antenna / sub-carrier domain with unitary FFTs.  A dynamic low-rank
decoder generates position-dependent angle, UE and delay factors; unlike the
PathField baseline, none of these factors are globally shared across receiver
positions.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from ..data.setup_config import ChannelSpec
from .base import ChannelModel
from .encodings import FourierFeatures
from .scene_encoder import (
    SCENE_FEATURE_DIM,
    MultiheadAttentionBlock,
    SceneEncoder,
    ScenePreprocessConfig,
    preprocess_point_cloud,
)


def _axis_fourier(coords: torch.Tensor, n_freqs: int = 6) -> torch.Tensor:
    """Fourier features for static decoder-axis coordinates."""
    freqs = (2.0 ** torch.arange(n_freqs, dtype=coords.dtype)) * torch.pi
    scaled = coords.unsqueeze(-1) * freqs
    return torch.cat(
        [coords, scaled.sin().flatten(1), scaled.cos().flatten(1)], dim=-1)


def _grid_coords(*sizes: int) -> torch.Tensor:
    axes = [
        torch.linspace(-1.0, 1.0, steps=s) if s > 1 else torch.zeros(1)
        for s in sizes
    ]
    mesh = torch.meshgrid(*axes, indexing="ij")
    return torch.stack(mesh, dim=-1).reshape(-1, len(sizes))


class PerceiverBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.cross = MultiheadAttentionBlock(
            d_model, n_heads, d_ff, dropout)
        self.self_attn = MultiheadAttentionBlock(
            d_model, n_heads, d_ff, dropout)

    def forward(self, latents: torch.Tensor, scene: torch.Tensor) -> torch.Tensor:
        latents = self.cross(latents, scene)
        return self.self_attn(latents, latents)


class AxisFactorDecoder(nn.Module):
    """Decode dynamic complex rank factors for one physical output axis."""

    def __init__(
        self,
        coords: torch.Tensor,
        d_model: int,
        n_heads: int,
        rank: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        self_attention: bool = True,
        coord_freqs: int = 6,
    ) -> None:
        super().__init__()
        coord_features = _axis_fourier(coords, coord_freqs)
        self.register_buffer("coord_features", coord_features, persistent=False)
        self.coord_proj = nn.Linear(coord_features.shape[-1], d_model)
        self.learned = nn.Parameter(
            torch.randn(1, coords.shape[0], d_model) / d_model**0.5)
        self.cross_blocks = nn.ModuleList(
            [
                MultiheadAttentionBlock(
                    d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ]
        )
        self.self_blocks = (
            nn.ModuleList(
                [
                    MultiheadAttentionBlock(
                        d_model, n_heads, d_ff, dropout)
                    for _ in range(n_layers)
                ]
            )
            if self_attention
            else None
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2 * rank)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        b = latents.shape[0]
        q = self.learned + self.coord_proj(self.coord_features).unsqueeze(0)
        q = q.expand(b, -1, -1)
        for i, cross in enumerate(self.cross_blocks):
            q = cross(q, latents)
            if self.self_blocks is not None:
                q = self.self_blocks[i](q, q)
        return self.head(self.norm(q))


class AngleDelayPerceiverModel(ChannelModel):
    """Map-conditioned Perceiver with a dynamic angle-delay decoder."""

    def __init__(
        self,
        spec: ChannelSpec,
        d_model: int = 256,
        n_heads: int = 8,
        n_freqs: int = 10,
        n_latents: int = 128,
        perceiver_layers: int = 4,
        scene_layers: int = 4,
        scene_inducing: int = 128,
        n_scene_latents: int = 256,
        decoder_layers: int = 2,
        decoder_self_attention: bool = True,
        channel_rank: int = 48,
        ff_mult: int = 4,
        dropout: float = 0.1,
        n_map_points: int = 16_384,
        n_scene_tokens: int = 512,
        scene_knn: int = 16,
        scene_seed: int = 0,
        cache_scene_in_eval: bool = True,
        use_power_head: bool = False,
    ) -> None:
        super().__init__(spec)
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if channel_rank <= 0:
            raise ValueError("channel_rank must be positive")

        self.d_model = d_model
        self.channel_rank = channel_rank
        self.cache_scene_in_eval = cache_scene_in_eval
        self.use_power_head = bool(use_power_head)
        self.scene_preprocess = ScenePreprocessConfig(
            n_input_points=n_map_points,
            n_scene_tokens=n_scene_tokens,
            knn=scene_knn,
            seed=scene_seed,
        )
        d_ff = ff_mult * d_model

        self.scene_encoder = SceneEncoder(
            input_dim=SCENE_FEATURE_DIM,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=scene_layers,
            n_inducing=scene_inducing,
            n_output_tokens=n_scene_latents,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.fallback_scene = nn.Parameter(
            torch.randn(1, n_scene_latents, d_model) / d_model**0.5)

        self.position_encoder = FourierFeatures(
            in_dim=6, n_freqs=n_freqs, include_input=True)
        self.position_mlp = nn.Sequential(
            nn.Linear(self.position_encoder.out_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )
        self.latents = nn.Parameter(
            torch.randn(1, n_latents, d_model) / d_model**0.5)
        self.perceiver = nn.ModuleList(
            [
                PerceiverBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(perceiver_layers)
            ]
        )
        self.latent_norm = nn.LayerNorm(d_model)

        angle_coords = _grid_coords(spec.mh, spec.mv, spec.mp)
        ue_coords = _grid_coords(spec.nh, spec.nv, spec.np)
        delay_coords = _grid_coords(spec.s)
        decoder_kwargs = dict(
            d_model=d_model,
            n_heads=n_heads,
            rank=channel_rank,
            n_layers=decoder_layers,
            d_ff=d_ff,
            dropout=dropout,
            self_attention=decoder_self_attention,
        )
        self.angle_decoder = AxisFactorDecoder(angle_coords, **decoder_kwargs)
        self.ue_decoder = AxisFactorDecoder(ue_coords, **decoder_kwargs)
        self.delay_decoder = AxisFactorDecoder(delay_coords, **decoder_kwargs)
        self.gain_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2 * channel_rank),
        )
        nn.init.normal_(self.gain_head[-1].weight, std=0.01)
        nn.init.zeros_(self.gain_head[-1].bias)
        if self.use_power_head:
            self.log_power_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1),
            )
            nn.init.zeros_(self.log_power_head[-1].weight)
            nn.init.zeros_(self.log_power_head[-1].bias)
        else:
            self.log_power_head = None

        self.register_buffer(
            "_scene_features", torch.empty(0, SCENE_FEATURE_DIM),
            persistent=False)
        self.register_buffer(
            "_bs_position_norm", torch.zeros(3), persistent=False)
        self._cached_scene_tokens: Optional[torch.Tensor] = None
        self._last_angle_delay: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Scene lifecycle
    def set_scene(
        self,
        point_cloud: np.ndarray,
        pos_mean: Sequence[float],
        pos_std: Sequence[float],
        bs_position: Optional[Sequence[float]] = None,
    ) -> None:
        """Preprocess and attach the map for this competition round."""
        features, bs_norm = preprocess_point_cloud(
            point_cloud,
            pos_mean=pos_mean,
            pos_std=pos_std,
            bs_position=(self.spec.bs_position if bs_position is None else bs_position),
            config=self.scene_preprocess,
        )
        device = self._bs_position_norm.device
        self._scene_features = torch.from_numpy(features).to(device)
        self._bs_position_norm = torch.from_numpy(bs_norm).to(device)
        self._cached_scene_tokens = None

    def export_scene_state(self) -> Dict[str, torch.Tensor]:
        """Small CPU payload stored in checkpoint metadata for inference."""
        return {
            "scene_features": self._scene_features.detach().cpu(),
            "bs_position_norm": self._bs_position_norm.detach().cpu(),
        }

    def load_scene_state(self, state: Optional[Dict]) -> None:
        if not state:
            return
        device = self._bs_position_norm.device
        self._scene_features = torch.as_tensor(
            state["scene_features"], dtype=torch.float32, device=device)
        self._bs_position_norm = torch.as_tensor(
            state["bs_position_norm"], dtype=torch.float32, device=device)
        self._cached_scene_tokens = None

    def train(self, mode: bool = True):
        if mode:
            self._cached_scene_tokens = None
        return super().train(mode)

    def _encode_scene(self) -> torch.Tensor:
        if (
            not self.training
            and self.cache_scene_in_eval
            and self._cached_scene_tokens is not None
        ):
            return self._cached_scene_tokens

        if self._scene_features.numel() == 0:
            tokens = self.fallback_scene
        else:
            tokens = self.scene_encoder(self._scene_features.unsqueeze(0))
        if not self.training and self.cache_scene_in_eval:
            self._cached_scene_tokens = tokens.detach()
        return tokens

    # ------------------------------------------------------------------
    @staticmethod
    def _complex_factor(ri: torch.Tensor) -> torch.Tensor:
        real, imag = ri.chunk(2, dim=-1)
        # ``torch.complex`` does not accept bf16; cast here while allowing the
        # attention trunk itself to run under bf16 autocast.
        return torch.complex(real.float(), imag.float())

    @staticmethod
    def _rms_normalise(factor: torch.Tensor, axis: int) -> torch.Tensor:
        power = factor.abs().square().mean(dim=axis, keepdim=True)
        return factor / torch.sqrt(power + 1e-6)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        b = positions.shape[0]
        bs = self._bs_position_norm.to(
            device=positions.device, dtype=positions.dtype)
        rel = positions - bs.unsqueeze(0)
        # Raw coordinates are included in the Fourier encoding. Clamp only the
        # raw context to prevent an abnormal normalisation scale from overflowing
        # FP16; sinusoidal features still carry high-frequency variation.
        pos_input = torch.cat([positions, rel], dim=-1).clamp(-32.0, 32.0)
        pos_context = self.position_mlp(self.position_encoder(pos_input))

        scene = self._encode_scene().to(
            device=positions.device, dtype=pos_context.dtype)
        scene = scene.expand(b, -1, -1)
        latents = self.latents.to(pos_context.dtype).expand(b, -1, -1)
        latents = latents + pos_context.unsqueeze(1)
        for block in self.perceiver:
            latents = block(latents, scene)
        latents = self.latent_norm(latents)
        pooled = latents.mean(dim=1) + pos_context

        angle = self._complex_factor(self.angle_decoder(latents))
        ue = self._complex_factor(self.ue_decoder(latents))
        delay = self._complex_factor(self.delay_decoder(latents))
        gain = self._complex_factor(self.gain_head(pooled))

        angle = self._rms_normalise(angle, axis=1)
        ue = self._rms_normalise(ue, axis=1)
        delay = self._rms_normalise(delay, axis=1)
        gain = gain / (self.channel_rank**0.5)

        # (B,A,R) x (B,N,R) x (B,S,R) x (B,R) -> (B,A,N,S)
        h_ad = torch.einsum(
            "bar,bnr,bsr,br->bans", angle, ue, delay, gain)
        h_ad = h_ad.reshape(
            b, self.spec.mh, self.spec.mv, self.spec.mp,
            self.spec.n, self.spec.s)
        self._last_angle_delay = h_ad

        # Angle -> physical BS array, delay -> OFDM frequency.  Orthonormal
        # transforms preserve NMSE exactly between the two domains.
        h_space_delay = torch.fft.ifft2(
            h_ad, dim=(1, 2), norm="ortho")
        h_space_freq = torch.fft.fft(
            h_space_delay, dim=-1, norm="ortho")
        h = h_space_freq.reshape(
            b, self.spec.m, self.spec.n, self.spec.s)
        if self.log_power_head is None:
            return h
        shape_rms = torch.sqrt(
            h.abs().square().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12))
        log_power = self.log_power_head(pooled).float().view(b, 1, 1, 1)
        return (h / shape_rms) * torch.exp(0.5 * log_power)

    def auxiliary_losses(self) -> Dict[str, torch.Tensor]:
        """Differentiable regularisers consumed by the generic trainer."""
        if self._last_angle_delay is None:
            return {}
        return {"angle_delay_l1": self._last_angle_delay.abs().mean()}
