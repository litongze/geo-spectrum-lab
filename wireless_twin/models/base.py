"""Abstract model interface shared by every backend.

Keeping a narrow contract here is what decouples *models* from *training* and
*evaluation*: the trainer only ever calls ``model(positions) -> complex H`` and
knows nothing about MLPs, Gaussians or rasterisers.
"""

from __future__ import annotations

import abc

import torch
import torch.nn as nn

from ..data.setup_config import ChannelSpec


class ChannelModel(nn.Module, abc.ABC):
    """Base class for ``H = f(x)`` channel generators.

    Subclasses must implement :meth:`forward` returning a *complex* tensor of
    shape ``(B, M, N, S)`` for a batch of ``(B, 3)`` positions.
    """

    def __init__(self, spec: ChannelSpec) -> None:
        super().__init__()
        self.spec = spec

    @abc.abstractmethod
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Map positions ``(B, 3)`` to complex channels ``(B, M, N, S)``."""
        raise NotImplementedError

    # Convenience: flat real/imag view used by MSE-style losses -----------
    def forward_ri(self, positions: torch.Tensor) -> torch.Tensor:
        """Return channels as flat real/imag vectors ``(B, M*N*S*2)``."""
        h = self.forward(positions)                      # (B, M, N, S) complex
        ri = torch.stack([h.real, h.imag], dim=-1)       # (B, M, N, S, 2)
        return ri.reshape(positions.shape[0], -1)
