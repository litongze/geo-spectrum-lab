"""Model layer: Physical-AI networks mapping position -> MIMO-OFDM channel.

Every model implements the :class:`ChannelModel` interface (``forward`` returns
a complex channel tensor of shape ``(B, M, N, S)``), so the training and
evaluation layers never depend on a specific architecture.

Backends
--------
* ``path_field`` (default) -- a pure-PyTorch, CUDA-compile-free baseline that
  runs on Linux *and* Windows out of the box.  See :mod:`.path_field`.
* ``wrfgs``                 -- optional 3D-Gaussian-Splatting backend that reuses
  the WRF-GS+ code; requires the CUDA rasteriser submodules.  See
  :mod:`.wrfgs_backend` for the (documented) integration point.
"""

from .base import ChannelModel
from .registry import build_model, register_model, available_models

__all__ = [
    "ChannelModel",
    "build_model",
    "register_model",
    "available_models",
]
