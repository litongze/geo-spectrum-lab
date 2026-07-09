"""Optional 3D-Gaussian-Splatting (WRF-GS+) backend — integration scaffold.

This module is the documented seam for reusing the WRF-GS+ 3D-GS model behind
the same :class:`ChannelModel` interface as the pure-PyTorch baseline.  It is
imported lazily by the registry: on a machine where the CUDA rasteriser
submodules are **not** built (e.g. the Windows teammate's laptop) the import
fails and the ``wrfgs`` backend is simply unavailable — the baseline still runs.

Wiring guide
------------
This lean repository does not ship the 3D-GS implementation.  The reference
WRF-GS+ code (``gaussian_renderer/``, ``scene/gaussian_model.py``,
``scene/deform_model.py`` and the CUDA rasteriser submodules) is preserved in
the local ``main`` branch's history and upstream (see
``docs/WRFGS_reference.md``).  To activate this backend:

1. Bring in the 3D-GS sources and build the submodules (``simple-knn``,
   ``diff-gaussian-rasterization``, ``fused-ssim``) — e.g.
   ``git checkout main -- gaussian_renderer scene arguments utils submodules``.
2. Make those packages importable, e.g. add the repo root to ``PYTHONPATH``.
3. Fill in the TODOs below so that ``forward(positions)`` renders the channel
   for each position and returns a complex ``(B, M, N, S)`` tensor (reshaped
   from the renderer's 2-D real/imag output, as in the original ``train.py``).

Until step 3 is done the class raises a clear error instead of silently
producing garbage.
"""

from __future__ import annotations

import torch

# Importing the CUDA rasteriser here means the registry only exposes this
# backend where the extension is actually available.
from diff_gaussian_rasterization import GaussianRasterizer  # noqa: F401

from ..data.setup_config import ChannelSpec
from .base import ChannelModel


class WrfGsModel(ChannelModel):
    """Adapter around the WRF-GS+ 3D-GS renderer (integration required)."""

    def __init__(self, spec: ChannelSpec, **kwargs) -> None:
        super().__init__(spec)
        # TODO: construct GaussianModel + DeformModel and their optimisers here,
        #       mirroring scene/gaussian_model.py and scene/deform_model.py.
        self._ready = False

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        if not self._ready:
            raise NotImplementedError(
                "WrfGsModel is a scaffold: wire the WRF-GS+ renderer into "
                "forward() (see module docstring) before selecting model="
                "'wrfgs'. The 'path_field' backend works out of the box.")
        # TODO: for each position, run the deform + rasteriser and reshape the
        #       (2, H, W) real/imag output into (M, N, S) complex.
        raise NotImplementedError
