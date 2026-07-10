"""A tiny name -> constructor registry so models are selected by config string.

This is the seam that lets ``configs/*.yaml`` pick a backend (``path_field``,
``wrfgs``, a future NeRF, ...) without the training code importing any concrete
model class.
"""

from __future__ import annotations

from typing import Callable, Dict, List

from ..data.setup_config import ChannelSpec
from .base import ChannelModel

_REGISTRY: Dict[str, Callable[..., ChannelModel]] = {}


def register_model(name: str) -> Callable:
    """Class decorator that registers a :class:`ChannelModel` under ``name``."""

    def _wrap(cls):
        _REGISTRY[name] = cls
        return cls

    return _wrap


def available_models() -> List[str]:
    return sorted(_REGISTRY)


def build_model(name: str, spec: ChannelSpec, **kwargs) -> ChannelModel:
    """Instantiate a registered model by name."""
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown model '{name}'. Available: {available_models()}")
    return _REGISTRY[name](spec, **kwargs)


# --- register built-in backends -----------------------------------------
from .path_field import PathFieldModel  # noqa: E402
from .scatter_field import ScatterFieldModel  # noqa: E402
from .nerf2_field import NeRF2Field  # noqa: E402

register_model("path_field")(PathFieldModel)
register_model("scatter_field")(ScatterFieldModel)
register_model("nerf2")(NeRF2Field)

# The 3D-GS backend is optional (needs CUDA rasteriser); register it lazily so
# importing the package never fails when the submodules aren't built.
try:  # pragma: no cover - depends on optional CUDA extensions
    from .wrfgs_backend import WrfGsModel  # noqa: E402

    register_model("wrfgs")(WrfGsModel)
except Exception:  # noqa: BLE001 - any import/CUDA error just disables backend
    pass
