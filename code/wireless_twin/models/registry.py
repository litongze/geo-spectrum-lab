"""Name -> constructor registry for channel-generation backends."""

from __future__ import annotations

from typing import Callable, Dict, List

from ..data.setup_config import ChannelSpec
from .base import ChannelModel

_REGISTRY: Dict[str, Callable[..., ChannelModel]] = {}


def register_model(name: str) -> Callable:
    def _wrap(cls):
        _REGISTRY[name] = cls
        return cls
    return _wrap


def available_models() -> List[str]:
    return sorted(_REGISTRY)


def build_model(name: str, spec: ChannelSpec, **kwargs) -> ChannelModel:
    if name not in _REGISTRY:
        raise KeyError(f"unknown model '{name}'. Available: {available_models()}")
    return _REGISTRY[name](spec, **kwargs)


from .path_field import PathFieldModel  # noqa: E402
from .transformer_model import AngleDelayPerceiverModel  # noqa: E402

register_model("path_field")(PathFieldModel)
register_model("transformer")(AngleDelayPerceiverModel)
register_model("angle_delay_transformer")(AngleDelayPerceiverModel)

try:  # pragma: no cover - optional CUDA extension
    from .wrfgs_backend import WrfGsModel  # type: ignore # noqa: E402
    register_model("wrfgs")(WrfGsModel)
except Exception:
    pass
