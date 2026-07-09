"""Cross-cutting helpers: config loading, device selection, seeding."""

from .config import load_config, merge_overrides
from .device import get_device
from .seed import set_seed

__all__ = ["load_config", "merge_overrides", "get_device", "set_seed"]
