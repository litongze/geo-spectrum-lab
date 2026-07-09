"""Training layer: losses + the optimisation loop.

Decoupled from both the model (only the :class:`ChannelModel` interface is
used) and the data (only ``(pos, target_ri)`` batches are consumed).
"""

from .losses import ChannelLoss, nmse
from .trainer import Trainer, TrainConfig

__all__ = ["ChannelLoss", "nmse", "Trainer", "TrainConfig"]
