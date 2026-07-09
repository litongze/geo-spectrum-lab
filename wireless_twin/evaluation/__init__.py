"""Evaluation layer: scoring metrics + test-channel prediction.

* :mod:`.metrics`   -- C1 (PAS), C2 (PDP), C3 (NMSE) and the combined score C.
* :mod:`.predictor` -- run a trained model over test positions and write
  ``RoundX_Test_Channel.npy`` in the exact submission format.
"""

from .metrics import (
    channel_nmse,
    pas_accuracy,
    pdp_accuracy,
    competition_score,
    evaluate_channels,
)
from .predictor import predict_test_channels, load_model_from_checkpoint

__all__ = [
    "channel_nmse",
    "pas_accuracy",
    "pdp_accuracy",
    "competition_score",
    "evaluate_channels",
    "predict_test_channels",
    "load_model_from_checkpoint",
]
