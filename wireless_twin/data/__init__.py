"""Data layer: everything that reads competition files off disk.

Nothing in here imports torch models or training code — the data layer is a
standalone, cross-platform (Linux/Windows) reader for the competition format:

    RoundX_Setup.json          -> ChannelSpec
    RoundX_Train_Pos.npy       -> (P_train, 3) float
    RoundX_Train_Channel.npy   -> (P_train, M, N, S) complex
    RoundX_Test_Pos.npy        -> (P_test, 3) float
    RoundX_Map.ply             -> (K, 3) point cloud (optional)
"""

from .setup_config import ChannelSpec, load_setup
from .normalization import ChannelScaler
from .channel_dataset import ChannelDataset, load_round
from .map_loader import load_point_cloud
from .augment import augment_positions, AUGMENTED_DIM

__all__ = [
    "ChannelSpec",
    "load_setup",
    "ChannelScaler",
    "ChannelDataset",
    "load_round",
    "load_point_cloud",
    "augment_positions",
    "AUGMENTED_DIM",
]
