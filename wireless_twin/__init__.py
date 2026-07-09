"""Wireless Digital Twin — Physical-AI channel generation framework.

A decoupled implementation for the Huawei "基于 Physical AI 的无线数字孪生信道生成"
challenge.  The package is split into four independent layers so that data,
model, training and evaluation can evolve separately:

    wireless_twin.data        -- read competition data (Setup/Pos/Channel/Map)
    wireless_twin.models       -- Physical-AI models: pos -> MIMO-OFDM channel
    wireless_twin.training     -- losses + training loop
    wireless_twin.evaluation   -- PAS/PDP/NMSE metrics + test-channel prediction

The public entry points live under ``scripts/`` (train / infer / evaluate).
"""

__version__ = "0.1.0"
