"""Device selection that works the same on Linux and Windows."""

from __future__ import annotations

from typing import Optional

import torch


def get_device(preference: Optional[str] = None) -> torch.device:
    """Return a torch device.

    ``preference`` may be ``"cuda"``, ``"cpu"``, ``"mps"`` or ``None`` (auto).
    Falls back to CPU when the requested accelerator is unavailable.
    """
    if preference:
        pref = preference.lower()
        if pref == "cuda" and not torch.cuda.is_available():
            print("[device] CUDA requested but unavailable -> using CPU")
            return torch.device("cpu")
        if pref == "mps" and not getattr(torch.backends, "mps", None):
            return torch.device("cpu")
        return torch.device(pref)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
