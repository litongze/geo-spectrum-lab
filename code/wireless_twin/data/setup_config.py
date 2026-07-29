"""Parse ``RoundX_Setup.json`` into a typed :class:`ChannelSpec`.

The Setup file (task book, Table 2) describes the geometry of one round::

    P_Train, P_Test          number of train / test positions
    M = MH * MV * MP         base-station (BS) antennas  (rows*cols*pol)
    N = NH * NV * NP         user-equipment (UE) antennas
    S                        number of OFDM sub-carriers
    Q                        real/imag parts (always 2)
    X                        BS position, e.g. [50, 0, 25]
    w                        metric weights (PAS, PDP, NMSE), e.g. [0.4,0.4,0.2]

Keys are matched case-insensitively so the loader is robust to ``M``/``m`` etc.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union


def _get(d: Dict[str, Any], *names: str, default: Any = None) -> Any:
    """Case-insensitive lookup that accepts several candidate key names."""
    lower = {str(k).lower(): v for k, v in d.items()}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return default


@dataclass
class ChannelSpec:
    """Geometry of a single competition round."""

    p_train: int = 0
    p_test: int = 0

    # Base-station antenna array
    m: int = 256
    mh: int = 16
    mv: int = 8
    mp: int = 2

    # User-equipment antenna array
    n: int = 4
    nh: int = 1
    nv: int = 2
    np: int = 2

    s: int = 192          # sub-carriers
    q: int = 2            # real / imag

    bs_position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    metric_weights: List[float] = field(default_factory=lambda: [0.4, 0.4, 0.2])

    raw: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @property
    def channel_shape(self) -> tuple:
        """Per-position complex channel shape ``(M, N, S)``."""
        return (self.m, self.n, self.s)

    @property
    def channel_numel(self) -> int:
        return self.m * self.n * self.s

    def validate(self) -> "ChannelSpec":
        """Sanity-check that the factorised antenna counts match M and N."""
        if self.mh * self.mv * self.mp != self.m:
            raise ValueError(
                f"MH*MV*MP = {self.mh*self.mv*self.mp} != M = {self.m}")
        if self.nh * self.nv * self.np != self.n:
            raise ValueError(
                f"NH*NV*NP = {self.nh*self.nv*self.np} != N = {self.n}")
        return self

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (f"ChannelSpec(P_train={self.p_train}, P_test={self.p_test}, "
                f"M={self.m}[{self.mh}x{self.mv}x{self.mp}], "
                f"N={self.n}[{self.nh}x{self.nv}x{self.np}], S={self.s}, "
                f"BS={self.bs_position}, w={self.metric_weights})")


def load_setup(path: Union[str, Path]) -> ChannelSpec:
    """Read a ``RoundX_Setup.json`` file and return a validated ChannelSpec."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    spec = ChannelSpec(
        p_train=int(_get(raw, "P_Train", "PTrain", "p_train", default=0)),
        p_test=int(_get(raw, "P_Test", "PTest", "p_test", default=0)),
        m=int(_get(raw, "M", default=256)),
        mh=int(_get(raw, "MH", "M_H", default=16)),
        mv=int(_get(raw, "MV", "M_V", default=8)),
        mp=int(_get(raw, "MP", "M_P", default=2)),
        n=int(_get(raw, "N", default=4)),
        nh=int(_get(raw, "NH", "N_H", default=1)),
        nv=int(_get(raw, "NV", "N_V", default=2)),
        np=int(_get(raw, "NP", "N_P", default=2)),
        s=int(_get(raw, "S", default=192)),
        q=int(_get(raw, "Q", default=2)),
        bs_position=_as_float_list(_get(raw, "X", "bs_position", default=[0, 0, 0])),
        metric_weights=_as_float_list(
            _get(raw, "w", "weights", default=[0.4, 0.4, 0.2])),
        raw=raw,
    )
    return spec.validate()


def _as_float_list(value: Any) -> List[float]:
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    if isinstance(value, (int, float)):
        return [float(value)]
    # e.g. a string like "(50, 0, 25)"
    cleaned = str(value).strip().strip("()[]")
    return [float(p) for p in cleaned.split(",") if p.strip()]
