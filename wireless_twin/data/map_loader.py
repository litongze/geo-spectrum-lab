"""Load ``RoundX_Map.ply`` environment point clouds.

The map is optional for the baseline model (which is purely position-driven),
but the loader is provided so physics-aware models (3DGS / NeRF backends) can
consume the environment geometry.  We avoid a hard dependency on ``plyfile``:
if it is not installed we fall back to a minimal ASCII/binary PLY reader that
extracts the ``x, y, z`` vertex coordinates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np


def load_point_cloud(path: Union[str, Path]) -> np.ndarray:
    """Return the ``(K, 3)`` float32 vertex coordinates of a PLY point cloud."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"map file not found: {path}")

    try:
        from plyfile import PlyData  # type: ignore

        ply = PlyData.read(str(path))
        v = ply["vertex"]
        return np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
    except ImportError:
        return _read_ply_minimal(path)


def _read_ply_minimal(path: Path) -> np.ndarray:
    """Tiny dependency-free PLY reader for x/y/z vertices (ascii or binary_le)."""
    with path.open("rb") as f:
        magic = f.readline().strip()
        if magic != b"ply":
            raise ValueError(f"not a PLY file: {path}")

        fmt = None
        n_vertices = 0
        props: list[tuple[str, str]] = []
        in_vertex = False
        while True:
            line = f.readline()
            if not line:
                raise ValueError("unexpected EOF in PLY header")
            toks = line.split()
            if toks[0] == b"format":
                fmt = toks[1].decode()
            elif toks[0] == b"element":
                in_vertex = toks[1] == b"vertex"
                if in_vertex:
                    n_vertices = int(toks[2])
            elif toks[0] == b"property" and in_vertex:
                props.append((toks[1].decode(), toks[2].decode()))
            elif toks[0] == b"end_header":
                break

        names = [p[1] for p in props]
        xi, yi, zi = names.index("x"), names.index("y"), names.index("z")

        if fmt == "ascii":
            data = np.loadtxt(f, max_rows=n_vertices)
            data = np.atleast_2d(data)
            return data[:, [xi, yi, zi]].astype(np.float32)

        # binary_little_endian / binary_big_endian
        endian = "<" if "little" in (fmt or "") else ">"
        np_types = {
            "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
            "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
            "ushort": "u2", "uint16": "u2", "short": "i2", "int16": "i2",
            "uint": "u4", "uint32": "u4", "int": "i4", "int32": "i4",
        }
        dtype = np.dtype([(p[1], endian + np_types[p[0]]) for p in props])
        rows = np.frombuffer(f.read(dtype.itemsize * n_vertices), dtype=dtype)
        return np.stack([rows["x"], rows["y"], rows["z"]], axis=-1).astype(np.float32)
