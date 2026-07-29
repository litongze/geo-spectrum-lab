#!/usr/bin/env python3
"""Rescale a complex submission without changing its PAS or PDP shape."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rms(path: Path, chunk_size: int) -> tuple[tuple[int, ...], np.dtype, float]:
    array = np.load(path, mmap_mode="r")
    energy = 0.0
    count = 0
    for start in range(0, len(array), chunk_size):
        chunk = np.asarray(array[start : start + chunk_size])
        energy += float(
            np.square(np.abs(chunk), dtype=np.float64).sum()
        )
        count += chunk.size
    return tuple(array.shape), array.dtype, float(
        np.sqrt(energy / max(count, 1))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-rms", type=float, required=True)
    parser.add_argument("--phase", type=float, default=0.0)
    parser.add_argument("--chunk-size", type=int, default=10)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if args.target_rms <= 0.0:
        raise ValueError("--target-rms must be positive")
    shape, dtype, input_rms = rms(input_path, args.chunk_size)
    if not np.issubdtype(dtype, np.complexfloating):
        raise TypeError(f"input must be complex, got {dtype}")
    factor = (
        args.target_rms
        / max(input_rms, np.finfo(np.float64).tiny)
        * np.exp(1j * args.phase)
    )
    source = np.load(input_path, mmap_mode="r")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.complex64,
        shape=shape,
    )
    for start in range(0, len(source), args.chunk_size):
        output[start : start + args.chunk_size] = (
            source[start : start + args.chunk_size] * factor
        ).astype(np.complex64)
    output.flush()
    del output
    output_shape, output_dtype, output_rms = rms(
        output_path, args.chunk_size
    )
    manifest = {
        "input": f"{input_path}:{sha256(input_path)}",
        "output": str(output_path),
        "shape": list(output_shape),
        "dtype": str(output_dtype),
        "input_rms": input_rms,
        "target_rms": args.target_rms,
        "output_rms": output_rms,
        "phase": args.phase,
        "scale_factor": abs(factor),
        "sha256": sha256(output_path),
    }
    manifest_path = output_path.parent / "rescale_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"RESCALE_DONE output={output_path} rms={output_rms:.9e} "
        f"sha256={manifest['sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
