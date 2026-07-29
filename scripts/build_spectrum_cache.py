#!/usr/bin/env python3
"""Build reusable train PAS/PDP caches without keeping channels in RAM."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from _bootstrap import ROOT  # noqa: F401
from wireless_twin.data.setup_config import load_setup


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", default="Round1_Map(2)")
    parser.add_argument("--cache-dir", default="cache/teammate_knn_hvp")
    parser.add_argument(
        "--pas-layout",
        choices=("hvp", "pvh", "phv"),
        default="phv",
    )
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    datadir = Path(args.datadir)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    spec = load_setup(datadir / "Round1_Setup.json")
    channels = np.load(
        datadir / "Round1_Train_Channel.npy", mmap_mode="r"
    )
    pas_path = cache_dir / f"train_pas_{args.pas_layout}.npy"
    pdp_path = cache_dir / "train_pdp.npy"
    expected_pas = (
        len(channels),
        spec.n,
        spec.s,
        spec.mh * spec.mv,
    )
    expected_pdp = (len(channels), spec.m, spec.n, spec.s)

    def valid(path: Path, shape: tuple[int, ...]) -> bool:
        if not path.is_file():
            return False
        array = np.load(path, mmap_mode="r")
        return array.shape == shape and array.dtype == np.float32

    if (
        not args.force
        and valid(pas_path, expected_pas)
        and valid(pdp_path, expected_pdp)
    ):
        print(f"[cache] reuse {pas_path} and {pdp_path}", flush=True)
        return

    pas_tmp = pas_path.with_name(f"{pas_path.stem}.tmp.npy")
    pdp_tmp = pdp_path.with_name(f"{pdp_path.stem}.tmp.npy")
    pas = np.lib.format.open_memmap(
        pas_tmp, mode="w+", dtype=np.float32, shape=expected_pas
    )
    pdp = np.lib.format.open_memmap(
        pdp_tmp, mode="w+", dtype=np.float32, shape=expected_pdp
    )
    for start in range(0, len(channels), args.batch_size):
        stop = min(start + args.batch_size, len(channels))
        channel = np.asarray(
            channels[start:stop], dtype=np.complex64
        )
        if args.pas_layout == "hvp":
            spatial = channel.reshape(
                -1,
                spec.mh,
                spec.mv,
                spec.mp,
                spec.n,
                spec.s,
            )
            angular = np.fft.fft2(spatial, axes=(1, 2))
            spectrum = np.square(
                np.abs(angular).astype(np.float32)
            ).sum(axis=3)
        else:
            first, second = (
                (spec.mv, spec.mh)
                if args.pas_layout == "pvh"
                else (spec.mh, spec.mv)
            )
            spatial = channel.reshape(
                -1,
                spec.mp,
                first,
                second,
                spec.n,
                spec.s,
            )
            angular = np.fft.fft2(spatial, axes=(2, 3))
            spectrum = np.square(
                np.abs(angular).astype(np.float32)
            ).sum(axis=1)
        pas[start:stop] = spectrum.reshape(
            -1, spec.mh * spec.mv, spec.n, spec.s
        ).transpose(0, 2, 3, 1)
        delay = np.fft.ifft(channel, axis=-1)
        pdp[start:stop] = np.square(
            np.abs(delay).astype(np.float32)
        )
        print(f"[cache] {stop}/{len(channels)}", flush=True)
    pas.flush()
    pdp.flush()
    del pas, pdp
    os.replace(pas_tmp, pas_path)
    os.replace(pdp_tmp, pdp_path)
    print(
        f"SPECTRUM_CACHE_DONE pas={pas_path} pdp={pdp_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
