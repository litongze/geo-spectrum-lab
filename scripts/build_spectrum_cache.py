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
        choices=("hvp", "hpv", "vhp", "vph", "phv", "pvh"),
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

    build_pas = args.force or not valid(pas_path, expected_pas)
    build_pdp = args.force or not valid(pdp_path, expected_pdp)
    if not build_pas and not build_pdp:
        print(f"[cache] reuse {pas_path} and {pdp_path}", flush=True)
        return

    pas_tmp = pas_path.with_name(f"{pas_path.stem}.tmp.npy")
    pdp_tmp = pdp_path.with_name(f"{pdp_path.stem}.tmp.npy")
    pas = (
        np.lib.format.open_memmap(
            pas_tmp, mode="w+", dtype=np.float32, shape=expected_pas
        )
        if build_pas
        else None
    )
    pdp = (
        np.lib.format.open_memmap(
            pdp_tmp, mode="w+", dtype=np.float32, shape=expected_pdp
        )
        if build_pdp
        else None
    )
    layout_size = {
        "h": spec.mh,
        "v": spec.mv,
        "p": spec.mp,
    }
    for start in range(0, len(channels), args.batch_size):
        stop = min(start + args.batch_size, len(channels))
        channel = np.asarray(
            channels[start:stop], dtype=np.complex64
        )
        if pas is not None:
            spatial = channel.reshape(
                -1,
                *[layout_size[name] for name in args.pas_layout],
                spec.n,
                spec.s,
            )
            axis = {
                name: 1 + args.pas_layout.index(name)
                for name in ("h", "v", "p")
            }
            canonical = spatial.transpose(
                0, axis["h"], axis["v"], axis["p"], 4, 5
            )
            angular = np.fft.fft2(canonical, axes=(1, 2))
            spectrum = np.square(
                np.abs(angular).astype(np.float32)
            ).sum(axis=3)
            pas[start:stop] = spectrum.reshape(
                -1, spec.mh * spec.mv, spec.n, spec.s
            ).transpose(0, 2, 3, 1)
        if pdp is not None:
            delay = np.fft.ifft(channel, axis=-1)
            pdp[start:stop] = np.square(
                np.abs(delay).astype(np.float32)
            )
        print(f"[cache] {stop}/{len(channels)}", flush=True)
    if pas is not None:
        pas.flush()
        del pas
        os.replace(pas_tmp, pas_path)
    if pdp is not None:
        pdp.flush()
        del pdp
        os.replace(pdp_tmp, pdp_path)
    print(
        f"SPECTRUM_CACHE_DONE pas={pas_path} pdp={pdp_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
