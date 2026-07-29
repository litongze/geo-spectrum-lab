#!/usr/bin/env python3
"""Select spectrum sharpening exponents on clean splits and build a candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from _bootstrap import ROOT  # noqa: F401
from wireless_twin.data.setup_config import load_setup
from wireless_twin.signal import (
    pas_spectrum,
    pas_spectrum_phv,
    pas_spectrum_pvh,
    pdp_spectrum,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", default="Round1_Map(2)")
    parser.add_argument("--panel", default="1890,3716,962,1022,2262")
    parser.add_argument(
        "--val-pattern",
        default="docs/clean_noeps_panel_phv_geom_k16/"
        "split_s{seed}_prediction.npy",
    )
    parser.add_argument(
        "--indices-pattern",
        default="docs/teammate_knn/s{seed}_indices.npy",
    )
    parser.add_argument(
        "--pas-layout", choices=("hvp", "pvh", "phv"), default="phv"
    )
    parser.add_argument(
        "--gamma-grid",
        default="0.5,0.65,0.8,0.9,1,1.1,1.2,1.35,1.5,1.75,2,2.5",
    )
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--test")
    parser.add_argument("--test-outdir")
    parser.add_argument("--outdir", default="docs/spectrum_sharpen")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=5)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    seeds = [int(value) for value in args.panel.split(",") if value.strip()]
    tune_seeds, audit_seed = seeds[:-1], seeds[-1]
    gammas = [
        float(value) for value in args.gamma_grid.split(",") if value.strip()
    ]
    spec = load_setup(Path(args.datadir) / "Round1_Setup.json")
    channels = np.load(
        Path(args.datadir) / "Round1_Train_Channel.npy", mmap_mode="r"
    )
    pas_transform = {
        "hvp": pas_spectrum,
        "pvh": pas_spectrum_pvh,
        "phv": pas_spectrum_phv,
    }[args.pas_layout]
    tiny = torch.finfo(torch.float32).tiny

    def normalize(value: torch.Tensor, dim: int) -> torch.Tensor:
        return value / value.norm(dim=dim, keepdim=True).clamp_min(tiny)

    def spectra(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            normalize(pas_transform(value, spec), -1),
            normalize(pdp_spectrum(value, spec), -1),
        )

    def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        return float((a * b).sum(-1).clamp(-1, 1).mean())

    domain_scores = {
        domain: {gamma: {} for gamma in gammas}
        for domain in ("pas", "pdp")
    }
    for seed in seeds:
        indices = np.load(args.indices_pattern.format(seed=seed)).astype(
            np.int64
        )
        prediction = torch.from_numpy(
            np.array(
                np.load(args.val_pattern.format(seed=seed), mmap_mode="r"),
                copy=True,
            )
        ).to(device)
        truth = torch.from_numpy(
            np.array(channels[indices], copy=True)
        ).to(device)
        pred_pas, pred_pdp = spectra(prediction)
        gt_pas, gt_pdp = spectra(truth)
        for domain, pred, gt in (
            ("pas", pred_pas, gt_pas),
            ("pdp", pred_pdp, gt_pdp),
        ):
            for gamma in gammas:
                sharpened = normalize(pred.clamp_min(tiny).pow(gamma), -1)
                domain_scores[domain][gamma][seed] = cosine(sharpened, gt)
        print(f"[gamma] split {seed} done", flush=True)
        del prediction, truth, pred_pas, pred_pdp, gt_pas, gt_pdp
        torch.cuda.empty_cache()

    selected = {}
    ranked = {}
    for domain in ("pas", "pdp"):
        rows = []
        for gamma in gammas:
            scores = domain_scores[domain][gamma]
            tune = [scores[seed] for seed in tune_seeds]
            rows.append(
                {
                    "gamma": gamma,
                    "tune_median": float(np.median(tune)),
                    "tune_mean": float(np.mean(tune)),
                    "tune_worst": float(np.min(tune)),
                    "audit": scores[audit_seed],
                    "scores": scores,
                }
            )
        rows.sort(
            key=lambda row: (
                row["tune_median"],
                row["tune_mean"],
                row["tune_worst"],
                -abs(row["gamma"] - 1.0),
            ),
            reverse=True,
        )
        selected[domain] = float(rows[0]["gamma"])
        ranked[domain] = rows
        print(
            f"[select {domain}] gamma={rows[0]['gamma']:.3f} "
            f"median={rows[0]['tune_median']:.6f} "
            f"audit={rows[0]['audit']:.6f}",
            flush=True,
        )

    def enforce_pas(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        b, mh, mv, mp, n, s = (
            len(value), spec.mh, spec.mv, spec.mp, spec.n, spec.s
        )
        if args.pas_layout == "hvp":
            shaped = value.reshape(b, mh, mv, mp, n, s)
            angular = torch.fft.fft2(shaped, dim=(1, 2), norm="ortho")
            current = angular.abs().square().sum(3).reshape(
                b, mh * mv, n, s
            ).permute(0, 2, 3, 1)
            gain_shape = (b, mh, mv, 1, n, s)
            axes = (1, 2)
        else:
            spatial = (mv, mh) if args.pas_layout == "pvh" else (mh, mv)
            shaped = value.reshape(b, mp, *spatial, n, s)
            angular = torch.fft.fft2(shaped, dim=(2, 3), norm="ortho")
            current = angular.abs().square().sum(1).reshape(
                b, mh * mv, n, s
            ).permute(0, 2, 3, 1)
            gain_shape = (b, 1, *spatial, n, s)
            axes = (2, 3)
        desired = target * current.norm(
            dim=-1, keepdim=True
        ).clamp_min(tiny)
        gain = torch.sqrt(
            desired / current.clamp_min(tiny)
        ).permute(0, 3, 1, 2).reshape(gain_shape)
        return torch.fft.ifft2(
            angular * gain, dim=axes, norm="ortho"
        ).reshape(value.shape)

    def enforce_pdp(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        delay = torch.fft.ifft(value, dim=-1, norm="ortho")
        current = delay.abs().square()
        desired = target * current.norm(
            dim=-1, keepdim=True
        ).clamp_min(tiny)
        phase = delay / delay.abs().clamp_min(1e-30)
        return torch.fft.fft(
            desired.clamp_min(0).sqrt() * phase,
            dim=-1,
            norm="ortho",
        )

    def sharpen(value: torch.Tensor) -> torch.Tensor:
        target_pas, target_pdp = spectra(value)
        target_pas = normalize(
            target_pas.clamp_min(tiny).pow(selected["pas"]), -1
        )
        target_pdp = normalize(
            target_pdp.clamp_min(tiny).pow(selected["pdp"]), -1
        )
        result = value
        for _ in range(args.iterations):
            result = enforce_pdp(result, target_pdp)
            result = enforce_pas(result, target_pas)
        return result

    realized = {}
    for seed in seeds:
        indices = np.load(args.indices_pattern.format(seed=seed)).astype(
            np.int64
        )
        prediction = torch.from_numpy(
            np.array(
                np.load(args.val_pattern.format(seed=seed), mmap_mode="r"),
                copy=True,
            )
        ).to(device)
        truth = torch.from_numpy(
            np.array(channels[indices], copy=True)
        ).to(device)
        output = sharpen(prediction)
        pred_pas, pred_pdp = spectra(output)
        gt_pas, gt_pdp = spectra(truth)
        c1, c2 = cosine(pred_pas, gt_pas), cosine(pred_pdp, gt_pdp)
        c3 = float(
            (output - truth).abs().square().sum()
            / truth.abs().square().sum().clamp_min(tiny)
        )
        combined = (
            spec.metric_weights[0] * c1
            + spec.metric_weights[1] * c2
            + spec.metric_weights[2] / (1.0 + c3)
        )
        realized[seed] = {
            "C1_PAS": c1,
            "C2_PDP": c2,
            "C3_NMSE": c3,
            "C": combined,
        }
        print(f"[realized] split {seed} C={combined:.6f}", flush=True)
        del prediction, truth, output
        torch.cuda.empty_cache()

    tune_values = [realized[seed]["C"] for seed in tune_seeds]
    result = {
        "panel": seeds,
        "pas_layout": args.pas_layout,
        "selected_gamma": selected,
        "iterations": args.iterations,
        "tune_median_C": float(np.median(tune_values)),
        "tune_mean_C": float(np.mean(tune_values)),
        "tune_worst_C": float(np.min(tune_values)),
        "audit_C": realized[audit_seed]["C"],
        "scores": realized,
        "gamma_grid": ranked,
    }
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if args.test and args.test_outdir:
        source = np.load(args.test, mmap_mode="r")
        output = np.empty(source.shape, dtype=np.complex64)
        for start in range(0, len(source), args.batch_size):
            stop = min(start + args.batch_size, len(source))
            value = torch.from_numpy(
                np.array(source[start:stop], copy=True)
            ).to(device)
            output[start:stop] = sharpen(value).cpu().numpy().astype(
                np.complex64
            )
            print(f"[test] {stop}/{len(source)}", flush=True)
        test_outdir = Path(args.test_outdir)
        test_outdir.mkdir(parents=True, exist_ok=True)
        test_path = test_outdir / "Round1_Test_Channel.npy"
        np.save(test_path, output)
        result["test"] = {
            "path": str(test_path),
            "sha256": sha256(test_path),
            "source": f"{args.test}:{sha256(Path(args.test))}",
            "shape": list(output.shape),
            "dtype": str(output.dtype),
            "finite": bool(np.isfinite(output).all()),
        }
        (test_outdir / "manifest.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(
            f"SHARPEN_TEST_DONE path={test_path} "
            f"sha256={result['test']['sha256']}",
            flush=True,
        )
    print(
        f"SHARPEN_DONE gamma_pas={selected['pas']:.3f} "
        f"gamma_pdp={selected['pdp']:.3f} "
        f"audit_C={realized[audit_seed]['C']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
