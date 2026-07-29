#!/usr/bin/env python3
"""Strict no-eps validation on the clean representative-split panel.

The first N-1 splits select common PAS/PDP blend parameters and output scale.
The final split is an untouched audit split. Every checkpoint path is explicit,
and checkpoint metadata must match the split being evaluated.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)

from score_holdout import reproduce_val_indices
from wireless_twin.data.map_loader import load_point_cloud
from wireless_twin.data.setup_config import load_setup
from wireless_twin.evaluation.metrics import evaluate_channels
from wireless_twin.evaluation.predictor import (
    load_model_from_checkpoint,
    predict_test_channels,
)
from wireless_twin.models.raytrace2 import build_heightmap


class AttnWide(nn.Module):
    def __init__(self, nf: int = 6):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(nf, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.idw_w = nn.Parameter(torch.tensor(2.0))

    def forward(self, feats, logd):
        return torch.softmax(
            self.mlp(feats).squeeze(-1) - self.idw_w * logd, dim=1
        )


@dataclass
class SplitBundle:
    seed: int
    gt: torch.Tensor
    gt_pas: torch.Tensor
    gt_pdp: torch.Tensor
    base_pas: torch.Tensor
    base_pdp: torch.Tensor
    e35_pas: torch.Tensor
    e35_pdp: torch.Tensor
    arm_pas: torch.Tensor
    arm_pdp: torch.Tensor
    href: torch.Tensor


def parse_ints(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def parse_floats(raw: str) -> list[float]:
    return [float(x) for x in raw.split(",") if x.strip()]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datadir", default="Round1_Map(2)")
    ap.add_argument("--panel", default="1890,3716,962,1022,2262")
    ap.add_argument("--ckpt-root", default="checkpoints/clean_panel")
    ap.add_argument(
        "--pas-arm-root",
        default=None,
        help="optional checkpoint root for PAS arms (base/e35 still use ckpt-root)",
    )
    ap.add_argument(
        "--pdp-arm-root",
        default=None,
        help="optional checkpoint root for PDP arms (base/e35 still use ckpt-root)",
    )
    ap.add_argument("--pas-arm-k", type=int, default=32)
    ap.add_argument("--pdp-arm-k", type=int, default=32)
    ap.add_argument("--pas-arm-temperature", type=float, default=1.0)
    ap.add_argument("--pdp-arm-temperature", type=float, default=1.0)
    ap.add_argument(
        "--pas-layout",
        choices=["legacy_hvp", "pvh", "phv"],
        default="legacy_hvp",
    )
    ap.add_argument("--outdir", default="docs/clean_noeps_panel")
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument(
        "--validation-neighbor-bank-fraction",
        type=float,
        default=1.0,
        help=(
            "deterministically subsample the legal holdout neighbor bank for "
            "density ablations; test inference is unaffected"
        ),
    )
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--m-grid", default="0,0.25,0.5,1,2,4")
    ap.add_argument("--alpha-grid", default="0,0.25,0.5,0.75,1")
    ap.add_argument(
        "--scale-grid",
        default="2.5e-7,5e-7,7.5e-7,1e-6,1.5e-6,2e-6,3e-6,4e-6,"
        "6e-6,8e-6,1e-5,1.5e-5,2e-5",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--test-outdir",
        default=None,
        help="optionally build a test candidate with the selected configuration",
    )
    ap.add_argument(
        "--save-split-predictions",
        action="store_true",
        help="save every clean split prediction for downstream fusion audits",
    )
    ap.add_argument(
        "--test-full-neighbor-bank",
        action="store_true",
        help=(
            "for test inference, query all labeled training points instead of "
            "artificially retaining each validation split's exclusion"
        ),
    )
    ap.add_argument("--href-seed", type=int, default=1890)
    args = ap.parse_args()
    if not 0 < args.validation_neighbor_bank_fraction <= 1:
        raise ValueError("validation neighbor bank fraction must be in (0, 1]")

    seeds = parse_ints(args.panel)
    if len(seeds) < 2:
        raise ValueError("panel needs at least one tuning split and one audit split")
    tune_seeds, audit_seed = seeds[:-1], seeds[-1]
    m_grid = parse_floats(args.m_grid)
    alpha_grid = parse_floats(args.alpha_grid)
    scale_grid = parse_floats(args.scale_grid)
    dev = torch.device(args.device)
    tiny = torch.finfo(torch.float32).tiny

    dd = Path(args.datadir)
    ckpt_root = Path(args.ckpt_root)
    pas_arm_root = Path(args.pas_arm_root or args.ckpt_root)
    pdp_arm_root = Path(args.pdp_arm_root or args.ckpt_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    setup = json.loads((dd / "Round1_Setup.json").read_text(encoding="utf-8"))
    mh, mv, mp = setup["M_H"], setup["M_V"], setup["M_P"]
    n, s = setup["N"], setup["S"]
    weights = [float(x) for x in setup["w"]]

    pos = np.load(dd / "Round1_Train_Pos.npy").astype(np.float32)
    channels = np.load(dd / "Round1_Train_Channel.npy", mmap_mode="r")
    points = load_point_cloud(dd / "Round1_Map.ply")
    hm, x0, y0, res = build_heightmap(points)
    gx = np.clip(np.floor((pos[:, 0] - x0) / res).astype(int), 0, hm.shape[0] - 1)
    gy = np.clip(np.floor((pos[:, 1] - y0) / res).astype(int), 0, hm.shape[1] - 1)
    indoor = (hm[gx, gy] > 2.0).astype(np.float32)

    def pas(x: torch.Tensor) -> torch.Tensor:
        if args.pas_layout in {"pvh", "phv"}:
            spatial = (mv, mh) if args.pas_layout == "pvh" else (mh, mv)
            a = x.reshape(-1, mp, *spatial, n, s)
            return (
                torch.fft.fft2(a, dim=(2, 3), norm="ortho")
                .abs()
                .square()
                .sum(1)
                .reshape(-1, mh * mv, n, s)
            )
        a = x.reshape(-1, mh, mv, mp, n, s)
        return (
            torch.fft.fft2(a, dim=(1, 2), norm="ortho")
            .abs()
            .square()
            .sum(3)
            .reshape(-1, mh * mv, n, s)
        )

    def pdp(x: torch.Tensor) -> torch.Tensor:
        return torch.fft.ifft(x, dim=-1, norm="ortho").abs().square()

    def nrm(x: torch.Tensor, dim: int) -> torch.Tensor:
        return x / x.norm(dim=dim, keepdim=True).clamp_min(1e-30)

    def cos_last(a: torch.Tensor, b: torch.Tensor, dim: int) -> float:
        den = a.norm(dim=dim) * b.norm(dim=dim)
        num = (a * b).sum(dim=dim)
        return float((num / den.clamp_min(tiny)).clamp(-1, 1).mean())

    print("[clean-noeps] precomputing labeled spectra", flush=True)
    all_pas = torch.empty(len(pos), mh * mv, n, s, device=dev)
    all_pdp = torch.empty(len(pos), mh * mv * mp, n, s, device=dev)
    for start in range(0, len(pos), 100):
        stop = min(start + 100, len(pos))
        h = torch.as_tensor(
            np.array(channels[start:stop], copy=True),
            dtype=torch.complex64,
            device=dev,
        )
        all_pas[start:stop] = nrm(pas(h), 1)
        all_pdp[start:stop] = nrm(pdp(h), -1)
        del h

    def split_paths(seed: int) -> dict[str, Path]:
        root = ckpt_root / f"s{seed}"
        pas_root = pas_arm_root / f"s{seed}"
        pdp_root = pdp_arm_root / f"s{seed}"
        return {
            "base": root / "base.pt",
            "e35": root / "base_e35.pt",
            "arm_pas": pas_root / (
                f"nbrattn_clean_k{args.pas_arm_k}_pas_"
                f"k{args.pas_arm_k}s0.pt"
            ),
            "arm_pdp": pdp_root / (
                f"nbrattn_clean_k{args.pdp_arm_k}_pdp_"
                f"k{args.pdp_arm_k}s0.pt"
            ),
        }

    manifest: dict[str, dict[str, str]] = {}
    for seed in seeds:
        paths = split_paths(seed)
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        manifest[str(seed)] = {
            name: f"{path}:{sha256(path)}" for name, path in paths.items()
        }

    def require_base(path: Path, seed: int) -> dict:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        meta = payload.get("meta", {})
        if int(meta.get("split_seed", -1)) != seed:
            raise ValueError(f"{path}: split_seed={meta.get('split_seed')} != {seed}")
        if float(meta.get("val_fraction", -1)) != args.val_fraction:
            raise ValueError(
                f"{path}: val_fraction={meta.get('val_fraction')} "
                f"!= {args.val_fraction}"
            )
        return meta

    def load_arm(path: Path, seed: int, expected_tag: str) -> tuple[nn.Module, dict]:
        payload = torch.load(path, map_location=dev, weights_only=False)
        meta = payload.get("meta", {})
        if not meta.get("clean_holdout"):
            raise ValueError(f"{path}: clean_holdout metadata missing")
        if int(meta.get("split_seed", -1)) != seed:
            raise ValueError(f"{path}: split_seed mismatch")
        if str(meta.get("tag", "")).split("_")[0] != expected_tag:
            raise ValueError(f"{path}: tag={meta.get('tag')} != {expected_tag}")
        expected_layout = args.pas_layout if expected_tag == "pas" else None
        found_layout = meta.get("pas_layout", "legacy_hvp")
        if expected_layout is not None and found_layout != expected_layout:
            raise ValueError(
                f"{path}: pas_layout={found_layout} != {expected_layout}"
            )
        model = AttnWide(int(meta.get("feature_dim", 6))).to(dev)
        model.load_state_dict(payload["model_state"])
        model.eval()
        return model, meta

    def build_feats(
        dn: np.ndarray,
        jn: np.ndarray,
        target_positions: np.ndarray,
        target_indoor: np.ndarray,
        spectra: torch.Tensor,
        dim: int,
        k: int,
        feature_set: str,
    ):
        batch = len(dn)
        neighbors = spectra[jn]
        if dim == 1:
            y = neighbors.permute(0, 1, 3, 4, 2).reshape(
                batch, k, -1, mh * mv
            )
        else:
            y = neighbors.reshape(batch, k, -1, s)
        mean = y.mean(1, keepdim=True)
        agree = F.cosine_similarity(y, mean, dim=-1)
        d = torch.as_tensor(dn, device=dev)[:, :, None].expand(
            -1, -1, y.shape[2]
        )
        neighbor_indoor = torch.as_tensor(indoor[jn], device=dev)[:, :, None]
        neighbor_indoor = neighbor_indoor.expand(-1, -1, y.shape[2])
        target_indoor_t = torch.as_tensor(target_indoor, device=dev)[:, None, None]
        target_indoor_t = target_indoor_t.expand(-1, k, y.shape[2])
        columns = [
            d / 3.0,
            (neighbor_indoor == target_indoor_t).float(),
            agree,
            agree.square(),
            torch.ones_like(d),
            (d < 2.5).float(),
        ]
        if feature_set == "geometry":
            target_xy = torch.as_tensor(
                target_positions[:, :2],
                dtype=torch.float32,
                device=dev,
            )
            neighbor_xy = torch.as_tensor(
                pos[jn, :2], dtype=torch.float32, device=dev
            )
            delta = neighbor_xy - target_xy[:, None, :]
            bs_xy = torch.tensor(
                setup["X"][:2], dtype=torch.float32, device=dev
            )
            radial = target_xy - bs_xy
            radius = radial.norm(dim=-1).clamp_min(1e-3)
            radial_unit = radial / radius[:, None]
            tangent_unit = torch.stack(
                [-radial_unit[:, 1], radial_unit[:, 0]], dim=-1
            )
            radial_delta = (delta * radial_unit[:, None, :]).sum(-1)
            tangent_delta = (delta * tangent_unit[:, None, :]).sum(-1)

            def expand(value: torch.Tensor) -> torch.Tensor:
                return value[:, :, None].expand(-1, -1, y.shape[2])

            angle = torch.atan2(radial[:, 1], radial[:, 0])
            columns.extend(
                [
                    expand(delta[..., 0] / 5.0),
                    expand(delta[..., 1] / 5.0),
                    expand(radial_delta / 5.0),
                    expand(tangent_delta / 5.0),
                    (radius / 200.0)[:, None, None].expand(
                        -1, k, y.shape[2]
                    ),
                    angle.sin()[:, None, None].expand(-1, k, y.shape[2]),
                    angle.cos()[:, None, None].expand(-1, k, y.shape[2]),
                    neighbor_indoor,
                    target_indoor_t,
                ]
            )
        feats = torch.stack(columns, dim=-1)
        logd = torch.log(
            torch.as_tensor(dn, device=dev).clamp_min(0.3)
        )[:, :, None]
        return feats, y, logd

    def arm_prediction(
        path: Path,
        seed: int,
        tag: str,
        val_idx: np.ndarray,
        train_idx: np.ndarray,
    ) -> torch.Tensor:
        model, meta = load_arm(path, seed, tag)
        k = int(meta["K"])
        dim = int(meta["dim"])
        tree = cKDTree(pos[train_idx, :2])
        dn, local_idx = tree.query(pos[val_idx, :2], k=k)
        dn = dn.astype(np.float32)
        jn = train_idx[local_idx]
        spectra = all_pas if tag == "pas" else all_pdp
        chunks = []
        with torch.no_grad():
            for start in range(0, len(val_idx), 50):
                stop = min(start + 50, len(val_idx))
                feat, y, logd = build_feats(
                    dn[start:stop],
                    jn[start:stop],
                    pos[val_idx[start:stop]],
                    indoor[val_idx[start:stop]],
                    spectra,
                    dim,
                    k,
                    str(meta.get("feature_set", "basic")),
                )
                arm_weight = model(feat, logd)
                temperature = (
                    args.pas_arm_temperature
                    if tag == "pas"
                    else args.pdp_arm_temperature
                )
                if temperature != 1.0:
                    arm_weight = arm_weight.clamp_min(1e-30).pow(
                        temperature
                    )
                    arm_weight /= arm_weight.sum(dim=1, keepdim=True)
                chunks.append((arm_weight[..., None] * y).sum(1))
        raw = torch.cat(chunks)
        del model
        if tag == "pas":
            return nrm(
                raw.reshape(len(val_idx), n, s, mh * mv)
                .permute(0, 3, 1, 2)
                .contiguous(),
                1,
            )
        return nrm(raw.reshape(len(val_idx), mh * mv * mp, n, s), -1)

    def model_prediction(
        path: Path, seed: int, target_positions: np.ndarray
    ):
        require_base(path, seed)
        model, meta = load_model_from_checkpoint(path, device=str(dev))
        pred = predict_test_channels(
            model, target_positions, meta, device=str(dev), batch_size=50
        )
        rms = float(np.sqrt(np.mean(np.abs(pred) ** 2)))
        if not np.isfinite(rms) or rms <= 0:
            raise ValueError(f"{path}: invalid prediction RMS {rms}")
        h = torch.as_tensor(pred / rms, dtype=torch.complex64, device=dev)
        del model, pred
        torch.cuda.empty_cache()
        return nrm(pas(h), 1), nrm(pdp(h), -1), h

    def base_prediction(path: Path, seed: int, val_idx: np.ndarray):
        return model_prediction(path, seed, pos[val_idx])

    bundles: dict[int, SplitBundle] = {}
    split_indices: dict[int, np.ndarray] = {}
    for seed in seeds:
        val_idx = np.asarray(
            sorted(reproduce_val_indices(len(pos), args.val_fraction, seed)),
            dtype=np.int64,
        )
        val_set = set(val_idx.tolist())
        split_indices[seed] = val_idx
        train_idx = np.asarray(
            [i for i in range(len(pos)) if i not in val_set], dtype=np.int64
        )
        if args.validation_neighbor_bank_fraction < 1:
            rng = np.random.default_rng(seed + 481516)
            kept = max(
                args.pas_arm_k,
                args.pdp_arm_k,
                int(
                    round(
                        len(train_idx)
                        * args.validation_neighbor_bank_fraction
                    )
                ),
            )
            train_idx = np.sort(
                rng.choice(train_idx, size=kept, replace=False)
            )
        paths = split_paths(seed)
        gt = torch.as_tensor(
            np.array(channels[val_idx], copy=True).reshape(
                len(val_idx), mh * mv * mp, n, s
            ),
            dtype=torch.complex64,
            device=dev,
        )
        gt_pas, gt_pdp = nrm(pas(gt), 1), nrm(pdp(gt), -1)
        base_pas, base_pdp, base_h = base_prediction(paths["base"], seed, val_idx)
        e35_pas, e35_pdp, e35_h = base_prediction(paths["e35"], seed, val_idx)
        arm_pas = arm_prediction(paths["arm_pas"], seed, "pas", val_idx, train_idx)
        arm_pdp = arm_prediction(paths["arm_pdp"], seed, "pdp", val_idx, train_idx)
        bundles[seed] = SplitBundle(
            seed,
            gt,
            gt_pas,
            gt_pdp,
            base_pas,
            base_pdp,
            e35_pas,
            e35_pdp,
            arm_pas,
            arm_pdp,
            e35_h,
        )
        print(
            f"[split {seed}] components "
            f"base(P={cos_last(base_pas, gt_pas, 1):.5f},"
            f"D={cos_last(base_pdp, gt_pdp, -1):.5f}) "
            f"e35(P={cos_last(e35_pas, gt_pas, 1):.5f},"
            f"D={cos_last(e35_pdp, gt_pdp, -1):.5f}) "
            f"arm(P={cos_last(arm_pas, gt_pas, 1):.5f},"
            f"D={cos_last(arm_pdp, gt_pdp, -1):.5f})",
            flush=True,
        )
        del base_h

    def target_for(
        bundle: SplitBundle, domain: str, mult: float, alpha: float
    ) -> torch.Tensor:
        if domain == "pas":
            pool = nrm(bundle.base_pas + mult * bundle.e35_pas, 1)
            return nrm(alpha * pool + (1.0 - alpha) * bundle.arm_pas, 1)
        pool = nrm(bundle.base_pdp + mult * bundle.e35_pdp, -1)
        return nrm(alpha * pool + (1.0 - alpha) * bundle.arm_pdp, -1)

    domain_rows = []
    selected: dict[str, tuple[float, float]] = {}
    for domain, dim in [("pas", 1), ("pdp", -1)]:
        ranked = []
        for mult in m_grid:
            for alpha in alpha_grid:
                vals = [
                    cos_last(
                        target_for(bundles[seed], domain, mult, alpha),
                        bundles[seed].gt_pas
                        if domain == "pas"
                        else bundles[seed].gt_pdp,
                        dim,
                    )
                    for seed in tune_seeds
                ]
                audit = cos_last(
                    target_for(bundles[audit_seed], domain, mult, alpha),
                    bundles[audit_seed].gt_pas
                    if domain == "pas"
                    else bundles[audit_seed].gt_pdp,
                    dim,
                )
                row = (
                    float(np.median(vals)),
                    float(np.mean(vals)),
                    float(np.min(vals)),
                    mult,
                    alpha,
                    audit,
                )
                ranked.append(row)
                domain_rows.append([domain, mult, alpha, *vals, audit])
        ranked.sort(reverse=True)
        best = ranked[0]
        selected[domain] = (best[3], best[4])
        print(
            f"[select {domain}] tune median={best[0]:.5f} "
            f"mean={best[1]:.5f} worst={best[2]:.5f} "
            f"m={best[3]:.2f} alpha={best[4]:.2f}; "
            f"configuration audit={best[5]:.5f}",
            flush=True,
        )

    def gs_project(pa_target, pd_target, href):
        h = href.clone()
        for _ in range(args.iterations):
            if args.pas_layout in {"pvh", "phv"}:
                spatial = (
                    (mv, mh) if args.pas_layout == "pvh" else (mh, mv)
                )
                angle = torch.fft.fft2(
                    h.reshape(-1, mp, *spatial, n, s),
                    dim=(2, 3),
                    norm="ortho",
                )
                current = angle.abs().square().sum(1).reshape(
                    -1, mh * mv, n, s
                )
                norm = current.norm(dim=1, keepdim=True).clamp_min(1e-30)
                gain = torch.sqrt(
                    (pa_target * norm).clamp_min(0)
                    / current.clamp_min(1e-38)
                ).reshape(-1, 1, *spatial, n, s)
                h = torch.fft.ifft2(
                    angle * gain, dim=(2, 3), norm="ortho"
                ).reshape(h.shape)
            else:
                angle = torch.fft.fft2(
                    h.reshape(-1, mh, mv, mp, n, s),
                    dim=(1, 2),
                    norm="ortho",
                )
                current = angle.abs().square().sum(3).reshape(
                    -1, mh * mv, n, s
                )
                norm = current.norm(dim=1, keepdim=True).clamp_min(1e-30)
                gain = torch.sqrt(
                    (pa_target * norm).clamp_min(0)
                    / current.clamp_min(1e-38)
                ).reshape(-1, mh, mv, 1, n, s)
                h = torch.fft.ifft2(
                    angle * gain, dim=(1, 2), norm="ortho"
                ).reshape(h.shape)
            delay = torch.fft.ifft(h, dim=-1, norm="ortho")
            current = delay.abs().square()
            norm = current.norm(dim=-1, keepdim=True).clamp_min(1e-30)
            h = torch.fft.fft(
                torch.sqrt((pd_target * norm).clamp_min(0))
                * (delay / delay.abs().clamp_min(1e-30)),
                dim=-1,
                norm="ortho",
            )
        return h / h.abs().square().mean().sqrt().clamp_min(1e-30)

    projected = {}
    shape_rows = []
    pas_mult, pas_alpha = selected["pas"]
    pdp_mult, pdp_alpha = selected["pdp"]
    for seed in seeds:
        bundle = bundles[seed]
        pa_target = target_for(bundle, "pas", pas_mult, pas_alpha)
        pd_target = target_for(bundle, "pdp", pdp_mult, pdp_alpha)
        h = gs_project(pa_target, pd_target, bundle.href)
        c1 = cos_last(nrm(pas(h), 1), bundle.gt_pas, 1)
        c2 = cos_last(nrm(pdp(h), -1), bundle.gt_pdp, -1)
        projected[seed] = (h, c1, c2)
        shape_rows.append([seed, c1, c2])
        print(f"[split {seed}] realized GS PAS={c1:.5f} PDP={c2:.5f}", flush=True)

    scale_rank = []
    score_rows = []
    for scale in scale_grid:
        tune_scores = []
        per_split = {}
        for seed in seeds:
            h, c1, c2 = projected[seed]
            gt = bundles[seed].gt
            nmse = float(
                ((h * scale - gt).abs().square().sum())
                / gt.abs().square().sum().clamp_min(1e-30)
            )
            score = (
                weights[0] * c1
                + weights[1] * c2
                + weights[2] / (1.0 + nmse)
            ) / sum(weights)
            per_split[seed] = (score, nmse)
            if seed in tune_seeds:
                tune_scores.append(score)
        scale_rank.append(
            (
                float(np.median(tune_scores)),
                float(np.mean(tune_scores)),
                float(np.min(tune_scores)),
                -scale,
                scale,
                per_split,
            )
        )
    scale_rank.sort(reverse=True)
    best_scale_row = scale_rank[0]
    best_scale = best_scale_row[4]
    best_scores = best_scale_row[5]
    for seed in seeds:
        score, nmse = best_scores[seed]
        c1, c2 = projected[seed][1:]
        score_rows.append([seed, best_scale, c1, c2, nmse, score])
        label = "audit" if seed == audit_seed else "tune"
        print(
            f"[{label} split {seed}] exact no-eps C={score:.5f} "
            f"PAS={c1:.5f} PDP={c2:.5f} NMSE={nmse:.5f} "
            f"scale={best_scale:.2e}",
            flush=True,
        )

    tune_final = [best_scores[seed][0] for seed in tune_seeds]
    audit_score = best_scores[audit_seed][0]
    audit_prediction = (
        projected[audit_seed][0] * best_scale
    ).detach().cpu().numpy().astype(np.complex64)
    audit_prediction_path = outdir / f"audit_s{audit_seed}_prediction.npy"
    audit_indices_path = outdir / f"audit_s{audit_seed}_indices.npy"
    np.save(audit_prediction_path, audit_prediction)
    np.save(audit_indices_path, split_indices[audit_seed])
    if args.save_split_predictions:
        for seed in seeds:
            split_prediction = (
                projected[seed][0] * best_scale
            ).detach().cpu().numpy().astype(np.complex64)
            np.save(outdir / f"split_s{seed}_prediction.npy", split_prediction)
            np.save(outdir / f"split_s{seed}_indices.npy", split_indices[seed])
    audit_reloaded = torch.as_tensor(
        np.load(audit_prediction_path),
        dtype=torch.complex64,
        device=dev,
    )
    audit_truth = torch.as_tensor(
        np.array(channels[split_indices[audit_seed]], copy=True),
        dtype=torch.complex64,
        device=dev,
    )
    audit_c1 = cos_last(pas(audit_reloaded), pas(audit_truth), 1)
    audit_c2 = cos_last(pdp(audit_reloaded), pdp(audit_truth), -1)
    audit_c3 = float(
        (audit_reloaded - audit_truth).abs().square().sum()
        / audit_truth.abs().square().sum().clamp_min(1e-30)
    )
    independent_audit = {
        "C1_PAS": audit_c1,
        "C2_PDP": audit_c2,
        "C3_NMSE": audit_c3,
        "C": (
            weights[0] * audit_c1
            + weights[1] * audit_c2
            + weights[2] / (1.0 + audit_c3)
        )
        / sum(weights),
    }
    if abs(independent_audit["C"] - audit_score) > 2e-5:
        raise RuntimeError(
            f"independent audit mismatch: {independent_audit['C']} != {audit_score}"
        )
    result = {
        "metric": "no-eps cosine (dtype tiny only) + global NMSE",
        "pas_layout": args.pas_layout,
        "validation_neighbor_bank_fraction": (
            args.validation_neighbor_bank_fraction
        ),
        "validation_neighbor_bank_size": int(
            round(
                (len(pos) - len(split_indices[seeds[0]]))
                * args.validation_neighbor_bank_fraction
            )
        ),
        "panel": seeds,
        "tune_seeds": tune_seeds,
        "audit_seed": audit_seed,
        "selected": {
            "pas": {"e35_multiplier": pas_mult, "pool_alpha": pas_alpha},
            "pdp": {"e35_multiplier": pdp_mult, "pool_alpha": pdp_alpha},
            "gs_iterations": args.iterations,
            "scale": best_scale,
        },
        "tune_median_C": float(np.median(tune_final)),
        "tune_mean_C": float(np.mean(tune_final)),
        "tune_worst_C": float(np.min(tune_final)),
        "audit_C": audit_score,
        "independent_audit": independent_audit,
        "passed_0_700": bool(audit_score >= 0.7),
        "audit_artifacts": {
            "prediction": f"{audit_prediction_path}:{sha256(audit_prediction_path)}",
            "indices": f"{audit_indices_path}:{sha256(audit_indices_path)}",
        },
        "scores": [
            {
                "seed": int(row[0]),
                "scale": row[1],
                "PAS": row[2],
                "PDP": row[3],
                "NMSE": row[4],
                "C": row[5],
            }
            for row in score_rows
        ],
        "manifest": manifest,
    }
    (outdir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (outdir / "domain_grid.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(
            ["domain", "e35_multiplier", "pool_alpha"]
            + [f"tune_s{seed}" for seed in tune_seeds]
            + [f"audit_s{audit_seed}"]
        )
        wr.writerows(domain_rows)
    with (outdir / "scores.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["split_seed", "scale", "PAS", "PDP", "NMSE", "C"])
        wr.writerows(score_rows)

    status = "PASS" if audit_score >= 0.7 else "MISS"
    (outdir / "README.md").write_text(
        "# Clean No-Eps Panel\n\n"
        f"Status: **{status}** (`audit C={audit_score:.5f}`)\n\n"
        f"- tune splits: `{tune_seeds}`\n"
        f"- configuration-audit split: `{audit_seed}`\n"
        f"- PAS layout: `{args.pas_layout}`\n"
        f"- tune median/mean/worst: `{np.median(tune_final):.5f}` / "
        f"`{np.mean(tune_final):.5f}` / `{np.min(tune_final):.5f}`\n"
        f"- PAS: e35 multiplier `{pas_mult}`, pool alpha `{pas_alpha}`\n"
        f"- PDP: e35 multiplier `{pdp_mult}`, pool alpha `{pdp_alpha}`\n"
        f"- GS iterations: `{args.iterations}`\n"
        f"- scale: `{best_scale:.2e}`\n\n"
        "The audit split was excluded from blend/scale selection; its checkpoints "
        "still use that split for ordinary early stopping. Checkpoint and audit "
        "artifact SHA256 hashes are frozen in `result.json`.\n",
        encoding="utf-8",
    )

    if args.test_outdir:
        test_outdir = Path(args.test_outdir)
        test_outdir.mkdir(parents=True, exist_ok=True)
        test_pos = np.load(dd / "Round1_Test_Pos.npy").astype(np.float32)
        test_gx = np.clip(
            np.floor((test_pos[:, 0] - x0) / res).astype(int),
            0,
            hm.shape[0] - 1,
        )
        test_gy = np.clip(
            np.floor((test_pos[:, 1] - y0) / res).astype(int),
            0,
            hm.shape[1] - 1,
        )
        test_indoor = (hm[test_gx, test_gy] > 2.0).astype(np.float32)

        # Validation tensors are no longer needed. Keep the full labeled spectra
        # because test arms query a labeled neighbor bank.
        del bundles, projected, audit_prediction
        torch.cuda.empty_cache()

        base_pas_sum = torch.zeros(
            len(test_pos), mh * mv, n, s, device=dev
        )
        base_pdp_sum = torch.zeros(
            len(test_pos), mh * mv * mp, n, s, device=dev
        )
        e35_pas_sum = torch.zeros_like(base_pas_sum)
        e35_pdp_sum = torch.zeros_like(base_pdp_sum)
        href = None
        for seed in seeds:
            paths = split_paths(seed)
            bp, bd, bh = model_prediction(paths["base"], seed, test_pos)
            ep, ed, eh = model_prediction(paths["e35"], seed, test_pos)
            base_pas_sum += bp
            base_pdp_sum += bd
            e35_pas_sum += ep
            e35_pdp_sum += ed
            if seed == args.href_seed:
                href = eh.clone()
            del bp, bd, bh, ep, ed, eh
            torch.cuda.empty_cache()
            print(f"[test] model spectra seed={seed} done", flush=True)
        if href is None:
            raise ValueError(f"href_seed {args.href_seed} not in panel")

        arm_pas_sum = torch.zeros(
            len(test_pos), n * s, mh * mv, device=dev
        )
        arm_pdp_sum = torch.zeros(
            len(test_pos), mh * mv * mp * n, s, device=dev
        )
        for seed in seeds:
            if args.test_full_neighbor_bank:
                train_idx = np.arange(len(pos), dtype=np.int64)
            else:
                val_set = set(
                    reproduce_val_indices(len(pos), args.val_fraction, seed)
                )
                train_idx = np.asarray(
                    [i for i in range(len(pos)) if i not in val_set],
                    dtype=np.int64,
                )
            tree = cKDTree(pos[train_idx, :2])
            for tag in ("pas", "pdp"):
                path = split_paths(seed)[f"arm_{tag}"]
                model, meta = load_arm(path, seed, tag)
                k = int(meta["K"])
                dim = int(meta["dim"])
                dn, local_idx = tree.query(test_pos[:, :2], k=k)
                dn = dn.astype(np.float32)
                jn = train_idx[local_idx]
                spectra = all_pas if tag == "pas" else all_pdp
                chunks = []
                with torch.no_grad():
                    for start in range(0, len(test_pos), 50):
                        stop = min(start + 50, len(test_pos))
                        feat, y, logd = build_feats(
                            dn[start:stop],
                            jn[start:stop],
                            test_pos[start:stop],
                            test_indoor[start:stop],
                            spectra,
                            dim,
                            k,
                            str(meta.get("feature_set", "basic")),
                        )
                        arm_weight = model(feat, logd)
                        temperature = (
                            args.pas_arm_temperature
                            if tag == "pas"
                            else args.pdp_arm_temperature
                        )
                        if temperature != 1.0:
                            arm_weight = arm_weight.clamp_min(1e-30).pow(
                                temperature
                            )
                            arm_weight /= arm_weight.sum(
                                dim=1, keepdim=True
                            )
                        chunks.append(
                            (arm_weight[..., None] * y).sum(1)
                        )
                raw = torch.cat(chunks)
                if tag == "pas":
                    arm_pas_sum += raw
                else:
                    arm_pdp_sum += raw
                del model, raw
                torch.cuda.empty_cache()
            print(f"[test] clean arms seed={seed} done", flush=True)

        count = float(len(seeds))
        base_pas_test = nrm(base_pas_sum / count, 1)
        base_pdp_test = nrm(base_pdp_sum / count, -1)
        e35_pas_test = nrm(e35_pas_sum / count, 1)
        e35_pdp_test = nrm(e35_pdp_sum / count, -1)
        arm_pas_test = nrm(
            (arm_pas_sum / count)
            .reshape(len(test_pos), n, s, mh * mv)
            .permute(0, 3, 1, 2)
            .contiguous(),
            1,
        )
        arm_pdp_test = nrm(
            (arm_pdp_sum / count).reshape(
                len(test_pos), mh * mv * mp, n, s
            ),
            -1,
        )
        pas_pool_test = nrm(
            base_pas_test + pas_mult * e35_pas_test, 1
        )
        pdp_pool_test = nrm(
            base_pdp_test + pdp_mult * e35_pdp_test, -1
        )
        pas_target_test = nrm(
            pas_alpha * pas_pool_test
            + (1.0 - pas_alpha) * arm_pas_test,
            1,
        )
        pdp_target_test = nrm(
            pdp_alpha * pdp_pool_test
            + (1.0 - pdp_alpha) * arm_pdp_test,
            -1,
        )
        test_h = gs_project(pas_target_test, pdp_target_test, href)
        test_array = (
            test_h * best_scale
        ).detach().cpu().numpy().astype(np.complex64)
        expected_shape = (
            len(test_pos),
            setup["M"],
            setup["N"],
            setup["S"],
        )
        if test_array.shape != expected_shape:
            raise ValueError(
                f"test output shape {test_array.shape} != {expected_shape}"
            )
        if not np.isfinite(test_array.real).all() or not np.isfinite(
            test_array.imag
        ).all():
            raise ValueError("test output contains non-finite values")
        test_path = test_outdir / "Round1_Test_Channel.npy"
        np.save(test_path, test_array)
        test_manifest = {
            "source_validation": str(outdir / "result.json"),
            "pas_layout": args.pas_layout,
            "selected": result["selected"],
            "href_seed": args.href_seed,
            "shape": list(test_array.shape),
            "dtype": str(test_array.dtype),
            "rms": float(np.sqrt(np.mean(np.abs(test_array) ** 2))),
            "output_sha256": sha256(test_path),
            "checkpoints": manifest,
            "test_full_neighbor_bank": args.test_full_neighbor_bank,
            "test_neighbor_bank_size": (
                len(pos)
                if args.test_full_neighbor_bank
                else len(pos)
                - len(
                    reproduce_val_indices(
                        len(pos), args.val_fraction, seeds[0]
                    )
                )
            ),
        }
        (test_outdir / "manifest.json").write_text(
            json.dumps(test_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"TEST_BUILD_DONE path={test_path} "
            f"rms={test_manifest['rms']:.3e} "
            f"sha256={test_manifest['output_sha256']}",
            flush=True,
        )
    print(
        f"CLEAN_NOEPS_DONE status={status} audit_C={audit_score:.5f} "
        f"tune_median={np.median(tune_final):.5f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
