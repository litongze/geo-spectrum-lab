#!/usr/bin/env python3
"""Search PAS and PDP KNN interpolation parameters independently."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from _bootstrap import ROOT  # noqa: F401

sys.path.insert(0, str(ROOT / "scripts"))
from score_holdout import reproduce_val_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", default="Round1_Map(2)")
    parser.add_argument("--cache-dir", default="cache/teammate_knn_hvp")
    parser.add_argument(
        "--pas-layout",
        choices=("hvp", "hpv", "vhp", "vph", "phv", "pvh"),
        default="hvp",
    )
    parser.add_argument("--panel", default="1890,3716,962,1022,2262")
    parser.add_argument(
        "--testmatched",
        default="handoff_to_teammate_20260727/splits/"
        "test_matched_seed2026_val.npy",
    )
    parser.add_argument("--k-grid", default="1,2,3,4,5,6,8,10,12")
    parser.add_argument("--dp-grid", default="1,1.5,2,2.5,3,3.5,4,5")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="docs/knn_domain_sweep/result.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(value) for value in args.panel.split(",") if value.strip()]
    tune_seeds, audit_seed = seeds[:-1], seeds[-1]
    k_grid = [int(value) for value in args.k_grid.split(",") if value.strip()]
    dp_grid = [
        float(value) for value in args.dp_grid.split(",") if value.strip()
    ]
    configs = [(k, dp) for k in k_grid for dp in dp_grid]
    max_k = max(k_grid)
    device = torch.device(args.device)

    datadir = Path(args.datadir)
    train_pos = np.load(datadir / "Round1_Train_Pos.npy").astype(np.float32)
    cache_dir = Path(args.cache_dir)
    pas_np = np.load(
        cache_dir / f"train_pas_{args.pas_layout}.npy", mmap_mode="r"
    )
    pdp_np = np.load(cache_dir / "train_pdp.npy", mmap_mode="r")
    print("[domain-sweep] loading cached spectra to GPU", flush=True)
    all_pas = torch.as_tensor(
        np.array(pas_np, copy=True), dtype=torch.float32, device=device
    )
    all_pdp = torch.as_tensor(
        np.array(pdp_np, copy=True), dtype=torch.float32, device=device
    )

    split_indices = {
        str(seed): np.asarray(
            sorted(reproduce_val_indices(len(train_pos), 0.1, seed)),
            dtype=np.int64,
        )
        for seed in seeds
    }
    split_indices["testmatched"] = np.load(args.testmatched).astype(np.int64)

    neighbor_tables = {}
    all_indices = np.arange(len(train_pos), dtype=np.int64)
    for name, val_idx in split_indices.items():
        pool_idx = np.setdiff1d(
            all_indices, val_idx, assume_unique=False
        )
        tree = cKDTree(train_pos[pool_idx])
        distance, local_idx = tree.query(train_pos[val_idx], k=max_k)
        neighbor_tables[name] = (
            distance.astype(np.float32),
            pool_idx[local_idx],
        )

    def domain_scores(
        spectra: torch.Tensor,
        vector_length: int,
    ) -> dict[str, list[float]]:
        result = {}
        tiny = torch.finfo(torch.float32).tiny
        for split_name, val_idx in split_indices.items():
            distance, neighbors = neighbor_tables[split_name]
            totals = torch.zeros(len(configs), device=device)
            count = 0
            gt = spectra[
                torch.as_tensor(val_idx, dtype=torch.long, device=device)
            ].reshape(len(val_idx), -1, vector_length)
            gt = gt / gt.norm(dim=-1, keepdim=True).clamp_min(tiny)
            for start in range(0, len(val_idx), args.batch_size):
                stop = min(start + args.batch_size, len(val_idx))
                batch_neighbors = torch.as_tensor(
                    neighbors[start:stop], dtype=torch.long, device=device
                )
                values = spectra[batch_neighbors].reshape(
                    stop - start, max_k, -1, vector_length
                )
                batch_distance = torch.as_tensor(
                    distance[start:stop], device=device
                ).clamp_min(1e-3)
                weights = torch.zeros(
                    len(configs), stop - start, max_k, device=device
                )
                for config_idx, (k, dp) in enumerate(configs):
                    local = batch_distance[:, :k].pow(-dp)
                    weights[config_idx, :, :k] = (
                        local / local.sum(dim=1, keepdim=True)
                    )
                prediction = torch.einsum(
                    "cbk,bkvl->cbvl", weights, values
                )
                prediction = prediction / prediction.norm(
                    dim=-1, keepdim=True
                ).clamp_min(tiny)
                target = gt[start:stop].unsqueeze(0)
                totals += (prediction * target).sum(dim=-1).sum(dim=(1, 2))
                count += target.shape[1] * target.shape[2]
                del values, weights, prediction, target
            scores = (totals / count).detach().cpu().numpy().tolist()
            result[split_name] = [float(value) for value in scores]
            print(f"[domain-sweep] {split_name} done", flush=True)
        return result

    pas_scores = domain_scores(all_pas, all_pas.shape[-1])
    del all_pas
    torch.cuda.empty_cache()
    pdp_scores = domain_scores(all_pdp, all_pdp.shape[-1])
    del all_pdp
    torch.cuda.empty_cache()

    def rank_domain(scores: dict[str, list[float]]) -> list[dict]:
        rows = []
        for config_idx, (k, dp) in enumerate(configs):
            tune = [
                scores[str(seed)][config_idx] for seed in tune_seeds
            ]
            rows.append(
                {
                    "k": k,
                    "distance_power": dp,
                    "tune_median": float(np.median(tune)),
                    "tune_mean": float(np.mean(tune)),
                    "tune_worst": float(np.min(tune)),
                    "audit": scores[str(audit_seed)][config_idx],
                    "testmatched": scores["testmatched"][config_idx],
                    "scores": {
                        name: values[config_idx]
                        for name, values in scores.items()
                    },
                }
            )
        return sorted(
            rows,
            key=lambda row: (
                row["tune_median"],
                row["tune_mean"],
                row["tune_worst"],
                -row["k"],
            ),
            reverse=True,
        )

    pas_ranked = rank_domain(pas_scores)
    pdp_ranked = rank_domain(pdp_scores)
    payload = {
        "tune_seeds": tune_seeds,
        "audit_seed": audit_seed,
        "pas_layout": args.pas_layout,
        "configs": [
            {"k": k, "distance_power": dp} for k, dp in configs
        ],
        "pas_ranked": pas_ranked,
        "pdp_ranked": pdp_ranked,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("[PAS TOP]", json.dumps(pas_ranked[:5]), flush=True)
    print("[PDP TOP]", json.dumps(pdp_ranked[:5]), flush=True)
    print(f"[domain-sweep] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
