# Reproduce GS54

GS54 keeps the clean GS49 source model, corrects PAS scoring/projection to the
official HVP array order, adds a learned source-only covariance-kriging
residual, and applies radial-height PAS/PDP corrections. Generated caches,
checkpoints, validation predictions, and submission arrays are intentionally
excluded from Git.

GS55 and GS56 contain exactly the same complex direction and spectra as GS54.
They only test lower global amplitude scales, so they are scale-risk controls,
not independent ensembles.

## Prerequisites

Place the competition files under `Round1_Map(2)/` and complete
`REPRODUCE_GS49.md`. In particular, these selected checkpoints must exist:

```text
fac890bb83535c75f01c2c86947161307daa6f8ea1a45a6c3e02794cd97998c6  checkpoints/complex_neighbor_mlp16_geomall_e300/selected.pt
4d9bcdd8e1e1ace1a0a4b81d4e04934d6bb249aaccef3e74c62210d4ed3685bb  checkpoints/complex_neighbor_set16_e300/selected.pt
1bddf4a5ab56818f8b79b5c4509a0c502f9f4b61848a1865800de9baa3451292  checkpoints/pas_transport_gate/selected.pt
```

The first GS49 complex-gate command also creates clean Gram/cross-correlation
caches for the four tune splits, the untouched audit split, and the final
all-training leave-one-out fit under `cache/complex_neighbor_gate_probe/`.

## Covariance Kriging

Train one independently fitted correlation predictor per validation split.
The empty external panel keeps this command limited to the four tune splits
and the audit split:

```bash
python3 scripts/train_covariance_kriging.py \
  --cache-dir cache/complex_neighbor_gate_probe \
  --checkpoint-dir checkpoints/covariance_kriging_full \
  --out docs/covariance_kriging/full_clean_panel.json \
  --external-names "" \
  --feature-set full \
  --epochs 100 \
  --hidden-dim 32 \
  --device cuda
```

The predictor sees source-pool channels and geometry only. For every split,
the complete validation index set is absent from its training pool. The
deployed checkpoint is then fitted once on all 2,000 training rows using
leave-one-out neighbors and frozen hyperparameters:

```bash
python3 scripts/train_covariance_kriging.py \
  --cache-dir cache/complex_neighbor_gate_probe \
  --checkpoint-dir checkpoints/covariance_kriging_full \
  --out docs/covariance_kriging/full_final.json \
  --feature-set full \
  --epochs 100 \
  --hidden-dim 32 \
  --train-final \
  --device cuda
```

Expected selected checkpoint:

```text
67191905e77007887be9ea7f26db85b6a333dbef10b70bd57507a76c372ddeb5  checkpoints/covariance_kriging_full/selected.pt
```

GPU kernels or a different PyTorch release can produce a different checkpoint
hash. In that case, compare clean-panel scores rather than assuming byte-level
checkpoint identity.

## Clean Validation

The frozen covariance hybrid can be rechecked before radial post-processing:

```bash
python3 scripts/validate_geometric_phase_neighbors_projection.py \
  --config configs/gs54_official_hvp_covkrig_radial.json \
  --sources anis_k16_r5_p2_equalized \
  --complex-gate-dir checkpoints/complex_neighbor_mlp16_geomall_e300 \
  --complex-gate-alpha 0.5 \
  --complex-gate-secondary-dir checkpoints/complex_neighbor_set16_e300 \
  --complex-gate-secondary-alpha 0.1 \
  --covariance-kriging-dir checkpoints/covariance_kriging_full \
  --covariance-kriging-loading 1 \
  --covariance-kriging-blend-grid 0.5 \
  --covariance-kriging-hybrid-grid 0.5 \
  --steering-result docs/array_steering_phase_bound64/result.json \
  --pas-blend 0.8 \
  --pdp-blend 0.4 \
  --pas-transport-blend 0.25 \
  --pas-transport-gate-dir cache/pas_transport_gate \
  --hvp-expert-dir cache/moment_attention_hvp \
  --hvp-blend-grid 1 \
  --dual-orders pdp_hvp \
  --iteration-grid 8 \
  --source-blend-grid 0.9 \
  --score-pas-layout hvp \
  --strict-audit \
  --out docs/official_hvp_projection/covariance_kriging_hybrid_strict.json \
  --device cuda
```

The selected pre-radial candidate has regular tune median `0.751672`, regular
audit `0.776113`, and strict audit `0.809191`. The radial-height PAS/PDP step
raises those to tune median `0.754371`, regular audit `0.777717`, and strict
audit `0.812968`. These are local validation scores, not online estimates.

## Submission

Build the high-upside candidate:

```bash
python3 scripts/build_gs37_geophase.py \
  --config configs/gs54_official_hvp_covkrig_radial.json \
  --device cuda
```

Expected artifact:

```text
best_submit/BLEND_GS54_OFFICIAL_HVP_COVKRIG_RADIAL/Round1_Test_Channel.npy
shape:  (500, 256, 4, 192)
dtype:  complex64
rms:    1.824417716e-05
sha256: 0555e6a527af4c8850195eb1d733d9798cdbe78759ef5a039bb8514404abdbea
```

Build the two amplitude controls without changing PAS, PDP, or channel
direction:

```bash
python3 scripts/rescale_submission.py \
  --input best_submit/BLEND_GS54_OFFICIAL_HVP_COVKRIG_RADIAL/Round1_Test_Channel.npy \
  --output best_submit/BLEND_GS55_OFFICIAL_HVP_COVKRIG_MIDSCALE/Round1_Test_Channel.npy \
  --target-rms 9.12208858e-06

python3 scripts/rescale_submission.py \
  --input best_submit/BLEND_GS54_OFFICIAL_HVP_COVKRIG_RADIAL/Round1_Test_Channel.npy \
  --output best_submit/BLEND_GS56_OFFICIAL_HVP_COVKRIG_LOWSCALE/Round1_Test_Channel.npy \
  --target-rms 5.44975018e-07
```

Expected hashes:

```text
a6c142f05e73fa07218050e602851308045bdcec0d6f7ac36f7c8d5f377ce3b8  GS55
95c96cd752d5efee779dae7eedd90783f60d238a8d613dd7cb5b2979fb8ee88c  GS56
```

GS54 is the primary submission because clean folds consistently favor its
NMSE-calibrated scale. GS55 is the conservative amplitude hedge. GS56 isolates
the official-HVP and radial spectral changes at the old GS30-scale amplitude;
it should only be used as an A/B diagnostic when submission slots permit.
