# Reproduce GS49

GS49 extends the GS37 clean no-eps pipeline with two independently trained
complex-neighbor gates and a source-only bilateral PDP correction. Generated
checkpoints, caches, validation predictions, and submission arrays are
intentionally excluded from Git.

## Prerequisites

Place the competition files under `Round1_Map(2)/`, then complete the spectrum
cache and five-fold attention steps in `REPRODUCE_GS37.md`.

Build the label-free test-geometry panel:

```bash
python3 scripts/build_test_geometry_panel.py
```

Build the physically transported PAS cache and its clean low-capacity gate:

```bash
python3 scripts/build_pas_transport_cache.py \
  --config configs/gs49_dual_gate_bilateral_pdp.json \
  --include-test

python3 scripts/train_pas_transport_gate.py
```

The expected selected PAS transport checkpoint SHA256 is:

```text
1bddf4a5ab56818f8b79b5c4509a0c502f9f4b61848a1865800de9baa3451292
```

## Complex Gates

Build the clean Gram caches and train the shared MLP gate:

```bash
python3 scripts/train_complex_neighbor_gate.py \
  --config configs/gs49_dual_gate_bilateral_pdp.json \
  --external-indices cache/test_geometry_panel/proxy_all.npy \
  --external-name geomall \
  --feature-set invariant \
  --architecture mlp16 \
  --epochs 300 \
  --cache-dir cache/complex_neighbor_gate_probe \
  --checkpoint-dir checkpoints/complex_neighbor_mlp16_geomall_e300 \
  --out docs/complex_neighbor_gate/mlp16_geomall_e300.json \
  --rebuild-cache
```

Train the complementary one-layer set Transformer on the same caches:

```bash
python3 scripts/train_complex_neighbor_gate.py \
  --config configs/gs49_dual_gate_bilateral_pdp.json \
  --external-indices cache/test_geometry_panel/proxy_all.npy \
  --external-name geomall \
  --feature-set invariant \
  --architecture set16 \
  --epochs 300 \
  --cache-dir cache/complex_neighbor_gate_probe \
  --checkpoint-dir checkpoints/complex_neighbor_set16_e300 \
  --out docs/complex_neighbor_gate/set16_e300.json
```

Expected selected checkpoint SHA256 values:

```text
fac890bb83535c75f01c2c86947161307daa6f8ea1a45a6c3e02794cd97998c6  mlp16
4d9bcdd8e1e1ace1a0a4b81d4e04934d6bb249aaccef3e74c62210d4ed3685bb  set16
```

## Submission

```bash
python3 scripts/build_gs37_geophase.py \
  --config configs/gs49_dual_gate_bilateral_pdp.json \
  --device cuda
```

Expected artifact:

```text
best_submit/BLEND_GS49_DUAL_GATE_BILATERAL_PDP/Round1_Test_Channel.npy
shape:  (500, 256, 4, 192)
dtype:  complex64
sha256: 6b87e80b6242096f993b625da294259e23fb7a5fabd18c6f97ce8d31579c75b9
```

The bilateral PDP arm uses only test positions and the training spectra. Its
parameters were frozen after four representative tune splits and then passed
the disjoint strict audit; no hidden channel labels are read.
