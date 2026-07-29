# GS37 Source-Only Reproduction

This repository intentionally excludes predictions, checkpoints, caches, and
competition data. The commands below regenerate the GS37 geometric-phase
candidate from local data and checkpoints.

## Data Layout

Place the organizer files under `Round1_Map(2)/`:

```text
Round1_Setup.json
Round1_Map.ply
Round1_Train_Pos.npy
Round1_Train_Channel.npy
Round1_Test_Pos.npy
```

Install the project dependencies from the repository root:

```bash
python3 -m pip install -r requirements-twin.txt
```

## Train Spectrum Arms

Train the five clean PAS/PDP attention arms. Each split excludes its validation
points from the legal neighbor pool.

```bash
python3 scripts/train_gs37_moment_arms.py
```

The checkpoints are written below `checkpoints/`, which is ignored by Git.

Build the reusable PHV PAS and PDP caches:

```bash
python3 scripts/build_spectrum_cache.py --pas-layout phv
```

The cache is written below `cache/`, which is also ignored by Git.

## Build GS37

The frozen parameters are in
`configs/gs37_geophase_moment.json`. They were selected on seeds
`1890,3716,962,1022`; seed `2262` was held out as an external audit split.

GS37 uses an existing spectrum baseline as a dense phase seed and spectral
anchor. Pass the local GS34 channel array explicitly when its path differs:

```bash
python3 scripts/build_gs37_geophase.py \
  --baseline best_submit/BLEND_GS34_PHV_FULLBANK_REF10_TF05/Round1_Test_Channel.npy
```

The generated submission is:

```text
best_submit/BLEND_GS37_GEOPHASE_MOMENT/Round1_Test_Channel.npy
```

All generated outputs remain ignored by Git. To reproduce the exact historical
array bit-for-bit, exchange the ten selected attention checkpoints and the GS34
baseline out of band; retraining can differ slightly across CUDA/PyTorch
versions.

## Optional Refit

Refit the physical carrier/subcarrier phase ramp using only the four tune
splits:

```bash
python3 scripts/probe_geometric_phase.py --out docs/geometric_phase/result.json
```

Pass that lightweight local result to the builder with `--phase-result`.
Validation outputs under `docs/` should remain local and are not required when
using the frozen config.

## Teammate Transformer

The teammate-provided Transformer source snapshot lives under `code/`. It is
kept separate from the root package and contains no model weights or result
arrays.
