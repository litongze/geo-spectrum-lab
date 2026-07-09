"""End-to-end smoke test: data -> model -> train -> infer -> score.

Runs on CPU with tiny dimensions in a few seconds, so CI (and the Windows
teammate) can verify the whole pipeline is wired correctly without the real
dataset or a GPU.

    python -m pytest tests/                # as a test
    python tests/test_pipeline.py          # as a script
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from wireless_twin.data.channel_dataset import ChannelDataset
from wireless_twin.data.normalization import ChannelScaler
from wireless_twin.data.setup_config import ChannelSpec
from wireless_twin.evaluation.metrics import evaluate_channels
from wireless_twin.evaluation.predictor import predict_test_channels
from wireless_twin.models import build_model
from wireless_twin.training import TrainConfig, Trainer


def _make_data(spec, n, seed=0):
    rng = np.random.default_rng(seed)
    k = 8
    pos = rng.uniform([-20, -20, 0], [20, 20, 5], size=(n, 3)).astype(np.float32)
    u = (rng.standard_normal((k, spec.m)) + 1j * rng.standard_normal((k, spec.m)))
    v = (rng.standard_normal((k, spec.n)) + 1j * rng.standard_normal((k, spec.n)))
    w = (rng.standard_normal((k, spec.s)) + 1j * rng.standard_normal((k, spec.s)))
    freqs = rng.standard_normal((k, 3)) * 0.15
    gains = np.exp(1j * (pos @ freqs.T))
    ch = np.einsum("pk,km,kn,ks->pmns", gains, u, v, w).astype(np.complex64)
    return pos, ch


def test_pipeline():
    spec = ChannelSpec(m=8, mh=2, mv=2, mp=2, n=2, nh=1, nv=1, np=2, s=16,
                       bs_position=[50, 0, 25], metric_weights=[0.4, 0.4, 0.2])
    spec.validate()

    train_pos, train_ch = _make_data(spec, 200, seed=1)
    test_pos, test_gt = _make_data(spec, 40, seed=2)

    pos_mean, pos_std = train_pos.mean(0), train_pos.std(0) + 1e-8
    scaler = ChannelScaler("std").fit(train_ch)
    train_ds = ChannelDataset((train_pos - pos_mean) / pos_std, train_ch, scaler)

    model = build_model("path_field", spec, n_paths=32, hidden_dim=128,
                        n_layers=3, n_freqs=6)

    cfg = TrainConfig(epochs=15, batch_size=32, lr=2e-3, val_fraction=0.1,
                      log_every=5, device="cpu")
    meta = {
        "model_name": "path_field",
        "model_kwargs": {"n_paths": 32, "hidden_dim": 128, "n_layers": 3,
                         "n_freqs": 6},
        "spec": spec.__dict__,
        "scaler": scaler.state_dict(),
        "pos_mean": pos_mean.tolist(),
        "pos_std": pos_std.tolist(),
        "round_tag": "Round1",
    }
    trainer = Trainer(model, train_ds, cfg, checkpoint_meta=meta)
    history = trainer.fit()

    # loss must go down
    assert history[-1]["train_loss"] < history[0]["train_loss"], "loss not decreasing"

    # checkpoint round-trips
    with tempfile.TemporaryDirectory() as d:
        ckpt = pathlib.Path(d) / "m.pt"
        trainer.save_checkpoint(ckpt)
        assert ckpt.exists()

    # predict + score
    pred = predict_test_channels(model, test_pos, meta, device="cpu")
    assert pred.shape == test_gt.shape
    assert pred.dtype == np.complex64

    scores = evaluate_channels(pred, test_gt, spec)
    print("scores:", {k: round(v, 4) for k, v in scores.items()})
    assert 0.0 <= scores["C1_PAS"] <= 1.0001
    assert 0.0 <= scores["C2_PDP"] <= 1.0001
    assert scores["C3_NMSE"] >= 0.0
    assert np.isfinite(scores["C"])


if __name__ == "__main__":
    test_pipeline()
    print("OK: end-to-end pipeline smoke test passed.")
