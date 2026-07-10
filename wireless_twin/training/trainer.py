"""Training loop, decoupled from model and data specifics.

The trainer consumes ``(position, target_ri)`` batches, asks the model for a
complex channel, scores it with :class:`ChannelLoss`, and steps an optimiser.
It knows nothing about the architecture beyond the :class:`ChannelModel`
interface, and nothing about the file format beyond the flat real/imag target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import DataLoader, random_split

from ..data.channel_dataset import ChannelDataset
from ..models.base import ChannelModel
from .losses import build_loss


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    loss_name: str = "competition"
    aug_noise_std: float = 0.0   # per-step complex-Gaussian noise on the target
    lambda_pas: float = 1.0
    lambda_pdp: float = 1.0
    lambda_mag: float = 1.0
    val_fraction: float = 0.1
    grad_clip: float = 1.0
    num_workers: int = 0            # 0 is the safe cross-platform default
    log_every: int = 10
    seed: int = 0
    device: Optional[str] = None    # "cuda" / "cpu" / None=auto


def _ri_to_complex(t: torch.Tensor, m: int, n: int, s: int) -> torch.Tensor:
    ri = t.reshape(t.shape[0], m, n, s, 2)
    return torch.complex(ri[..., 0], ri[..., 1])


class Trainer:
    def __init__(
        self,
        model: ChannelModel,
        train_set: ChannelDataset,
        config: TrainConfig,
        checkpoint_meta: Optional[Dict] = None,
    ) -> None:
        self.config = config
        self.device = torch.device(
            config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.spec = model.spec
        self.criterion = build_loss(
            config.loss_name, self.spec,
            lambda_pas=config.lambda_pas, lambda_pdp=config.lambda_pdp,
            lambda_mag=config.lambda_mag).to(self.device)
        self.checkpoint_meta = checkpoint_meta or {}

        # train / val split for monitoring
        n_val = int(len(train_set) * config.val_fraction)
        n_train = len(train_set) - n_val
        gen = torch.Generator().manual_seed(config.seed)
        if n_val > 0:
            self.train_split, self.val_split = random_split(
                train_set, [n_train, n_val], generator=gen)
        else:
            self.train_split, self.val_split = train_set, None

        self.train_loader = DataLoader(
            self.train_split, batch_size=config.batch_size, shuffle=True,
            num_workers=config.num_workers, drop_last=False)
        self.val_loader = (
            DataLoader(self.val_split, batch_size=config.batch_size,
                       num_workers=config.num_workers)
            if self.val_split is not None else None)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.lr,
            weight_decay=config.weight_decay)

        # best-by-validation-C snapshot (val peaks mid-training then overfits)
        self.best_score = -1.0
        self.best_epoch = -1
        self.best_state = None

    # ------------------------------------------------------------------
    def _run_batch(self, pos: torch.Tensor, target: torch.Tensor,
                   train: bool) -> Dict[str, float]:
        pos = pos.to(self.device)
        target = target.to(self.device)
        gt_h = _ri_to_complex(target, self.spec.m, self.spec.n, self.spec.s)

        # data augmentation: fresh complex-Gaussian noise on the target each step
        if train and self.config.aug_noise_std > 0:
            sigma = self.config.aug_noise_std * gt_h.abs().mean()
            noise = torch.randn_like(gt_h.real) + 1j * torch.randn_like(gt_h.imag)
            gt_h = gt_h + sigma * noise / (2 ** 0.5)

        pred_h = self.model(pos)
        out = self.criterion(pred_h, gt_h)

        if train:
            self.optimizer.zero_grad(set_to_none=True)
            out["loss"].backward()
            if self.config.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
        return {k: float(v.detach()) for k, v in out.items()}

    def _epoch(self, loader: DataLoader, train: bool) -> Dict[str, float]:
        self.model.train(train)
        agg: Dict[str, float] = {}
        n = 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for pos, target in loader:
                stats = self._run_batch(pos, target, train)
                bs = pos.shape[0]
                n += bs
                for k, v in stats.items():
                    agg[k] = agg.get(k, 0.0) + v * bs
        return {k: v / max(n, 1) for k, v in agg.items()}

    def _val_score(self, val_stats: Dict[str, float]) -> float:
        """Competition metric C on the validation split (higher is better)."""
        w1, w2, w3 = self.spec.metric_weights
        return (w1 * val_stats.get("pas", 0.0)
                + w2 * val_stats.get("pdp", 0.0)
                + w3 / (1.0 + val_stats.get("nmse", 1.0)))

    def fit(self) -> List[Dict[str, float]]:
        history: List[Dict[str, float]] = []
        import copy
        for epoch in range(1, self.config.epochs + 1):
            train_stats = self._epoch(self.train_loader, train=True)
            record = {"epoch": epoch, **{f"train_{k}": v
                                          for k, v in train_stats.items()}}
            if self.val_loader is not None:
                val_stats = self._epoch(self.val_loader, train=False)
                record.update({f"val_{k}": v for k, v in val_stats.items()})
                # keep the best-by-validation-C weights (val peaks then overfits)
                score = self._val_score(val_stats)
                if score > self.best_score:
                    self.best_score = score
                    self.best_epoch = epoch
                    self.best_state = copy.deepcopy(
                        {k: v.cpu() for k, v in self.model.state_dict().items()})
            history.append(record)

            if epoch == 1 or epoch % self.config.log_every == 0 \
                    or epoch == self.config.epochs:
                self._log(record)
        if self.best_state is not None:
            print(f"[trainer] best val C={self.best_score:.4f} @ epoch "
                  f"{self.best_epoch}", flush=True)
        return history

    @staticmethod
    def _log(record: Dict[str, float]) -> None:
        parts = [f"epoch {record['epoch']:>4d}"]
        for key in ("train_loss", "train_nmse", "train_pas", "train_pdp",
                    "val_loss", "val_nmse", "val_pas", "val_pdp"):
            if key in record:
                parts.append(f"{key}={record[key]:.4f}")
        print(" | ".join(parts), flush=True)

    # ------------------------------------------------------------------
    def save_checkpoint(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # prefer the best-by-validation weights over the (overfit) final ones
        state = self.best_state if self.best_state is not None \
            else self.model.state_dict()
        payload = {"model_state": state, "meta": self.checkpoint_meta}
        torch.save(payload, str(path))
        tag = (f" (best val C={self.best_score:.4f} @ ep{self.best_epoch})"
               if self.best_state is not None else "")
        print(f"[trainer] saved checkpoint -> {path}{tag}", flush=True)
