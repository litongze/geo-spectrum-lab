"""Single- or multi-GPU trainer for Physical-AI channel models.

Launch normally for one GPU/CPU, or with torchrun for DDP:

    torchrun --standalone --nproc_per_node=4 scripts/train.py ...
"""

from __future__ import annotations

import copy
import math
import os
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset, random_split

from ..data.channel_dataset import ChannelDataset
from ..models.base import ChannelModel
from .losses import ChannelLoss


@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 4
    lr: float = 2e-4
    min_lr_ratio: float = 0.05
    warmup_epochs: int = 10
    weight_decay: float = 1e-4
    lambda_nmse: float = 20.0
    lambda_pas: float = 1.0
    lambda_pdp: float = 1.0
    lambda_sparse: float = 1e-6
    lambda_log_power: float = 0.0
    nmse_mode: str = "dataset"
    val_fraction: float = 0.1
    val_split_mode: str = "spatial"  # spatial | random
    val_indices_file: Optional[str] = None
    val_regions: int = 4
    position_noise_std: float = 0.01
    grad_clip: float = 1.0
    accumulate_steps: int = 1
    num_workers: int = 0
    log_every: int = 1
    seed: int = 0
    split_seed: Optional[int] = None
    device: Optional[str] = None
    precision: str = "bf16"  # fp32 | bf16 | fp16
    pin_memory: bool = True
    early_stopping_patience: int = 50
    # FP16 AMP starts at 65536 by default, which is often too aggressive for
    # this complex-valued FFT loss. A lower initial scale avoids first-update
    # overflow while GradScaler can still grow it later.
    amp_init_scale: float = 1024.0
    amp_growth_interval: int = 2000
    amp_backoff_factor: float = 0.5
    amp_max_consecutive_overflows: int = 16


def _ri_to_complex(t: torch.Tensor, m: int, n: int, s: int) -> torch.Tensor:
    ri = t.reshape(t.shape[0], m, n, s, 2)
    return torch.complex(ri[..., 0], ri[..., 1])


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


class Trainer:
    def __init__(
        self,
        model: ChannelModel,
        train_set: ChannelDataset,
        config: TrainConfig,
        checkpoint_meta: Optional[Dict] = None,
    ) -> None:
        self.config = config
        self.distributed = _dist_ready()
        self.rank = dist.get_rank() if self.distributed else 0
        self.world_size = dist.get_world_size() if self.distributed else 1

        if config.device:
            self.device = torch.device(config.device)
        elif torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device("cpu")
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

        self.module = model.to(self.device)
        self.spec = model.spec
        if self.distributed:
            self.model: torch.nn.Module = DistributedDataParallel(
                self.module,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
                output_device=self.device.index if self.device.type == "cuda" else None,
                find_unused_parameters=False,
                broadcast_buffers=False,
            )
        else:
            self.model = self.module

        self.criterion = ChannelLoss(
            self.spec,
            lambda_nmse=config.lambda_nmse,
            lambda_pas=config.lambda_pas,
            lambda_pdp=config.lambda_pdp,
            lambda_sparse=config.lambda_sparse,
            lambda_log_power=config.lambda_log_power,
            nmse_mode=config.nmse_mode,
        ).to(self.device)
        self.checkpoint_meta = checkpoint_meta or {}

        n_val = int(len(train_set) * config.val_fraction)
        n_train = len(train_set) - n_val
        split_seed = config.seed if config.split_seed is None else config.split_seed
        generator = torch.Generator().manual_seed(split_seed)
        if config.val_indices_file:
            val_idx = np.load(config.val_indices_file).astype(np.int64)
            val_idx = np.unique(val_idx)
            if len(val_idx) == 0:
                raise ValueError("val_indices_file contains no validation indices")
            if val_idx.min() < 0 or val_idx.max() >= len(train_set):
                raise ValueError(
                    "val_indices_file contains indices outside the training set")
            all_idx = np.arange(len(train_set), dtype=np.int64)
            train_idx = np.setdiff1d(all_idx, val_idx, assume_unique=False)
            self.train_split = Subset(train_set, train_idx.tolist())
            self.val_split = Subset(train_set, val_idx.tolist())
        elif n_val > 0 and config.val_split_mode == "spatial":
            train_idx, val_idx = self._spatial_split_indices(
                train_set.positions, n_val, config.val_regions, split_seed)
            self.train_split = Subset(train_set, train_idx.tolist())
            self.val_split = Subset(train_set, val_idx.tolist())
        elif n_val > 0:
            self.train_split, self.val_split = random_split(
                train_set, [n_train, n_val], generator=generator)
        else:
            self.train_split, self.val_split = train_set, None

        self.train_sampler = (
            DistributedSampler(
                self.train_split,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                seed=config.seed,
            )
            if self.distributed
            else None
        )
        self.val_sampler = (
            DistributedSampler(
                self.val_split,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
                drop_last=False,
            )
            if self.distributed and self.val_split is not None
            else None
        )
        loader_kwargs = dict(
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory and self.device.type == "cuda",
            persistent_workers=config.num_workers > 0,
        )
        self.train_loader = DataLoader(
            self.train_split,
            shuffle=self.train_sampler is None,
            sampler=self.train_sampler,
            drop_last=False,
            **loader_kwargs,
        )
        self.val_loader = (
            DataLoader(
                self.val_split,
                shuffle=False,
                sampler=self.val_sampler,
                drop_last=False,
                **loader_kwargs,
            )
            if self.val_split is not None
            else None
        )

        self.optimizer = torch.optim.AdamW(
            self.module.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        updates_per_epoch = max(
            1, math.ceil(len(self.train_loader) / max(config.accumulate_steps, 1)))
        self.total_updates = max(1, config.epochs * updates_per_epoch)
        self.warmup_updates = max(0, config.warmup_epochs * updates_per_epoch)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, self._lr_multiplier)

        use_amp = self.device.type == "cuda" and config.precision in {"bf16", "fp16"}
        self.amp_dtype = (
            torch.bfloat16 if config.precision == "bf16" else torch.float16)
        self.use_amp = use_amp
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=use_amp and config.precision == "fp16",
            init_scale=float(config.amp_init_scale),
            growth_interval=int(config.amp_growth_interval),
            backoff_factor=float(config.amp_backoff_factor),
        )
        self._consecutive_amp_overflows = 0
        self._total_amp_overflows = 0

        self.best_val = float("inf")
        self.best_epoch = 0
        self.best_state: Optional[Dict[str, torch.Tensor]] = None


    @staticmethod
    def _spatial_split_indices(
        positions, n_val: int, n_regions: int, seed: int
    ):
        """Select several compact validation regions to expose spatial overfit."""
        import numpy as np

        pos = np.asarray(positions, dtype=np.float32)
        n = len(pos)
        n_regions = max(1, min(int(n_regions), n_val))
        rng = np.random.default_rng(seed)
        # Farthest-point anchors, starting from a deterministic random point.
        anchors = [int(rng.integers(n))]
        min_dist = np.sum((pos - pos[anchors[0]]) ** 2, axis=1)
        for _ in range(1, n_regions):
            anchors.append(int(np.argmax(min_dist)))
            dist = np.sum((pos - pos[anchors[-1]]) ** 2, axis=1)
            min_dist = np.minimum(min_dist, dist)

        selected = np.zeros(n, dtype=bool)
        remaining = n_val
        for region_i, anchor in enumerate(anchors):
            quota = remaining // (len(anchors) - region_i)
            dist = np.sum((pos - pos[anchor]) ** 2, axis=1)
            order = np.argsort(dist)
            available = order[~selected[order]]
            chosen = available[:quota]
            selected[chosen] = True
            remaining -= len(chosen)
        if remaining > 0:
            available = np.flatnonzero(~selected)
            selected[available[:remaining]] = True
        val_idx = np.flatnonzero(selected)
        train_idx = np.flatnonzero(~selected)
        return train_idx, val_idx

    def _lr_multiplier(self, step: int) -> float:
        if self.warmup_updates > 0 and step < self.warmup_updates:
            return max(1e-8, (step + 1) / self.warmup_updates)
        progress = (step - self.warmup_updates) / max(
            1, self.total_updates - self.warmup_updates)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.config.min_lr_ratio + (1.0 - self.config.min_lr_ratio) * cosine

    def _autocast(self):
        if not self.use_amp:
            return nullcontext()
        return torch.autocast(
            device_type="cuda", dtype=self.amp_dtype, enabled=True)

    def _sparse_term(self) -> Optional[torch.Tensor]:
        if hasattr(self.module, "auxiliary_losses"):
            aux = self.module.auxiliary_losses()  # type: ignore[attr-defined]
            return aux.get("angle_delay_l1")
        return None

    def _reduce_stats(self, sums: Dict[str, float], count: int) -> Dict[str, float]:
        """Reduce epoch statistics and form NMSE as one global ratio.

        ``nmse_num`` and ``nmse_den`` are already power sums, so they must not
        be divided by the number of samples independently.  All other entries
        are sample-weighted means.
        """
        keys = sorted(sums)
        values = [sums[k] for k in keys] + [float(count)]
        tensor = torch.tensor(values, device=self.device, dtype=torch.float64)
        if self.distributed:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        reduced = {k: float(tensor[i]) for i, k in enumerate(keys)}
        total_count = max(float(tensor[-1]), 1.0)

        power_sums = {"nmse_num", "nmse_den", "pred_power", "cross_real"}
        stats = {
            key: value / total_count
            for key, value in reduced.items()
            if key not in power_sums
        }
        nmse_num = reduced.get("nmse_num", 0.0)
        nmse_den = max(reduced.get("nmse_den", 0.0), 1e-30)
        stats["nmse"] = nmse_num / nmse_den
        stats["pred_power_ratio"] = reduced.get("pred_power", 0.0) / nmse_den
        stats["cross_ratio"] = reduced.get("cross_real", 0.0) / nmse_den
        return stats

    def _epoch(self, loader: DataLoader, train: bool, epoch: int) -> Dict[str, float]:
        self.model.train(train)
        if train and self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
        if not train and self.val_sampler is not None:
            self.val_sampler.set_epoch(epoch)

        sums: Dict[str, float] = {}
        seen = 0
        accum = max(self.config.accumulate_steps, 1)
        if train:
            self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, (pos, target) in enumerate(loader):
            pos = pos.to(self.device, non_blocking=True)
            if train and self.config.position_noise_std > 0:
                pos = pos + torch.randn_like(pos) * self.config.position_noise_std
            target = target.to(self.device, non_blocking=True)
            gt_h = _ri_to_complex(
                target, self.spec.m, self.spec.n, self.spec.s)
            should_step = (
                (batch_idx + 1) % accum == 0 or batch_idx + 1 == len(loader))
            group_start = (batch_idx // accum) * accum
            actual_accum = min(accum, len(loader) - group_start)

            sync_context = nullcontext()
            if (
                train
                and self.distributed
                and not should_step
                and isinstance(self.model, DistributedDataParallel)
            ):
                sync_context = self.model.no_sync()

            grad_context = torch.enable_grad() if train else torch.no_grad()
            with grad_context, sync_context:
                # Attention/MLPs use mixed precision, while FFTs, powers, cosine
                # similarities and NMSE are accumulated in FP32/complex64.
                with self._autocast():
                    pred_h = self.model(pos)
                    sparse_term = self._sparse_term()
                with torch.autocast(device_type="cuda", enabled=False):
                    pred_h = pred_h.to(torch.complex64)
                    gt_h = gt_h.to(torch.complex64)
                    if sparse_term is not None:
                        sparse_term = sparse_term.float()
                    out = self.criterion(
                        pred_h, gt_h, sparse_term=sparse_term)
                    scaled_loss = out["loss"] / max(actual_accum, 1)

            if not torch.isfinite(out["loss"]):
                pred_finite = bool(
                    torch.isfinite(pred_h.real).all()
                    and torch.isfinite(pred_h.imag).all())
                gt_finite = bool(
                    torch.isfinite(gt_h.real).all()
                    and torch.isfinite(gt_h.imag).all())
                raise FloatingPointError(
                    "non-finite loss detected at "
                    f"epoch={epoch}, batch={batch_idx}; "
                    f"pred_finite={pred_finite}, gt_finite={gt_finite}, "
                    f"pos_abs_max={float(pos.abs().max()):.3e}, "
                    f"pred_abs_max={float(torch.nan_to_num(pred_h.abs()).max()):.3e}, "
                    f"gt_abs_max={float(torch.nan_to_num(gt_h.abs()).max()):.3e}. "
                    "Check position/map normalisation and mixed precision."
                )

            if train:
                self.scaler.scale(scaled_loss).backward()
                if should_step:
                    grad_norm = None
                    if self.scaler.is_enabled():
                        # AMP overflows are expected occasionally, especially at
                        # the first update. Unscale before clipping, then let
                        # GradScaler inspect the gradients and skip the optimizer
                        # step while reducing its scale. Raising here would prevent
                        # GradScaler from doing the recovery it is designed for.
                        self.scaler.unscale_(self.optimizer)
                        if self.config.grad_clip:
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                self.module.parameters(), self.config.grad_clip)

                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        # On an overflow GradScaler skips optimizer.step() and
                        # backs off the scale. On success it stays equal or grows.
                        optimizer_stepped = scale_after >= scale_before
                        if optimizer_stepped:
                            self._consecutive_amp_overflows = 0
                        else:
                            self._consecutive_amp_overflows += 1
                            self._total_amp_overflows += 1
                            if self.rank == 0:
                                norm_text = (
                                    "unknown" if grad_norm is None
                                    else f"{float(grad_norm):.3e}"
                                )
                                print(
                                    "[amp] skipped optimizer update due to "
                                    f"non-finite gradients at epoch={epoch}, "
                                    f"batch={batch_idx}; grad_norm={norm_text}, "
                                    f"scale={scale_before:.1f}->{scale_after:.1f}",
                                    flush=True,
                                )
                            limit = max(1, int(
                                self.config.amp_max_consecutive_overflows))
                            if self._consecutive_amp_overflows >= limit:
                                raise FloatingPointError(
                                    "AMP gradients remained non-finite for "
                                    f"{limit} consecutive optimizer updates. "
                                    "Try train.precision=bf16 (when supported), "
                                    "or reduce train.lr / train.amp_init_scale.")
                    else:
                        if self.config.grad_clip:
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                self.module.parameters(),
                                self.config.grad_clip,
                                error_if_nonfinite=False,
                            )
                            if not torch.isfinite(grad_norm):
                                raise FloatingPointError(
                                    "non-finite unscaled gradient norm at "
                                    f"epoch={epoch}, batch={batch_idx}: "
                                    f"{float(grad_norm)}")
                        self.optimizer.step()
                        optimizer_stepped = True

                    self.optimizer.zero_grad(set_to_none=True)
                    if optimizer_stepped:
                        self.scheduler.step()

            bs = pos.shape[0]
            seen += bs
            for key, value in out.items():
                scalar = float(value.detach())
                if key in {"nmse_num", "nmse_den", "pred_power", "cross_real"}:
                    # These are full tensor power sums for the batch.
                    sums[key] = sums.get(key, 0.0) + scalar
                else:
                    sums[key] = sums.get(key, 0.0) + scalar * bs

        return self._reduce_stats(sums, seen)

    def fit(self) -> List[Dict[str, float]]:
        history: List[Dict[str, float]] = []
        stale_epochs = 0
        for epoch in range(1, self.config.epochs + 1):
            train_stats = self._epoch(self.train_loader, True, epoch)
            record = {
                "epoch": epoch,
                "lr": self.optimizer.param_groups[0]["lr"],
                **{f"train_{k}": v for k, v in train_stats.items()},
            }
            if self.val_loader is not None:
                val_stats = self._epoch(self.val_loader, False, epoch)
                record.update({f"val_{k}": v for k, v in val_stats.items()})

                w1, w2, w3 = map(float, self.spec.metric_weights)

                competition_score = (
                    w1 * val_stats["pas"]
                    + w2 * val_stats["pdp"]
                    + w3 / (1.0 + val_stats["nmse"])
                )

                record["val_score"] = float(competition_score)

                # 现有 best_val 逻辑是数值越小越好，因此取负数。
                score = -float(competition_score)

            else:
                w1, w2, w3 = map(float, self.spec.metric_weights)

                competition_score = (
                    w1 * train_stats["pas"]
                    + w2 * train_stats["pdp"]
                    + w3 / (1.0 + train_stats["nmse"])
                )

                record["train_score"] = float(competition_score)
                score = -float(competition_score)
            history.append(record)

            if score < self.best_val:
                self.best_val = score
                self.best_epoch = epoch
                stale_epochs = 0
                if self.rank == 0:
                    self.best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in self.module.state_dict().items()
                    }
            else:
                stale_epochs += 1

            if self.rank == 0 and (
                epoch == 1
                or epoch % self.config.log_every == 0
                or epoch == self.config.epochs
            ):
                self._log(record)

            patience = self.config.early_stopping_patience
            if patience > 0 and stale_epochs >= patience:
                if self.rank == 0:
                    print(
                        f"[trainer] early stop at epoch {epoch}; "
                        f"best epoch={self.best_epoch}",
                        flush=True,
                    )
                break

        if self.distributed:
            dist.barrier()
        return history

    @staticmethod
    def _log(record: Dict[str, float]) -> None:
        parts = [f"epoch {record['epoch']:>4d}", f"lr={record['lr']:.3e}"]
        for key in (
            "train_loss", "train_nmse", "train_pas", "train_pdp",
            "train_pred_power_ratio", "train_cross_ratio", "train_sparse",
            "train_log_power", "train_score",
            "val_loss", "val_nmse", "val_pas", "val_pdp",
            "val_pred_power_ratio", "val_cross_ratio", "val_sparse",
            "val_log_power", "val_score",
        ):
            if key in record:
                parts.append(f"{key}={record[key]:.4f}")
        print(" | ".join(parts), flush=True)

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        if self.rank != 0:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = self.best_state or {
            k: v.detach().cpu() for k, v in self.module.state_dict().items()
        }
        payload = {
            "model_state": state,
            "meta": self.checkpoint_meta,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val,
            "train_config": copy.deepcopy(self.config.__dict__),
        }
        torch.save(payload, str(path))
        print(
            f"[trainer] saved best checkpoint (epoch {self.best_epoch}) -> {path}",
            flush=True,
        )
