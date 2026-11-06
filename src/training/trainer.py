"""Reproducible supervised segmentation training and validation engine."""

from __future__ import annotations

import json
import os
import random
import copy
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.evaluation import SegmentationMetricAccumulator


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, torch, and CUDA; optionally require deterministic ops."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _valid_pixels(batch: dict[str, Any], ignore_index: int | None, device: torch.device):
    target = batch["mask"].to(device=device, dtype=torch.long)
    valid = batch["valid_mask"].to(device=device, dtype=torch.bool)
    if valid.shape != target.shape:
        raise ValueError("batch valid_mask must match mask shape")
    if ignore_index is not None:
        valid = valid & (target != ignore_index)
    invalid = valid & (target < 0)
    if bool(invalid.any()):
        raise ValueError("valid target pixels must have nonnegative class IDs")
    return target, valid


def _loss_sum(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
              num_classes: int) -> tuple[torch.Tensor, int]:
    count = int(valid.sum().item())
    if count == 0:
        return logits.sum() * 0.0, 0
    if bool((valid & (target >= num_classes)).any()):
        raise ValueError("valid target labels must be smaller than num_classes")
    safe_target = target.masked_fill(~valid, 0)
    per_pixel = F.cross_entropy(logits, safe_target, reduction="none")
    return (per_pixel * valid.to(per_pixel.dtype)).sum(), count


class SegmentationTrainer:
    """Train/validate one model. There is deliberately no test-loader argument."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        *,
        num_classes: int,
        epochs: int,
        checkpoint_dir: str | Path,
        config: dict[str, Any],
        device: str | torch.device = "auto",
        ignore_index: int | None = None,
        scheduler: Any | None = None,
        metric_options: dict[str, Any] | None = None,
        full_metrics_every_epochs: int | None = None,
        full_metrics_at_end: bool = True,
        early_stopping: dict[str, Any] | None = None,
    ) -> None:
        if num_classes < 2 or epochs < 1:
            raise ValueError("num_classes must be >= 2 and epochs must be positive")
        self.device = resolve_device(str(device)) if not isinstance(device, torch.device) else device
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.num_classes = num_classes
        self.epochs = epochs
        self.ignore_index = ignore_index
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.metric_options = metric_options or {}
        if (full_metrics_every_epochs is not None and
                (not isinstance(full_metrics_every_epochs, int) or isinstance(full_metrics_every_epochs, bool)
                 or full_metrics_every_epochs < 1)):
            raise ValueError("full_metrics_every_epochs must be positive or None")
        if not isinstance(full_metrics_at_end, bool):
            raise TypeError("full_metrics_at_end must be a boolean")
        self.full_metrics_every_epochs = full_metrics_every_epochs
        self.full_metrics_at_end = bool(full_metrics_at_end)
        early_stopping = early_stopping or {"enabled": False}
        if not isinstance(early_stopping, dict):
            raise TypeError("early_stopping must be a mapping")
        enabled = early_stopping.get("enabled", False)
        if not isinstance(enabled, bool):
            raise TypeError("early_stopping.enabled must be a boolean")
        self.early_stopping_enabled = enabled
        self.early_stopping_patience: int | None = None
        self.early_stopping_min_delta: float | None = None
        if self.early_stopping_enabled:
            patience = early_stopping.get("patience")
            min_delta = early_stopping.get("min_delta")
            if not isinstance(patience, int) or isinstance(patience, bool) or patience < 1:
                raise ValueError("enabled Early Stopping requires positive integer patience")
            if (not isinstance(min_delta, (int, float)) or isinstance(min_delta, bool)
                    or not math.isfinite(float(min_delta)) or min_delta < 0):
                raise ValueError("enabled Early Stopping requires finite nonnegative min_delta")
            self.early_stopping_patience = patience
            self.early_stopping_min_delta = float(min_delta)
        self.early_stopping_best_miou: float | None = None
        self.epochs_without_improvement = 0
        self.early_stopping_stopped = False
        self.best_miou = float("-inf")
        self.start_epoch = 0
        self.log_path = self.checkpoint_dir / "metrics.jsonl"

    def train_epoch(self) -> dict[str, float | int]:
        self.model.train()
        loss_sum, valid_count, skipped = 0.0, 0, 0
        for batch in self.train_loader:
            images = batch["image"].to(self.device, dtype=torch.float32, non_blocking=True)
            target, valid = _valid_pixels(batch, self.ignore_index, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(images)
            batch_sum, count = _loss_sum(logits, target, valid, self.num_classes)
            if count == 0:
                skipped += 1
                continue
            (batch_sum / count).backward()
            self.optimizer.step()
            loss_sum += float(batch_sum.detach().item())
            valid_count += count
        if valid_count == 0:
            raise ValueError("training epoch contained no valid pixels")
        return {"loss": loss_sum / valid_count, "valid_pixels": valid_count, "skipped_batches": skipped}

    @torch.no_grad()
    def validate(self, *, full_metrics: bool = True) -> dict[str, Any]:
        self.model.eval()
        loss_sum, valid_count = 0.0, 0
        accumulator = SegmentationMetricAccumulator(
            self.num_classes, input_type="logits", ignore_index=self.ignore_index,
            **self.metric_options,
        ) if full_metrics else None
        confusion = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64)
        for batch in self.validation_loader:
            images = batch["image"].to(self.device, dtype=torch.float32, non_blocking=True)
            target, valid = _valid_pixels(batch, self.ignore_index, self.device)
            logits = self.model(images)
            batch_sum, count = _loss_sum(logits, target, valid, self.num_classes)
            if count == 0:
                continue
            loss_sum += float(batch_sum.item())
            valid_count += count
            if full_metrics:
                accumulator.update(logits, target, valid_mask=valid)
            else:
                prediction = logits.argmax(dim=1)
                encoded = target[valid] * self.num_classes + prediction[valid]
                confusion += torch.bincount(encoded.cpu(), minlength=self.num_classes ** 2).reshape(
                    self.num_classes, self.num_classes
                )
        if valid_count == 0:
            raise ValueError("validation epoch contained no valid pixels")
        result: dict[str, float | int | None] = {
            "loss": loss_sum / valid_count,
            "valid_pixels": valid_count,
        }
        if accumulator is not None:
            result.update(accumulator.compute())
        else:
            true_positive = confusion.diag().double()
            union = confusion.sum(0).double() + confusion.sum(1).double() - true_positive
            present = union > 0
            if not bool(present.any()):
                raise ValueError("mIoU is undefined because every class has zero union")
            result["mIoU"] = float((true_positive[present] / union[present]).mean().item())
        return result

    def _rng_state(self) -> dict[str, Any]:
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        state["train_loader_generator"] = (
            self.train_loader.generator.get_state()
            if getattr(self.train_loader, "generator", None) is not None else None
        )
        state["validation_loader_generator"] = (
            self.validation_loader.generator.get_state()
            if getattr(self.validation_loader, "generator", None) is not None else None
        )
        return state

    def _early_stopping_state(self) -> dict[str, Any] | None:
        if not self.early_stopping_enabled:
            return None
        return {
            "best_miou": self.early_stopping_best_miou,
            "epochs_without_improvement": self.epochs_without_improvement,
            "stopped": self.early_stopping_stopped,
        }

    def _update_early_stopping(self, miou: float) -> bool:
        """Return true when patience has been exhausted.

        The first finite score establishes the reference. Later scores qualify
        only when they improve it by at least min_delta.
        """
        if not self.early_stopping_enabled:
            return False
        if not math.isfinite(miou):
            raise ValueError("checkpoint-selection mIoU must be finite for Early Stopping")
        if self.early_stopping_best_miou is None:
            self.early_stopping_best_miou = miou
            self.epochs_without_improvement = 0
        elif miou >= self.early_stopping_best_miou + float(self.early_stopping_min_delta):
            self.early_stopping_best_miou = miou
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        self.early_stopping_stopped = (
            self.epochs_without_improvement >= int(self.early_stopping_patience)
        )
        return self.early_stopping_stopped

    def _restore_rng(self, state: dict[str, Any]) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"].cpu())
        if state.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
        for loader_name, key in (("train_loader", "train_loader_generator"),
                                 ("validation_loader", "validation_loader_generator")):
            loader = getattr(self, loader_name)
            generator_state = state.get(key)
            if generator_state is not None and loader.generator is not None:
                loader.generator.set_state(generator_state.cpu())

    def save_checkpoint(self, path: str | Path, epoch: int) -> None:
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "epoch": epoch,
            "configuration": self.config,
            "best_validation_miou": self.best_miou,
            "early_stopping": self._early_stopping_state(),
            "rng_states": self._rng_state(),
        }
        path = Path(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)

    def resume(self, path: str | Path, *, strict_config: bool = True) -> None:
        # Keep RNG and DataLoader generator ByteTensors on CPU; module and
        # optimizer loading copy state to the active device as needed.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        saved_config = copy.deepcopy(checkpoint["configuration"])
        current_config = copy.deepcopy(self.config)
        for value in (saved_config, current_config):
            value.pop("resume", None)
            for section in ("training", "resolved_training"):
                if isinstance(value.get(section), dict):
                    value[section].pop("resume", None)
        if strict_config and saved_config != current_config:
            raise ValueError("resume configuration differs from checkpoint configuration")
        if (checkpoint["scheduler"] is None) != (self.scheduler is None):
            raise ValueError("checkpoint scheduler configuration does not match this run")
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.best_miou = float(checkpoint["best_validation_miou"])
        saved_early_stopping = checkpoint.get("early_stopping")
        if self.early_stopping_enabled:
            if not isinstance(saved_early_stopping, dict):
                raise ValueError("checkpoint has no Early Stopping state; exact resume is unavailable")
            best_score = saved_early_stopping.get("best_miou")
            counter = saved_early_stopping.get("epochs_without_improvement")
            stopped = saved_early_stopping.get("stopped")
            if (best_score is not None and not math.isfinite(float(best_score))):
                raise ValueError("checkpoint Early Stopping metric is invalid")
            if not isinstance(counter, int) or isinstance(counter, bool) or counter < 0:
                raise ValueError("checkpoint Early Stopping counter is invalid")
            if not isinstance(stopped, bool):
                raise ValueError("checkpoint Early Stopping stopped flag is invalid")
            if (counter >= int(self.early_stopping_patience)) != stopped:
                raise ValueError("checkpoint Early Stopping counter and stopped flag disagree")
            self.early_stopping_best_miou = None if best_score is None else float(best_score)
            self.epochs_without_improvement = counter
            self.early_stopping_stopped = stopped
        elif saved_early_stopping is not None:
            raise ValueError("checkpoint enables Early Stopping but current configuration does not")
        self._restore_rng(checkpoint["rng_states"])

    def fit(self) -> list[dict[str, Any]]:
        history: list[dict[str, Any]] = []
        if self.early_stopping_stopped:
            return history
        for epoch in range(self.start_epoch, self.epochs):
            train_metrics = self.train_epoch()
            scheduled = (self.full_metrics_every_epochs is not None and
                         (epoch + 1) % self.full_metrics_every_epochs == 0)
            final = epoch + 1 == self.epochs
            validation_metrics = self.validate(full_metrics=scheduled or (final and self.full_metrics_at_end))
            miou = validation_metrics["mIoU"]
            if miou is None or not math.isfinite(float(miou)):
                raise ValueError("checkpoint-selection validation mIoU must be finite")
            stop = self._update_early_stopping(float(miou))
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(float(miou))
                else:
                    self.scheduler.step()
            record = {
                "epoch": epoch,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "train": train_metrics,
                "validation": validation_metrics,
                "early_stopping": self._early_stopping_state(),
            }
            history.append(record)
            with self.log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            if float(miou) > self.best_miou:
                self.best_miou = float(miou)
                self.save_checkpoint(self.checkpoint_dir / "best.pt", epoch)
            self.save_checkpoint(self.checkpoint_dir / "latest.pt", epoch)
            if stop:
                break
        return history
