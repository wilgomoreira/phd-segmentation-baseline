"""Deterministic CPU tests of training, checkpointing, and data integration."""

import json

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.models import build_model
from src.training.config import build_dataset, build_loaders, make_model, make_scheduler
from src.training.trainer import SegmentationTrainer, set_seed
from src.training.runner import create_experiment_directory


class TinyPixels(Dataset):
    def __init__(self, offset=0):
        self.images = torch.tensor([
            [[[0.0, 1.0], [0.2, 0.9]]],
            [[[1.0, 0.0], [0.8, 0.1]]],
        ])
        self.masks = torch.tensor([
            [[0, 1], [255, 1]],
            [[1, 0], [1, 0]],
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        mask = self.masks[index]
        return {
            "image": self.images[index],
            "mask": mask,
            "valid_mask": mask != 255,
            "metadata": {"sample_id": index},
        }


def make_tiny_trainer(tmp_path, *, epochs=2, seed=17, config=None, early_stopping=None, scheduler=None):
    set_seed(seed)
    train_gen = torch.Generator().manual_seed(seed)
    val_gen = torch.Generator().manual_seed(seed + 1)
    train_loader = DataLoader(TinyPixels(), batch_size=2, shuffle=True, generator=train_gen)
    val_loader = DataLoader(TinyPixels(), batch_size=2, shuffle=False, generator=val_gen)
    model = nn.Conv2d(1, 2, kernel_size=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
    return SegmentationTrainer(
        model, optimizer, train_loader, val_loader,
        num_classes=2, epochs=epochs, checkpoint_dir=tmp_path,
        config=config or {"test_run": "tiny", "epochs": 2},
        device="cpu", ignore_index=255,
        early_stopping=early_stopping, scheduler=scheduler,
    )


def set_scripted_epochs(trainer, scores):
    iterator = iter(scores)
    def scripted_train_epoch():
        trainer.optimizer.step()
        return {"loss": 1.0, "valid_pixels": 7, "skipped_batches": 0}
    trainer.train_epoch = scripted_train_epoch
    trainer.validate = lambda *, full_metrics=True: {
        "loss": 1.0, "valid_pixels": 7, "mIoU": next(iterator),
    }


def test_train_step_validation_and_ignored_pixels(tmp_path):
    trainer = make_tiny_trainer(tmp_path)
    original = [p.detach().clone() for p in trainer.model.parameters()]
    training = trainer.train_epoch()
    assert training["valid_pixels"] == 7
    assert training["loss"] > 0
    assert any(not torch.equal(a, b) for a, b in zip(original, trainer.model.parameters()))
    assert all(parameter.grad is not None for parameter in trainer.model.parameters())

    for parameter in trainer.model.parameters():
        parameter.grad = None
    validation = trainer.validate()
    assert validation["valid_pixels"] == 7
    assert validation["mIoU"] is not None
    assert validation["loss"] > 0
    assert all(parameter.grad is None for parameter in trainer.model.parameters())


def test_checkpoint_round_trip_resume_and_best_selection(tmp_path):
    uninterrupted_dir = tmp_path / "uninterrupted"
    uninterrupted = make_tiny_trainer(uninterrupted_dir, epochs=2)
    full_history = uninterrupted.fit()
    expected = {k: v.detach().clone() for k, v in uninterrupted.model.state_dict().items()}
    best_epoch = max(full_history, key=lambda row: row["validation"]["mIoU"])["epoch"]
    best_payload = torch.load(uninterrupted_dir / "best.pt", weights_only=False)
    assert best_payload["epoch"] == best_epoch

    staged_dir = tmp_path / "staged"
    stage1 = make_tiny_trainer(staged_dir, epochs=1)
    stage1.fit()
    stage2 = make_tiny_trainer(staged_dir, epochs=2)
    stage2.resume(staged_dir / "latest.pt")
    assert stage2.start_epoch == 1
    stage2.fit()
    actual = stage2.model.state_dict()
    assert all(torch.equal(expected[k], actual[k]) for k in expected)


def test_resume_restores_python_numpy_torch_and_loader_rng(tmp_path):
    trainer = make_tiny_trainer(tmp_path / "run", epochs=1)
    trainer.save_checkpoint(trainer.checkpoint_dir / "state.pt", 0)
    expected = (torch.rand(1), np.random.rand(), __import__("random").random(),
                trainer.train_loader.generator.get_state().clone())
    resumed = make_tiny_trainer(tmp_path / "other", epochs=1)
    resumed.resume(trainer.checkpoint_dir / "state.pt")
    actual = (torch.rand(1), np.random.rand(), __import__("random").random(),
              resumed.train_loader.generator.get_state())
    assert torch.equal(expected[0], actual[0])
    assert expected[1:3] == actual[1:3]
    assert torch.equal(expected[3], actual[3])


def test_experiment_directory_is_unique_and_resume_reuses_checkpoint_directory(tmp_path):
    first = create_experiment_directory(tmp_path / "runs")
    second = create_experiment_directory(tmp_path / "runs")
    assert first != second and first.is_dir() and second.is_dir()
    checkpoint = first / "latest.pt"
    checkpoint.write_bytes(b"synthetic")
    assert create_experiment_directory(tmp_path / "runs", resume=checkpoint) == first


def test_validation_schedule_keeps_miou_each_epoch_and_full_metrics_at_end(tmp_path):
    trainer = make_tiny_trainer(tmp_path, epochs=3)
    trainer.full_metrics_every_epochs = None
    trainer.full_metrics_at_end = True
    history = trainer.fit()
    assert all("mIoU" in row["validation"] for row in history)
    assert "AUROC" not in history[0]["validation"]
    assert "AUROC" not in history[1]["validation"]
    assert "AUROC" in history[2]["validation"]


def test_early_stopping_min_delta_best_checkpoint_and_patience(tmp_path):
    trainer = make_tiny_trainer(
        tmp_path, epochs=30,
        early_stopping={"enabled": True, "patience": 3, "min_delta": 0.001},
    )
    set_scripted_epochs(trainer, [0.5, 0.5005, 0.502, 0.5025, 0.503, 0.5031, 0.5032, 0.5033])
    history = trainer.fit()
    # A sub-threshold rise is not an Early Stopping improvement, but it still
    # becomes the best raw-mIoU checkpoint.
    assert [row["early_stopping"]["epochs_without_improvement"] for row in history] == [0, 1, 0, 1, 0, 1, 2, 3]
    assert len(history) == 8
    best = torch.load(tmp_path / "best.pt", weights_only=False)
    latest = torch.load(tmp_path / "latest.pt", weights_only=False)
    assert best["best_validation_miou"] == pytest.approx(0.5033)
    assert best["epoch"] == 7
    assert latest["early_stopping"]["stopped"] is True
    assert latest["early_stopping"]["epochs_without_improvement"] == 3


def test_early_stopping_resumes_counter_optimizer_scheduler_and_metric(tmp_path):
    early = {"enabled": True, "patience": 3, "min_delta": 0.001}
    config = {"test_run": "early-resume", "epochs": 6, "early_stopping": early}
    first = make_tiny_trainer(
        tmp_path, epochs=2, config=config, early_stopping=early,
    )
    first.scheduler = torch.optim.lr_scheduler.StepLR(first.optimizer, step_size=2, gamma=0.5)
    set_scripted_epochs(first, [0.4, 0.4])
    first.fit()
    saved = torch.load(tmp_path / "latest.pt", weights_only=False)
    assert saved["early_stopping"]["epochs_without_improvement"] == 1
    assert saved["scheduler"]["last_epoch"] == 2

    resumed = make_tiny_trainer(
        tmp_path / "resumed", epochs=6, config=config, early_stopping=early,
    )
    resumed.scheduler = torch.optim.lr_scheduler.StepLR(resumed.optimizer, step_size=2, gamma=0.5)
    resumed.resume(tmp_path / "latest.pt")
    assert resumed.early_stopping_best_miou == pytest.approx(0.4)
    assert resumed.epochs_without_improvement == 1
    assert resumed.scheduler.state_dict() == saved["scheduler"]
    set_scripted_epochs(resumed, [0.4, 0.4])
    history = resumed.fit()
    assert len(history) == 2
    assert history[-1]["early_stopping"]["stopped"] is True


def test_scheduler_advances_once_per_epoch_up_to_100_epoch_max(tmp_path):
    trainer = make_tiny_trainer(
        tmp_path, epochs=100,
        early_stopping={"enabled": True, "patience": 15, "min_delta": 0.001},
    )
    trainer.scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, step_size=20, gamma=0.5)
    set_scripted_epochs(trainer, [0.01 + i * 0.01 for i in range(100)])
    history = trainer.fit()
    assert len(history) == 100
    assert trainer.scheduler.last_epoch == 100
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.01 * 0.5 ** 5)


def test_polynomial_scheduler_resume_matches_uninterrupted_100_epoch_sequence():
    def create():
        parameter = nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=1e-4, weight_decay=1e-4)
        scheduler = make_scheduler(optimizer, {
            "scheduler": {"name": "polynomial", "parameters": {"total_iters": 100, "power": 0.9}},
        })
        return optimizer, scheduler

    full_optimizer, full_scheduler = create()
    full_rates = []
    for _ in range(100):
        full_optimizer.step()
        full_scheduler.step()
        full_rates.append(full_optimizer.param_groups[0]["lr"])

    staged_optimizer, staged_scheduler = create()
    for _ in range(37):
        staged_optimizer.step()
        staged_scheduler.step()
    optimizer_state = staged_optimizer.state_dict()
    scheduler_state = staged_scheduler.state_dict()
    resumed_optimizer, resumed_scheduler = create()
    resumed_optimizer.load_state_dict(optimizer_state)
    resumed_scheduler.load_state_dict(scheduler_state)
    resumed_rates = []
    for _ in range(63):
        resumed_optimizer.step()
        resumed_scheduler.step()
        resumed_rates.append(resumed_optimizer.param_groups[0]["lr"])
    assert resumed_rates == pytest.approx(full_rates[37:], rel=0, abs=1e-18)
    assert full_rates[0] == pytest.approx(1e-4 * (0.99 ** 0.9))
    assert full_rates[-1] == 0.0


def test_polynomial_scheduler_checkpoint_resume_matches_trainer_epoch_sequence(tmp_path):
    config = {"test_run": "polynomial-resume", "epochs": 6,
             "scheduler": {"name": "polynomial", "parameters": {"total_iters": 100, "power": 0.9}}}
    full = make_tiny_trainer(tmp_path / "full", epochs=6, config=config)
    full.scheduler = make_scheduler(full.optimizer, config)
    set_scripted_epochs(full, [0.1 + i * 0.01 for i in range(6)])
    full_history = full.fit()

    staged = make_tiny_trainer(tmp_path / "staged", epochs=3, config=config)
    staged.scheduler = make_scheduler(staged.optimizer, config)
    set_scripted_epochs(staged, [0.1, 0.11, 0.12])
    staged_history = staged.fit()
    saved = torch.load(tmp_path / "staged" / "latest.pt", map_location="cpu", weights_only=False)
    assert saved["scheduler"]["last_epoch"] == 3

    resumed = make_tiny_trainer(tmp_path / "resumed", epochs=6, config=config)
    resumed.scheduler = make_scheduler(resumed.optimizer, config)
    resumed.resume(tmp_path / "staged" / "latest.pt")
    set_scripted_epochs(resumed, [0.13, 0.14, 0.15])
    resumed_history = resumed.fit()
    rates = [row["learning_rate"] for row in full_history]
    combined_rates = [row["learning_rate"] for row in staged_history] + [
        row["learning_rate"] for row in resumed_history
    ]
    assert combined_rates == pytest.approx(rates, rel=0, abs=1e-18)


def test_polynomial_scheduler_configuration_validation():
    parameter = nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4)
    assert isinstance(make_scheduler(optimizer, {
        "scheduler": {"name": "polynomial", "parameters": {"total_iters": 100, "power": 0.9}},
    }), torch.optim.lr_scheduler.PolynomialLR)
    with pytest.raises(ValueError, match="positive total_iters"):
        make_scheduler(optimizer, {
            "scheduler": {"name": "polynomial", "parameters": {"total_iters": 0, "power": 0.9}},
        })


def test_early_stopping_configuration_and_legacy_checkpoint_validation(tmp_path):
    with pytest.raises(ValueError, match="patience"):
        make_tiny_trainer(tmp_path / "bad", early_stopping={"enabled": True, "patience": 0, "min_delta": 0.001})
    old = make_tiny_trainer(tmp_path / "old", epochs=1)
    old.save_checkpoint(tmp_path / "old.pt", 0)
    enabled = make_tiny_trainer(
        tmp_path / "new", epochs=1,
        early_stopping={"enabled": True, "patience": 2, "min_delta": 0.001},
    )
    with pytest.raises(ValueError, match="no Early Stopping state"):
        enabled.resume(tmp_path / "old.pt")


def test_early_stopping_real_synthetic_training_smoke(tmp_path):
    trainer = make_tiny_trainer(
        tmp_path, epochs=2,
        early_stopping={"enabled": True, "patience": 15, "min_delta": 0.001},
    )
    history = trainer.fit()
    assert len(history) == 2
    assert all(np.isfinite(row["train"]["loss"]) for row in history)
    assert all(np.isfinite(row["validation"]["mIoU"]) for row in history)
    assert (tmp_path / "best.pt").is_file()
    assert (tmp_path / "latest.pt").is_file()
    restored = make_tiny_trainer(
        tmp_path / "restored", epochs=2,
        early_stopping={"enabled": True, "patience": 15, "min_delta": 0.001},
    )
    restored.resume(tmp_path / "latest.pt")
    assert restored.start_epoch == 2
    assert restored.early_stopping_best_miou == pytest.approx(trainer.early_stopping_best_miou)


def test_dataset_model_evaluator_integration_and_test_split_exclusion(tmp_path):
    image_train = np.random.default_rng(4).normal(size=(32, 32, 4)).astype(np.float32)
    image_val = np.random.default_rng(5).normal(size=(32, 32, 4)).astype(np.float32)
    mask = np.zeros((32, 32), dtype=np.int64)
    np.save(tmp_path / "train.npy", image_train)
    np.save(tmp_path / "validation.npy", image_val)
    np.save(tmp_path / "mask.npy", mask)
    rows = [
        {"image": "train.npy", "mask": "mask.npy", "image_layout": "HWC", "split": "train", "image_id": "tr"},
        {"image": "validation.npy", "mask": "mask.npy", "image_layout": "HWC", "split": "validation", "image_id": "va"},
        {"image": "train.npy", "mask": "mask.npy", "image_layout": "HWC", "split": "test", "image_id": "te"},
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    data_config = {
        "dataset": "five_billion_pixels", "manifest": str(manifest),
        "channel_count": 4, "splits": {"train": "train", "validation": "validation"},
        "ignore_index": None,
    }
    assert len(build_dataset(data_config, "train")) == 1
    with pytest.raises(ValueError, match="only train and validation"):
        build_dataset(data_config, "test")

    model_config = {
        "name": "segformer_b1", "num_classes": None, "pretrained": False,
        "in_channels_by_dataset": {"five_billion_pixels": 4},
    }
    model = make_model(model_config, data_config, num_classes=2)
    train_loader, val_loader = build_loaders(data_config, batch_size=1, num_workers=0,
                                             seed=22, pin_memory=False)
    assert train_loader.dataset.records[0][0]["image_id"] == "tr"
    assert val_loader.dataset.records[0][0]["image_id"] == "va"
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    trainer = SegmentationTrainer(
        model, optimizer, train_loader, val_loader, num_classes=2, epochs=1,
        checkpoint_dir=tmp_path / "run", config={"integration": True},
        device="cpu",
    )
    history = trainer.fit()
    assert history[0]["train"]["valid_pixels"] == 1024
    assert history[0]["validation"]["valid_pixels"] == 1024
    assert "mIoU" in history[0]["validation"]


def test_factory_rejects_configuration_mismatch():
    with pytest.raises(ValueError, match="conflicts"):
        build_model({"name": "segformer_b1", "num_classes": 3}, num_classes=2, in_channels=4)
