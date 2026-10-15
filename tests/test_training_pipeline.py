"""Deterministic CPU tests of training, checkpointing, and data integration."""

import json

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.models import build_model
from src.training.config import build_dataset, build_loaders, make_model
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


def make_tiny_trainer(tmp_path, *, epochs=2, seed=17, config=None):
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
    )


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
