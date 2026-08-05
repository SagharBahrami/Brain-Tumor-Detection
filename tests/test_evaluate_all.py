from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from scripts import evaluate_all


class _LogitModel(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return images[:, :1]


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)


def test_find_best_checkpoint_prefers_newest_best(tmp_path: Path) -> None:
    first = (
        tmp_path
        / "tumor_detection"
        / "resnet50_20260101_000000"
        / "checkpoints"
        / "resnet50_best-epoch=01.ckpt"
    )
    second = (
        tmp_path
        / "tumor_detection"
        / "resnet50_20260102_000000"
        / "checkpoints"
        / "resnet50_best-epoch=02.ckpt"
    )
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.touch()
    second.touch()
    os.utime(first, (1, 1))
    os.utime(second, (2, 2))

    assert evaluate_all.find_best_checkpoint(tmp_path, "resnet50") == second


def test_find_best_checkpoint_rejects_missing_model(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No best checkpoint"):
        evaluate_all.find_best_checkpoint(tmp_path, "resnet50")


def test_load_trained_model_removes_lightning_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = tmp_path / "model.ckpt"
    torch.save(
        {
            "state_dict": {
                "model.linear.weight": torch.tensor([[3.0]]),
                "model.linear.bias": torch.tensor([2.0]),
            }
        },
        checkpoint_path,
    )
    monkeypatch.setattr(evaluate_all, "create_model", lambda _: _TinyModel())

    model = evaluate_all.load_trained_model(
        "resnet50",
        checkpoint_path,
        torch.device("cpu"),
    )

    assert torch.equal(model.linear.weight, torch.tensor([[3.0]]))
    assert torch.equal(model.linear.bias, torch.tensor([2.0]))
    assert model.training is False


def test_evaluate_model_collects_flat_slice_metrics() -> None:
    test_loader = [
        {
            "images": torch.tensor([[[-2.0], [-1.0], [1.0], [2.0]]]),
            "labels": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
        }
    ]

    metrics = evaluate_all.evaluate_model(
        _LogitModel(),
        test_loader,  # type: ignore[arg-type]
        torch.device("cpu"),
    )

    assert metrics["samples"] == 4
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["auc"] == pytest.approx(1.0)
    assert (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) == (
        2,
        0,
        0,
        2,
    )
