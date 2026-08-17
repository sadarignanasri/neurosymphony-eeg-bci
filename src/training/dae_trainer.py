"""
Neuro-Symphony Phase 2 — DAE Training.

Reusable training loop for EEG and fNIRS denoising
autoencoders.

Training objective:

    noisy signal -> DAE -> clean signal

Loss:
    Mean Squared Error (MSE)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class EpochMetrics:
    """Metrics recorded for one training epoch."""

    epoch: int
    train_loss: float
    validation_loss: float


@dataclass(frozen=True)
class TrainingResult:
    """Complete result returned by DAE training."""

    model: nn.Module
    history: tuple[EpochMetrics, ...]
    best_validation_loss: float
    checkpoint_path: Path


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    """
    Run one training or validation epoch.
    """

    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0

    for noisy, clean in loader:

        noisy = noisy.to(
            device,
            dtype=torch.float32,
        )

        clean = clean.to(
            device,
            dtype=torch.float32,
        )

        if training:
            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(training):

            reconstructed = model(noisy)

            loss = criterion(
                reconstructed,
                clean,
            )

            if training:
                loss.backward()

                optimizer.step()

        batch_size = noisy.shape[0]

        total_loss += (
            float(loss.detach())
            * batch_size
        )

        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError(
            "DAE DataLoader contained no samples."
        )

    return total_loss / total_samples


def train_dae(
    model: nn.Module,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    *,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    checkpoint_path: str | Path = "checkpoints/dae_best.pt",
    random_seed: int = 42,
    device: str = "cpu",
) -> TrainingResult:
    """
    Train a denoising autoencoder.

    Parameters
    ----------
    model:
        DenoisingAutoencoder instance.

    train_dataset:
        Training noisy/clean pairs.

    validation_dataset:
        Validation noisy/clean pairs.

    epochs:
        Number of training epochs.

    batch_size:
        Mini-batch size.

    learning_rate:
        Adam learning rate.

    weight_decay:
        Adam weight decay.

    checkpoint_path:
        Path where the best model is saved.

    random_seed:
        Deterministic DataLoader seed.

    device:
        "cpu" or another supported PyTorch device.
    """

    if epochs <= 0:
        raise ValueError(
            "epochs must be positive."
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )

    if learning_rate <= 0:
        raise ValueError(
            "learning_rate must be positive."
        )

    if weight_decay < 0:
        raise ValueError(
            "weight_decay cannot be negative."
        )

    if len(train_dataset) == 0:
        raise ValueError(
            "Training dataset is empty."
        )

    if len(validation_dataset) == 0:
        raise ValueError(
            "Validation dataset is empty."
        )

    torch.manual_seed(
        random_seed
    )

    np.random.seed(
        random_seed
    )

    resolved_device = torch.device(
        device
    )

    model = model.to(
        resolved_device
    )

    generator = torch.Generator()

    generator.manual_seed(
        random_seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    checkpoint = Path(
        checkpoint_path
    )

    checkpoint.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_validation_loss = float(
        "inf"
    )

    history: list[EpochMetrics] = []

    for epoch in range(
        1,
        epochs + 1,
    ):

        train_loss = _run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=resolved_device,
        )

        validation_loss = _run_epoch(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            optimizer=None,
            device=resolved_device,
        )

        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            validation_loss=validation_loss,
        )

        history.append(
            metrics
        )

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"validation_loss={validation_loss:.6f}"
        )

        if validation_loss < best_validation_loss:

            best_validation_loss = (
                validation_loss
            )

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),
                    "epoch":
                        epoch,
                    "train_loss":
                        train_loss,
                    "validation_loss":
                        validation_loss,
                    "random_seed":
                        random_seed,
                },
                checkpoint,
            )

    if not checkpoint.exists():
        raise RuntimeError(
            "DAE checkpoint was not created."
        )

    # Reload the best checkpoint.
    checkpoint_data = torch.load(
        checkpoint,
        map_location=resolved_device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint_data[
            "model_state_dict"
        ]
    )

    return TrainingResult(
        model=model,
        history=tuple(history),
        best_validation_loss=float(
            best_validation_loss
        ),
        checkpoint_path=checkpoint,
    )


__all__ = [
    "EpochMetrics",
    "TrainingResult",
    "train_dae",
]
