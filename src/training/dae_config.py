"""
Neuro-Symphony Phase 2 — DAE training configuration.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DAETrainingConfig:
    """Configuration shared by EEG and fNIRS DAE training."""

    window_size: int = 512

    batch_size: int = 16

    epochs: int = 10

    learning_rate: float = 1e-3

    weight_decay: float = 1e-5

    noise_std: float = 0.05

    validation_fraction: float = 0.15

    test_fraction: float = 0.15

    random_seed: int = 42

    checkpoint_dir: str = "checkpoints"


__all__ = [
    "DAETrainingConfig",
]
