"""
Validate the Phase 2 DAE dataset pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.training.dae_config import DAETrainingConfig
from src.training.dae_dataset import (
    DAEDataset,
    create_windows,
    split_indices,
)


print("=" * 75)
print("PHASE 2 — DAE DATASET PIPELINE VALIDATION")
print("=" * 75)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

config = DAETrainingConfig()

print()
print("[1] Configuration")

print(
    "Window size:",
    config.window_size,
)

print(
    "Validation fraction:",
    config.validation_fraction,
)

print(
    "Test fraction:",
    config.test_fraction,
)

print(
    "Noise standard deviation:",
    config.noise_std,
)

print("PASS: Configuration loaded.")


# ------------------------------------------------------------
# Synthetic signal
# ------------------------------------------------------------

print()
print("[2] Creating synthetic Phase 2 signal...")

rng = np.random.default_rng(
    config.random_seed
)

signal = rng.normal(
    0.0,
    1.0,
    size=(8, 4096),
).astype(
    np.float32
)

original_signal = signal.copy()

print(
    "Signal shape:",
    signal.shape
)

print("PASS: Synthetic signal created.")


# ------------------------------------------------------------
# Windowing
# ------------------------------------------------------------

print()
print("[3] Creating temporal windows...")

windows = create_windows(
    signal,
    config.window_size,
)

print(
    "Window shape:",
    windows.shape
)

assert windows.ndim == 3

assert windows.shape[1] == 8

assert windows.shape[2] == config.window_size

print(
    "Number of windows:",
    windows.shape[0]
)

print("PASS: Temporal windowing works.")


# ------------------------------------------------------------
# Train / validation / test split
# ------------------------------------------------------------

print()
print("[4] Creating train/validation/test split...")

split = split_indices(
    number_of_windows=windows.shape[0],
    validation_fraction=config.validation_fraction,
    test_fraction=config.test_fraction,
    random_seed=config.random_seed,
)

print(
    "Training windows:",
    len(split.train)
)

print(
    "Validation windows:",
    len(split.validation)
)

print(
    "Test windows:",
    len(split.test)
)

all_indices = np.concatenate(
    [
        split.train,
        split.validation,
        split.test,
    ]
)

assert len(
    np.unique(all_indices)
) == windows.shape[0]

assert len(all_indices) == windows.shape[0]

print(
    "PASS: Train/validation/test windows are disjoint."
)


# ------------------------------------------------------------
# Dataset creation
# ------------------------------------------------------------

print()
print("[5] Creating DAE dataset...")

train_windows = windows[
    split.train
]

dataset = DAEDataset(
    train_windows,
    noise_std=config.noise_std,
    random_seed=config.random_seed,
)

print(
    "Dataset size:",
    len(dataset)
)

assert len(dataset) == len(
    split.train
)

noisy, clean = dataset[0]

print(
    "Noisy sample shape:",
    tuple(noisy.shape)
)

print(
    "Clean sample shape:",
    tuple(clean.shape)
)

assert noisy.shape == clean.shape

print(
    "PASS: Noisy/clean pairs created."
)


# ------------------------------------------------------------
# Confirm noise exists
# ------------------------------------------------------------

print()
print("[6] Checking corruption...")

noise_difference = (
    noisy.numpy()
    - clean.numpy()
)

noise_std_measured = float(
    noise_difference.std()
)

print(
    "Measured noise standard deviation:",
    noise_std_measured
)

assert noise_std_measured > 0

print("PASS: Synthetic corruption was applied.")


# ------------------------------------------------------------
# Confirm original signal unchanged
# ------------------------------------------------------------

print()
print("[7] Checking immutability...")

assert np.array_equal(
    signal,
    original_signal,
)

print(
    "PASS: Original signal remains unchanged."
)


# ------------------------------------------------------------
# Final
# ------------------------------------------------------------

print()
print("=" * 75)
print("DAE DATASET PIPELINE VALIDATION PASSED")
print("=" * 75)
