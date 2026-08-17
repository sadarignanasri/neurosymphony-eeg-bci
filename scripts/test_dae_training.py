"""
Validate the DAE training pipeline using synthetic data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.models.dae import (
    DAEConfig,
    DenoisingAutoencoder,
)

from src.training.dae_dataset import (
    DAEDataset,
    create_windows,
    split_indices,
)

from src.training.dae_trainer import (
    train_dae,
)


print("=" * 75)
print("PHASE 2 — DAE TRAINING VALIDATION")
print("=" * 75)


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)


# ------------------------------------------------------------
# 1. Synthetic signal
# ------------------------------------------------------------

print()
print("[1] Creating synthetic signal...")

rng = np.random.default_rng(
    SEED
)

signal = rng.normal(
    0.0,
    1.0,
    size=(8, 16384),
).astype(
    np.float32
)

print(
    "Signal shape:",
    signal.shape
)

print("PASS: Synthetic signal created.")


# ------------------------------------------------------------
# 2. Windowing
# ------------------------------------------------------------

print()
print("[2] Creating windows...")

windows = create_windows(
    signal,
    window_size=512,
)

print(
    "Windows:",
    windows.shape
)

print("PASS: Windows created.")


# ------------------------------------------------------------
# 3. Split
# ------------------------------------------------------------

print()
print("[3] Creating train/validation/test split...")

split = split_indices(
    len(windows),
    validation_fraction=0.20,
    test_fraction=0.20,
    random_seed=SEED,
)

print(
    "Train:",
    len(split.train)
)

print(
    "Validation:",
    len(split.validation)
)

print(
    "Test:",
    len(split.test)
)

print("PASS: Dataset split created.")


# ------------------------------------------------------------
# 4. Dataset objects
# ------------------------------------------------------------

print()
print("[4] Creating DAE datasets...")

train_dataset = DAEDataset(
    windows[split.train],
    noise_std=0.05,
    random_seed=SEED,
)

validation_dataset = DAEDataset(
    windows[split.validation],
    noise_std=0.05,
    random_seed=SEED + 1,
)

test_dataset = DAEDataset(
    windows[split.test],
    noise_std=0.05,
    random_seed=SEED + 2,
)

print(
    "Train samples:",
    len(train_dataset)
)

print(
    "Validation samples:",
    len(validation_dataset)
)

print(
    "Test samples:",
    len(test_dataset)
)

print("PASS: DAE datasets created.")


# ------------------------------------------------------------
# 5. Model
# ------------------------------------------------------------

print()
print("[5] Creating DAE model...")

model = DenoisingAutoencoder(
    DAEConfig(
        input_channels=8,
        hidden_channels=16,
        latent_channels=8,
        kernel_size=5,
    )
)

print("PASS: DAE model created.")


# ------------------------------------------------------------
# 6. Training
# ------------------------------------------------------------

print()
print("[6] Training DAE...")

checkpoint = (
    PROJECT_ROOT
    / "checkpoints"
    / "test_dae.pt"
)

result = train_dae(
    model,
    train_dataset,
    validation_dataset,
    epochs=3,
    batch_size=4,
    learning_rate=1e-3,
    weight_decay=1e-5,
    checkpoint_path=checkpoint,
    random_seed=SEED,
    device="cpu",
)

print()
print(
    "Best validation loss:",
    result.best_validation_loss,
)

print(
    "Checkpoint:",
    result.checkpoint_path,
)

assert checkpoint.exists()

assert np.isfinite(
    result.best_validation_loss
)

print(
    "PASS: DAE training completed."
)

print(
    "PASS: Checkpoint was created."
)


# ------------------------------------------------------------
# 7. Test reconstruction
# ------------------------------------------------------------

print()
print("[7] Testing trained model...")

model.eval()

noisy, clean = test_dataset[0]

with torch.no_grad():

    prediction = model(
        noisy.unsqueeze(0)
    )

test_loss = torch.nn.functional.mse_loss(
    prediction.squeeze(0),
    clean,
)

print(
    "Test reconstruction loss:",
    float(test_loss)
)

assert torch.isfinite(
    test_loss
)

print(
    "PASS: Trained DAE produces finite reconstruction."
)


# ------------------------------------------------------------
# Final
# ------------------------------------------------------------

print()
print("=" * 75)
print("DAE TRAINING VALIDATION PASSED")
print("=" * 75)
