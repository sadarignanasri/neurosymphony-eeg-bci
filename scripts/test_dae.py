"""
Neuro-Symphony Phase 2 — DAE model validation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.models.dae import (
    DAEConfig,
    DenoisingAutoencoder,
)


print("=" * 75)
print("PHASE 2 — DAE MODEL VALIDATION")
print("=" * 75)


# ============================================================
# 1. EEG DAE
# ============================================================

print()
print("[1] Creating EEG DAE...")

eeg_config = DAEConfig(
    input_channels=8,
)

eeg_model = DenoisingAutoencoder(
    eeg_config
)

print("PASS: EEG DAE created.")


# ============================================================
# 2. EEG FORWARD PASS
# ============================================================

print()
print("[2] Testing EEG forward pass...")

eeg_input = torch.randn(
    4,
    8,
    512,
)

eeg_output = eeg_model(
    eeg_input
)

print(
    "Input shape :",
    tuple(eeg_input.shape)
)

print(
    "Output shape:",
    tuple(eeg_output.shape)
)

assert eeg_output.shape == eeg_input.shape

print("PASS: EEG output shape preserved.")


# ============================================================
# 3. fNIRS DAE
# ============================================================

print()
print("[3] Creating fNIRS DAE...")

fnirs_config = DAEConfig(
    input_channels=8,
)

fnirs_model = DenoisingAutoencoder(
    fnirs_config
)

print("PASS: fNIRS DAE created.")


# ============================================================
# 4. fNIRS FORWARD PASS
# ============================================================

print()
print("[4] Testing fNIRS forward pass...")

fnirs_input = torch.randn(
    4,
    8,
    256,
)

fnirs_output = fnirs_model(
    fnirs_input
)

print(
    "Input shape :",
    tuple(fnirs_input.shape)
)

print(
    "Output shape:",
    tuple(fnirs_output.shape)
)

assert fnirs_output.shape == fnirs_input.shape

print("PASS: fNIRS output shape preserved.")


# ============================================================
# 5. RECONSTRUCTION LOSS
# ============================================================

print()
print("[5] Testing reconstruction loss...")

criterion = torch.nn.MSELoss()

loss = criterion(
    eeg_output,
    eeg_input,
)

assert torch.isfinite(loss)

print(
    "Reconstruction loss:",
    float(loss)
)

print("PASS: Reconstruction loss is finite.")


# ============================================================
# 6. BACKPROPAGATION
# ============================================================

print()
print("[6] Testing backpropagation...")

eeg_model.zero_grad()

loss.backward()

gradient_found = any(
    parameter.grad is not None
    for parameter in eeg_model.parameters()
)

assert gradient_found

print("PASS: Gradients were generated.")


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 75)
print("DAE MODEL VALIDATION PASSED")
print("=" * 75)
