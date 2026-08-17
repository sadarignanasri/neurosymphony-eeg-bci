"""
Neuro-Symphony Phase 2 — Final DAE Integration Validation.

Validates:
1. EEG DAE checkpoint loading
2. fNIRS DAE checkpoint loading
3. EEG input/output shape preservation
4. fNIRS input/output shape preservation
5. Finite reconstruction outputs
6. Reconstruction metrics
"""

from pathlib import Path

import numpy as np
import torch

from src.models.dae import (
    DAEConfig,
    DenoisingAutoencoder,
)
from src.evaluation.dae_metrics import evaluate_reconstruction


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


def load_checkpoint(path: Path):
    """Load a training checkpoint."""
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:
        raise RuntimeError(
            f"Invalid DAE checkpoint: {path}"
        )

    return checkpoint


def validate_model(
    name: str,
    checkpoint_path: Path,
    channels: int,
    window_size: int,
) -> None:

    print()
    print("=" * 75)
    print(f"{name} DAE INTEGRATION")
    print("=" * 75)

    # --------------------------------------------------------
    # CHECKPOINT
    # --------------------------------------------------------

    print("\n[1] Checking checkpoint...")

    assert checkpoint_path.exists(), (
        f"Checkpoint does not exist: {checkpoint_path}"
    )

    checkpoint = load_checkpoint(
        checkpoint_path
    )

    print(
        "Checkpoint:",
        checkpoint_path,
    )

    print(
        "Saved epoch:",
        checkpoint.get("epoch"),
    )

    print(
        "Saved validation loss:",
        checkpoint.get("validation_loss"),
    )

    print("PASS: Checkpoint is valid.")

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print("\n[2] Creating model...")

    # The architecture must exactly match the architecture
    # used when this checkpoint was trained.
    #
    # The existing EEG validation checkpoint was trained with:
    #   hidden_channels = 16
    #   latent_channels = 8
    #
    # The real fNIRS checkpoint was trained with the same
    # architecture.
    # Match the exact architecture used to train each checkpoint.
    #
    # EEG checkpoint:
    #   hidden_channels = 16
    #   latent_channels = 8
    #
    # Real fNIRS checkpoint:
    #   hidden_channels = 32
    #   latent_channels = 16
    if name == "EEG":
        hidden_channels = 16
        latent_channels = 8
    elif name == "fNIRS":
        hidden_channels = 32
        latent_channels = 16
    else:
        raise ValueError(
            f"Unknown DAE modality: {name}"
        )

    model = DenoisingAutoencoder(
        DAEConfig(
            input_channels=channels,
            hidden_channels=hidden_channels,
            latent_channels=latent_channels,
        )
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        "Channels:",
        channels,
    )

    print(
        "Parameters:",
        sum(
            parameter.numel()
            for parameter in model.parameters()
        ),
    )

    print("PASS: Model loaded.")

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    print("\n[3] Creating test signal...")

    rng = np.random.default_rng(42)

    clean = rng.normal(
        0.0,
        1.0,
        size=(
            4,
            channels,
            window_size,
        ),
    ).astype(
        np.float32
    )

    noise = rng.normal(
        0.0,
        0.05,
        size=clean.shape,
    ).astype(
        np.float32
    )

    noisy = clean + noise

    noisy_tensor = torch.from_numpy(
        noisy
    )

    print(
        "Input shape:",
        tuple(noisy_tensor.shape),
    )

    # --------------------------------------------------------
    # FORWARD PASS
    # --------------------------------------------------------

    print("\n[4] Running DAE...")

    with torch.no_grad():
        reconstructed = model(
            noisy_tensor
        )

    print(
        "Output shape:",
        tuple(reconstructed.shape),
    )

    assert reconstructed.shape == noisy_tensor.shape, (
        "DAE changed the input shape."
    )

    assert torch.isfinite(
        reconstructed
    ).all(), (
        "DAE produced NaN or Inf values."
    )

    print(
        "PASS: Output shape preserved."
    )

    print(
        "PASS: Reconstruction is finite."
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    print("\n[5] Computing reconstruction metrics...")

    metrics = evaluate_reconstruction(
        clean,
        reconstructed.numpy(),
    )

    for key, value in metrics.items():
        print(
            f"{key}: {value}"
        )

    print(
        "PASS: Reconstruction metrics computed."
    )

    print()
    print(
        f"PASS: {name} DAE integration validated."
    )


def main() -> None:

    print("=" * 75)
    print("PHASE 2 — FINAL DAE INTEGRATION VALIDATION")
    print("=" * 75)

    # --------------------------------------------------------
    # EEG
    #
    # test_dae.pt was created by the earlier synthetic EEG
    # DAE training validation using 8 EEG channels and
    # 512-sample windows.
    # --------------------------------------------------------

    validate_model(
        name="EEG",
        checkpoint_path=(
            CHECKPOINT_DIR / "test_dae.pt"
        ),
        channels=8,
        window_size=512,
    )

    # --------------------------------------------------------
    # fNIRS
    #
    # Real fNIRS DAE was trained using 28 channels and
    # 256-sample windows.
    # --------------------------------------------------------

    validate_model(
        name="fNIRS",
        checkpoint_path=(
            CHECKPOINT_DIR / "fnirs_dae_real.pt"
        ),
        channels=28,
        window_size=256,
    )

    print()
    print("=" * 75)
    print("PHASE 2 — FINAL DAE INTEGRATION VALIDATION PASSED")
    print("=" * 75)

    print()
    print("EEG DAE:")
    print("  checkpoint       : PASS")
    print("  model loading    : PASS")
    print("  forward pass     : PASS")
    print("  shape preserved  : PASS")
    print("  finite output    : PASS")

    print()
    print("fNIRS DAE:")
    print("  checkpoint       : PASS")
    print("  model loading    : PASS")
    print("  forward pass     : PASS")
    print("  shape preserved  : PASS")
    print("  finite output    : PASS")


if __name__ == "__main__":
    main()
