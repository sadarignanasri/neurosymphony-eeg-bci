"""
Neuro-Symphony Phase 2 — Denoising Autoencoder.

Generic configurable 1D convolutional DAE for physiological signals.

Input shape:
    (batch, channels, samples)

Output shape:
    (batch, channels, samples)

The same architecture can therefore be used for:
    - EEG
    - fNIRS

The modality-specific channel count is supplied through configuration.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class DAEConfig:
    """Configuration for the denoising autoencoder."""

    input_channels: int

    hidden_channels: int = 32

    latent_channels: int = 16

    kernel_size: int = 5

    dropout: float = 0.0


class DenoisingAutoencoder(nn.Module):
    """
    Convolutional denoising autoencoder.

    The network preserves the temporal length of the input.
    """

    def __init__(self, config: DAEConfig) -> None:
        super().__init__()

        if config.input_channels <= 0:
            raise ValueError(
                "input_channels must be positive."
            )

        if config.hidden_channels <= 0:
            raise ValueError(
                "hidden_channels must be positive."
            )

        if config.latent_channels <= 0:
            raise ValueError(
                "latent_channels must be positive."
            )

        if config.kernel_size <= 0:
            raise ValueError(
                "kernel_size must be positive."
            )

        if config.kernel_size % 2 == 0:
            raise ValueError(
                "kernel_size must be odd so temporal length is preserved."
            )

        if not 0.0 <= config.dropout < 1.0:
            raise ValueError(
                "dropout must be in the range [0, 1)."
            )

        padding = config.kernel_size // 2

        self.config = config

        self.encoder = nn.Sequential(
            nn.Conv1d(
                config.input_channels,
                config.hidden_channels,
                kernel_size=config.kernel_size,
                padding=padding,
            ),
            nn.ReLU(),

            nn.Dropout(config.dropout),

            nn.Conv1d(
                config.hidden_channels,
                config.latent_channels,
                kernel_size=config.kernel_size,
                padding=padding,
            ),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Conv1d(
                config.latent_channels,
                config.hidden_channels,
                kernel_size=config.kernel_size,
                padding=padding,
            ),
            nn.ReLU(),

            nn.Dropout(config.dropout),

            nn.Conv1d(
                config.hidden_channels,
                config.input_channels,
                kernel_size=config.kernel_size,
                padding=padding,
            ),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the noisy signal into a latent representation."""
        return self.encoder(x)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode the latent representation."""
        return self.decoder(latent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct the clean signal from the noisy input."""

        latent = self.encode(x)

        reconstructed = self.decode(latent)

        return reconstructed


__all__ = [
    "DAEConfig",
    "DenoisingAutoencoder",
]
