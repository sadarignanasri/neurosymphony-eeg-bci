"""
Neuro-Symphony Phase 3 — Universal Temporal Convolutional Network.

This module provides a reusable Temporal Convolutional Network (TCN)
for learning temporal representations from physiological signals.

Supported modalities:
    - EEG
    - fNIRS

Input shape:
    (batch, channels, samples)

Output:
    Learned temporal representation of shape:
    (batch, embedding_dimension)

The implementation is dataset-independent.
No dataset, task, channel count, or sampling rate is hard-coded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn


# ============================================================
# CONFIGURATION
# ============================================================


@dataclass(frozen=True)
class TCNConfig:
    """Configuration for the universal Temporal Convolutional Network."""

    input_channels: int

    hidden_channels: int = 64

    embedding_dimension: int = 128

    kernel_size: int = 3

    num_blocks: int = 4

    dropout: float = 0.1

    use_batch_norm: bool = True


# ============================================================
# TEMPORAL BLOCK
# ============================================================


class TemporalBlock(nn.Module):
    """
    Residual dilated temporal convolution block.

    Each block contains two dilated Conv1D layers followed by
    normalization, activation, dropout, and a residual connection.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        use_batch_norm: bool,
    ) -> None:
        super().__init__()

        if kernel_size <= 0:
            raise ValueError(
                "kernel_size must be positive."
            )

        if kernel_size % 2 == 0:
            raise ValueError(
                "kernel_size must be odd."
            )

        if dilation <= 0:
            raise ValueError(
                "dilation must be positive."
            )

        padding = (
            (kernel_size - 1) * dilation // 2
        )

        layers: list[nn.Module] = []

        # ----------------------------------------------------
        # First temporal convolution
        # ----------------------------------------------------

        layers.append(
            nn.Conv1d(
                input_channels,
                output_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
            )
        )

        if use_batch_norm:
            layers.append(
                nn.BatchNorm1d(
                    output_channels
                )
            )

        layers.append(
            nn.ReLU()
        )

        layers.append(
            nn.Dropout(dropout)
        )

        # ----------------------------------------------------
        # Second temporal convolution
        # ----------------------------------------------------

        layers.append(
            nn.Conv1d(
                output_channels,
                output_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
            )
        )

        if use_batch_norm:
            layers.append(
                nn.BatchNorm1d(
                    output_channels
                )
            )

        layers.append(
            nn.ReLU()
        )

        layers.append(
            nn.Dropout(dropout)
        )

        self.network = nn.Sequential(
            *layers
        )

        # ----------------------------------------------------
        # Residual projection
        # ----------------------------------------------------

        if input_channels != output_channels:

            self.residual = nn.Conv1d(
                input_channels,
                output_channels,
                kernel_size=1,
            )

        else:

            self.residual = nn.Identity()

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """Apply temporal block with residual connection."""

        residual = self.residual(x)

        output = self.network(x)

        # Defensive alignment in case of numerical/shape
        # differences caused by convolution configuration.
        if output.shape[-1] != residual.shape[-1]:

            minimum_length = min(
                output.shape[-1],
                residual.shape[-1],
            )

            output = output[..., :minimum_length]

            residual = residual[..., :minimum_length]

        return torch.relu(
            output + residual
        )


# ============================================================
# TEMPORAL CONVOLUTIONAL NETWORK
# ============================================================


class TemporalConvolutionalNetwork(nn.Module):
    """
    Universal Temporal Convolutional Network.

    The network learns a compact representation from a
    multichannel physiological time series.

    Input:
        (batch, channels, samples)

    Output:
        (batch, embedding_dimension)
    """

    def __init__(
        self,
        config: TCNConfig,
    ) -> None:

        super().__init__()

        # ----------------------------------------------------
        # Configuration validation
        # ----------------------------------------------------

        if config.input_channels <= 0:
            raise ValueError(
                "input_channels must be positive."
            )

        if config.hidden_channels <= 0:
            raise ValueError(
                "hidden_channels must be positive."
            )

        if config.embedding_dimension <= 0:
            raise ValueError(
                "embedding_dimension must be positive."
            )

        if config.kernel_size <= 0:
            raise ValueError(
                "kernel_size must be positive."
            )

        if config.kernel_size % 2 == 0:
            raise ValueError(
                "kernel_size must be odd."
            )

        if config.num_blocks <= 0:
            raise ValueError(
                "num_blocks must be positive."
            )

        if not 0.0 <= config.dropout < 1.0:
            raise ValueError(
                "dropout must be in the range [0, 1)."
            )

        self.config = config

        # ----------------------------------------------------
        # Temporal blocks
        #
        # Dilation grows exponentially:
        #
        # 1, 2, 4, 8, ...
        #
        # This gives the network a progressively larger
        # temporal receptive field.
        # ----------------------------------------------------

        blocks: list[nn.Module] = []

        input_channels = (
            config.input_channels
        )

        for block_index in range(
            config.num_blocks
        ):

            dilation = 2 ** block_index

            blocks.append(
                TemporalBlock(
                    input_channels=input_channels,
                    output_channels=config.hidden_channels,
                    kernel_size=config.kernel_size,
                    dilation=dilation,
                    dropout=config.dropout,
                    use_batch_norm=config.use_batch_norm,
                )
            )

            input_channels = (
                config.hidden_channels
            )

        self.temporal_network = nn.Sequential(
            *blocks
        )

        # ----------------------------------------------------
        # Global temporal pooling
        # ----------------------------------------------------

        self.global_pool = (
            nn.AdaptiveAvgPool1d(1)
        )

        # ----------------------------------------------------
        # Learned temporal representation
        # ----------------------------------------------------

        self.embedding = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                config.hidden_channels,
                config.embedding_dimension,
            ),

            nn.ReLU(),

            nn.Dropout(
                config.dropout
            ),
        )

    # ========================================================
    # TEMPORAL ENCODER
    # ========================================================

    def encode(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Convert a physiological signal into a temporal embedding.

        Parameters
        ----------
        x:
            Tensor with shape:

                (batch, channels, samples)

        Returns
        -------
        Tensor
            Temporal representation with shape:

                (batch, embedding_dimension)
        """

        if x.ndim != 3:
            raise ValueError(
                "TCN input must have shape "
                "(batch, channels, samples)."
            )

        if x.shape[1] != self.config.input_channels:
            raise ValueError(
                "TCN input channel count does not match "
                f"configuration. Expected "
                f"{self.config.input_channels}, "
                f"received {x.shape[1]}."
            )

        if x.shape[2] <= 0:
            raise ValueError(
                "TCN input must contain samples."
            )

        if not torch.isfinite(x).all():
            raise ValueError(
                "TCN input contains NaN or infinite values."
            )

        temporal_features = (
            self.temporal_network(x)
        )

        pooled = self.global_pool(
            temporal_features
        )

        representation = self.embedding(
            pooled
        )

        if not torch.isfinite(
            representation
        ).all():

            raise ValueError(
                "TCN representation contains "
                "NaN or infinite values."
            )

        return representation

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """Return the learned temporal representation."""

        return self.encode(x)


# ============================================================
# FEATURE EXTRACTION
# ============================================================


@torch.no_grad()
def extract_tcn_features(
    model: TemporalConvolutionalNetwork,
    data: Tensor,
) -> Tensor:
    """
    Extract learned temporal representations.

    The model is temporarily switched to evaluation mode.

    Parameters
    ----------
    model:
        Initialized TCN.

    data:
        Tensor with shape:

            (batch, channels, samples)

    Returns
    -------
    Tensor
        Learned temporal representations.
    """

    if not isinstance(
        model,
        TemporalConvolutionalNetwork,
    ):
        raise TypeError(
            "model must be a "
            "TemporalConvolutionalNetwork."
        )

    was_training = model.training

    model.eval()

    try:

        representation = model.encode(
            data
        )

    finally:

        if was_training:
            model.train()

    return representation


# ============================================================
# NUMPY HELPER
# ============================================================


def tcn_feature_vector(
    representation: Tensor,
) -> Tensor:
    """
    Validate and return a TCN representation.

    This helper ensures the representation is
    two-dimensional and finite.
    """

    if not isinstance(
        representation,
        torch.Tensor,
    ):
        raise TypeError(
            "representation must be a torch.Tensor."
        )

    if representation.ndim != 2:
        raise ValueError(
            "TCN representation must have shape "
            "(batch, embedding_dimension)."
        )

    if not torch.isfinite(
        representation
    ).all():

        raise ValueError(
            "TCN representation contains "
            "NaN or infinite values."
        )

    return representation


# ============================================================
# PUBLIC API
# ============================================================


__all__ = [
    "TCNConfig",
    "TemporalBlock",
    "TemporalConvolutionalNetwork",
    "extract_tcn_features",
    "tcn_feature_vector",
]
