"""
Neuro-Symphony Phase 3 — Universal Multimodal Representation.

This module combines learned temporal representations from
the EEG and fNIRS TCN encoders.

Input:
    EEG temporal representation:
        (batch, eeg_embedding_dimension)

    fNIRS temporal representation:
        (batch, fnirs_embedding_dimension)

Output:
    Multimodal temporal representation:
        (batch, eeg_dimension + fnirs_dimension)

The module is dataset-independent.

No dataset-specific task names, channel names, sampling
rates, or subject identifiers are used here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


# ============================================================
# CONFIGURATION
# ============================================================


@dataclass(frozen=True)
class MultimodalRepresentationConfig:
    """
    Configuration for universal multimodal representation.

    Parameters
    ----------
    eeg_dimension:
        Dimension of the learned EEG temporal representation.

    fnirs_dimension:
        Dimension of the learned fNIRS temporal representation.

    output_dimension:
        Optional projected multimodal dimension.

        If None, the raw concatenated representation is returned:

            eeg_dimension + fnirs_dimension
    """

    eeg_dimension: int
    fnirs_dimension: int

    output_dimension: int | None = None

    dropout: float = 0.0


# ============================================================
# VALIDATION
# ============================================================


def _validate_config(
    config: MultimodalRepresentationConfig,
) -> None:

    if config.eeg_dimension <= 0:
        raise ValueError(
            "eeg_dimension must be positive."
        )

    if config.fnirs_dimension <= 0:
        raise ValueError(
            "fnirs_dimension must be positive."
        )

    if config.output_dimension is not None:
        if config.output_dimension <= 0:
            raise ValueError(
                "output_dimension must be positive."
            )

    if not 0.0 <= config.dropout < 1.0:
        raise ValueError(
            "dropout must be in the range [0, 1)."
        )


# ============================================================
# MODEL
# ============================================================


class MultimodalRepresentation(nn.Module):
    """
    Universal EEG-fNIRS multimodal representation module.

    The default representation is simple concatenation:

        EEG representation
                +
        fNIRS representation
                ↓
        multimodal representation

    An optional projection layer can be enabled when a fixed
    lower-dimensional multimodal embedding is required.
    """

    def __init__(
        self,
        config: MultimodalRepresentationConfig,
    ) -> None:

        super().__init__()

        _validate_config(config)

        self.config = config

        input_dimension = (
            config.eeg_dimension
            + config.fnirs_dimension
        )

        self.input_dimension = input_dimension

        # ----------------------------------------------------
        # Optional projection
        # ----------------------------------------------------

        if config.output_dimension is None:

            self.projection = nn.Identity()

            self.output_dimension = (
                input_dimension
            )

        else:

            layers: list[nn.Module] = [
                nn.Linear(
                    input_dimension,
                    config.output_dimension,
                ),
                nn.ReLU(),
            ]

            if config.dropout > 0.0:

                layers.append(
                    nn.Dropout(
                        config.dropout
                    )
                )

            self.projection = nn.Sequential(
                *layers
            )

            self.output_dimension = (
                config.output_dimension
            )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        eeg_representation: torch.Tensor,
        fnirs_representation: torch.Tensor,
    ) -> torch.Tensor:
        """
        Combine EEG and fNIRS temporal representations.

        Parameters
        ----------
        eeg_representation:
            Tensor of shape:

                (batch, eeg_dimension)

        fnirs_representation:
            Tensor of shape:

                (batch, fnirs_dimension)

        Returns
        -------
        torch.Tensor
            Multimodal representation with shape:

                (batch, output_dimension)

        """

        # ----------------------------------------------------
        # Validate tensor dimensions
        # ----------------------------------------------------

        if not isinstance(
            eeg_representation,
            torch.Tensor,
        ):
            raise TypeError(
                "eeg_representation must be a torch.Tensor."
            )

        if not isinstance(
            fnirs_representation,
            torch.Tensor,
        ):
            raise TypeError(
                "fnirs_representation must be a torch.Tensor."
            )

        if eeg_representation.ndim != 2:
            raise ValueError(
                "EEG representation must have shape "
                "(batch, features)."
            )

        if fnirs_representation.ndim != 2:
            raise ValueError(
                "fNIRS representation must have shape "
                "(batch, features)."
            )

        # ----------------------------------------------------
        # Validate batch alignment
        # ----------------------------------------------------

        if (
            eeg_representation.shape[0]
            != fnirs_representation.shape[0]
        ):

            raise ValueError(
                "EEG and fNIRS representations must "
                "contain the same number of samples."
            )

        # ----------------------------------------------------
        # Validate feature dimensions
        # ----------------------------------------------------

        if (
            eeg_representation.shape[1]
            != self.config.eeg_dimension
        ):

            raise ValueError(
                "EEG representation dimension does not "
                "match the configured eeg_dimension."
            )

        if (
            fnirs_representation.shape[1]
            != self.config.fnirs_dimension
        ):

            raise ValueError(
                "fNIRS representation dimension does not "
                "match the configured fnirs_dimension."
            )

        # ----------------------------------------------------
        # Validate device
        # ----------------------------------------------------

        if (
            eeg_representation.device
            != fnirs_representation.device
        ):

            raise ValueError(
                "EEG and fNIRS representations must "
                "be on the same device."
            )

        # ----------------------------------------------------
        # Validate numerical integrity
        # ----------------------------------------------------

        if not torch.isfinite(
            eeg_representation
        ).all():

            raise ValueError(
                "EEG representation contains "
                "NaN or infinite values."
            )

        if not torch.isfinite(
            fnirs_representation
        ).all():

            raise ValueError(
                "fNIRS representation contains "
                "NaN or infinite values."
            )

        # ----------------------------------------------------
        # Concatenate modalities
        # ----------------------------------------------------

        combined = torch.cat(
            (
                eeg_representation,
                fnirs_representation,
            ),
            dim=1,
        )

        # ----------------------------------------------------
        # Optional learned projection
        # ----------------------------------------------------

        multimodal = self.projection(
            combined
        )

        # ----------------------------------------------------
        # Final integrity check
        # ----------------------------------------------------

        if not torch.isfinite(
            multimodal
        ).all():

            raise ValueError(
                "Multimodal representation contains "
                "NaN or infinite values."
            )

        return multimodal


# ============================================================
# FUNCTIONAL API
# ============================================================


def create_multimodal_representation(
    eeg_representation: torch.Tensor,
    fnirs_representation: torch.Tensor,
    config: MultimodalRepresentationConfig | None = None,
) -> torch.Tensor:
    """
    Create a multimodal EEG-fNIRS representation.

    This functional API is useful when a reusable model object
    is not required.
    """

    if config is None:

        if eeg_representation.ndim != 2:
            raise ValueError(
                "EEG representation must have shape "
                "(batch, features)."
            )

        if fnirs_representation.ndim != 2:
            raise ValueError(
                "fNIRS representation must have shape "
                "(batch, features)."
            )

        config = MultimodalRepresentationConfig(
            eeg_dimension=eeg_representation.shape[1],
            fnirs_dimension=fnirs_representation.shape[1],
        )

    model = MultimodalRepresentation(
        config
    )

    model = model.to(
        eeg_representation.device
    )

    model.eval()

    with torch.no_grad():

        return model(
            eeg_representation,
            fnirs_representation,
        )


# ============================================================
# FLAT REPRESENTATION
# ============================================================


def multimodal_feature_vector(
    representation: torch.Tensor,
) -> torch.Tensor:
    """
    Validate and return a multimodal representation.

    Expected shape:

        (batch, features)

    This helper is intended for downstream prediction,
    classification, and Phase 4 semantic alignment.
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
            "Multimodal representation must have shape "
            "(batch, features)."
        )

    if not torch.isfinite(
        representation
    ).all():

        raise ValueError(
            "Multimodal representation contains "
            "NaN or infinite values."
        )

    return representation


__all__ = [
    "MultimodalRepresentationConfig",
    "MultimodalRepresentation",
    "create_multimodal_representation",
    "multimodal_feature_vector",
]
