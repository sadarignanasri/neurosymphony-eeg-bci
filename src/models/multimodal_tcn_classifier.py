"""
Neuro-Symphony Phase 3 — Trainable Multimodal TCN Classifier.

This model implements the complete temporal representation pipeline:

    EEG
      ↓
    EEG TCN
      ↓
    EEG temporal representation

    fNIRS
      ↓
    fNIRS TCN
      ↓
    fNIRS temporal representation

              ↓

    Multimodal representation
              ↓
    Classification head
              ↓
    Task prediction

The EEG TCN, fNIRS TCN, multimodal representation, and
classifier are trained jointly using classification loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from src.models.tcn import (
    TCNConfig,
    TemporalConvolutionalNetwork,
)


# ============================================================
# CONFIGURATION
# ============================================================


@dataclass(frozen=True)
class MultimodalTCNClassifierConfig:
    """Configuration for the trainable multimodal TCN model."""

    eeg_channels: int
    fnirs_channels: int

    num_classes: int

    embedding_dimension: int = 128

    eeg_hidden_channels: int = 32
    fnirs_hidden_channels: int = 32

    eeg_num_blocks: int = 4
    fnirs_num_blocks: int = 3

    kernel_size: int = 3

    dropout: float = 0.1

    classifier_hidden_dimension: int = 128


# ============================================================
# MODEL
# ============================================================


class MultimodalTCNClassifier(nn.Module):
    """
    End-to-end trainable EEG-fNIRS temporal classifier.

    Input
    -----
    EEG:
        (batch, eeg_channels, samples)

    fNIRS:
        (batch, fnirs_channels, samples)

    Output
    ------
    logits:
        (batch, num_classes)
    """

    def __init__(
        self,
        config: MultimodalTCNClassifierConfig,
    ) -> None:

        super().__init__()

        # ----------------------------------------------------
        # Validate configuration
        # ----------------------------------------------------

        if config.eeg_channels <= 0:
            raise ValueError(
                "eeg_channels must be positive."
            )

        if config.fnirs_channels <= 0:
            raise ValueError(
                "fnirs_channels must be positive."
            )

        if config.num_classes <= 1:
            raise ValueError(
                "num_classes must be greater than 1."
            )

        if config.embedding_dimension <= 0:
            raise ValueError(
                "embedding_dimension must be positive."
            )

        if config.classifier_hidden_dimension <= 0:
            raise ValueError(
                "classifier_hidden_dimension must be positive."
            )

        if not 0.0 <= config.dropout < 1.0:
            raise ValueError(
                "dropout must be in the range [0, 1)."
            )

        self.config = config

        # ----------------------------------------------------
        # EEG temporal encoder
        # ----------------------------------------------------

        self.eeg_tcn = (
            TemporalConvolutionalNetwork(
                TCNConfig(
                    input_channels=(
                        config.eeg_channels
                    ),
                    hidden_channels=(
                        config.eeg_hidden_channels
                    ),
                    embedding_dimension=(
                        config.embedding_dimension
                    ),
                    kernel_size=(
                        config.kernel_size
                    ),
                    num_blocks=(
                        config.eeg_num_blocks
                    ),
                    dropout=config.dropout,
                    use_batch_norm=True,
                )
            )
        )

        # ----------------------------------------------------
        # fNIRS temporal encoder
        # ----------------------------------------------------

        self.fnirs_tcn = (
            TemporalConvolutionalNetwork(
                TCNConfig(
                    input_channels=(
                        config.fnirs_channels
                    ),
                    hidden_channels=(
                        config.fnirs_hidden_channels
                    ),
                    embedding_dimension=(
                        config.embedding_dimension
                    ),
                    kernel_size=(
                        config.kernel_size
                    ),
                    num_blocks=(
                        config.fnirs_num_blocks
                    ),
                    dropout=config.dropout,
                    use_batch_norm=True,
                )
            )
        )

        # ----------------------------------------------------
        # Multimodal representation
        #
        # EEG embedding:
        #     128
        #
        # fNIRS embedding:
        #     128
        #
        # Combined:
        #     256
        # ----------------------------------------------------

        self.multimodal_dimension = (
            2 * config.embedding_dimension
        )

        # ----------------------------------------------------
        # Classification head
        # ----------------------------------------------------

        self.classifier = nn.Sequential(
            nn.Linear(
                self.multimodal_dimension,
                config.classifier_hidden_dimension,
            ),

            nn.ReLU(),

            nn.Dropout(
                config.dropout
            ),

            nn.Linear(
                config.classifier_hidden_dimension,
                config.num_classes,
            ),
        )

    # ========================================================
    # TEMPORAL REPRESENTATIONS
    # ========================================================

    def encode_eeg(
        self,
        eeg: Tensor,
    ) -> Tensor:
        """
        Extract the learned EEG temporal representation.

        Returns:
            (batch, embedding_dimension)
        """

        representation = self.eeg_tcn(
            eeg
        )

        if representation.ndim != 2:
            raise ValueError(
                "EEG TCN representation must have "
                "shape (batch, embedding_dimension)."
            )

        return representation

    def encode_fnirs(
        self,
        fnirs: Tensor,
    ) -> Tensor:
        """
        Extract the learned fNIRS temporal representation.

        Returns:
            (batch, embedding_dimension)
        """

        representation = self.fnirs_tcn(
            fnirs
        )

        if representation.ndim != 2:
            raise ValueError(
                "fNIRS TCN representation must have "
                "shape (batch, embedding_dimension)."
            )

        return representation

    # ========================================================
    # MULTIMODAL REPRESENTATION
    # ========================================================

    def create_multimodal_representation(
        self,
        eeg_representation: Tensor,
        fnirs_representation: Tensor,
    ) -> Tensor:
        """
        Combine EEG and fNIRS temporal representations.

        EEG:
            (batch, 128)

        fNIRS:
            (batch, 128)

        Output:
            (batch, 256)
        """

        if eeg_representation.ndim != 2:
            raise ValueError(
                "EEG representation must be 2-dimensional."
            )

        if fnirs_representation.ndim != 2:
            raise ValueError(
                "fNIRS representation must be 2-dimensional."
            )

        if (
            eeg_representation.shape[0]
            != fnirs_representation.shape[0]
        ):
            raise ValueError(
                "EEG and fNIRS batch sizes must match."
            )

        representation = torch.cat(
            [
                eeg_representation,
                fnirs_representation,
            ],
            dim=1,
        )

        if representation.shape[1] != (
            self.multimodal_dimension
        ):
            raise ValueError(
                "Unexpected multimodal representation "
                "dimension."
            )

        return representation

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        eeg: Tensor,
        fnirs: Tensor,
        return_representations: bool = False,
    ):
        """
        Perform the complete forward pass.

        Parameters
        ----------
        eeg:
            EEG tensor:

                (batch, eeg_channels, samples)

        fnirs:
            fNIRS tensor:

                (batch, fnirs_channels, samples)

        return_representations:
            If True, return EEG representation,
            fNIRS representation, multimodal representation,
            and logits.

        Returns
        -------
        logits
            (batch, num_classes)

        or

        dictionary containing learned representations
        and logits.
        """

        if not isinstance(
            eeg,
            torch.Tensor,
        ):
            raise TypeError(
                "eeg must be a torch.Tensor."
            )

        if not isinstance(
            fnirs,
            torch.Tensor,
        ):
            raise TypeError(
                "fnirs must be a torch.Tensor."
            )

        if eeg.ndim != 3:
            raise ValueError(
                "EEG input must have shape "
                "(batch, channels, samples)."
            )

        if fnirs.ndim != 3:
            raise ValueError(
                "fNIRS input must have shape "
                "(batch, channels, samples)."
            )

        if eeg.shape[0] != fnirs.shape[0]:
            raise ValueError(
                "EEG and fNIRS batch sizes must match."
            )

        if eeg.shape[1] != (
            self.config.eeg_channels
        ):
            raise ValueError(
                "EEG channel count does not match "
                "the model configuration."
            )

        if fnirs.shape[1] != (
            self.config.fnirs_channels
        ):
            raise ValueError(
                "fNIRS channel count does not match "
                "the model configuration."
            )

        if not torch.isfinite(eeg).all():
            raise ValueError(
                "EEG input contains NaN or infinite values."
            )

        if not torch.isfinite(fnirs).all():
            raise ValueError(
                "fNIRS input contains NaN or infinite values."
            )

        # ----------------------------------------------------
        # EEG temporal representation
        # ----------------------------------------------------

        eeg_representation = self.encode_eeg(
            eeg
        )

        # ----------------------------------------------------
        # fNIRS temporal representation
        # ----------------------------------------------------

        fnirs_representation = self.encode_fnirs(
            fnirs
        )

        # ----------------------------------------------------
        # Multimodal representation
        # ----------------------------------------------------

        multimodal_representation = (
            self.create_multimodal_representation(
                eeg_representation,
                fnirs_representation,
            )
        )

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        logits = self.classifier(
            multimodal_representation
        )

        if not torch.isfinite(logits).all():
            raise ValueError(
                "Classifier logits contain "
                "NaN or infinite values."
            )

        if return_representations:

            return {
                "eeg_representation": (
                    eeg_representation
                ),

                "fnirs_representation": (
                    fnirs_representation
                ),

                "multimodal_representation": (
                    multimodal_representation
                ),

                "logits": logits,
            }

        return logits


# ============================================================
# LOSS
# ============================================================


def multimodal_classification_loss(
    logits: Tensor,
    targets: Tensor,
) -> Tensor:
    """
    Calculate classification loss for the complete
    multimodal TCN model.
    """

    if logits.ndim != 2:
        raise ValueError(
            "logits must have shape "
            "(batch, num_classes)."
        )

    if targets.ndim != 1:
        raise ValueError(
            "targets must have shape (batch,)."
        )

    if logits.shape[0] != targets.shape[0]:
        raise ValueError(
            "Logits and targets must contain "
            "the same number of samples."
        )

    loss = nn.functional.cross_entropy(
        logits,
        targets,
    )

    if not torch.isfinite(loss):
        raise ValueError(
            "Classification loss is not finite."
        )

    return loss


# ============================================================
# PREDICTION
# ============================================================


@torch.no_grad()
def predict(
    model: MultimodalTCNClassifier,
    eeg: Tensor,
    fnirs: Tensor,
) -> Tensor:
    """
    Predict class indices from EEG and fNIRS signals.
    """

    was_training = model.training

    model.eval()

    try:

        logits = model(
            eeg,
            fnirs,
        )

        predictions = torch.argmax(
            logits,
            dim=1,
        )

    finally:

        if was_training:
            model.train()

    return predictions


# ============================================================
# PUBLIC API
# ============================================================


__all__ = [
    "MultimodalTCNClassifierConfig",
    "MultimodalTCNClassifier",
    "multimodal_classification_loss",
    "predict",
]
