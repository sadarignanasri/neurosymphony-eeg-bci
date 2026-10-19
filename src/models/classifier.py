"""
Neuro-Symphony Phase 3 — Universal Multimodal Classifier.

This module receives learned multimodal neural representations
from the EEG-fNIRS TCN pipeline and performs task classification.

Input:
    (batch, representation_dimension)

Output:
    Class logits:
    (batch, num_classes)

The implementation is dataset-independent.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


# ============================================================
# CONFIGURATION
# ============================================================


@dataclass(frozen=True)
class ClassifierConfig:
    """Configuration for the universal Phase 3 classifier."""

    input_dimension: int

    num_classes: int

    hidden_dimension: int = 128

    dropout: float = 0.2


# ============================================================
# MODEL
# ============================================================


class MultimodalClassifier(nn.Module):
    """
    Feed-forward classifier operating on learned multimodal
    temporal representations.

    Input:
        (batch, input_dimension)

    Output:
        (batch, num_classes)
    """

    def __init__(
        self,
        config: ClassifierConfig,
    ) -> None:

        super().__init__()

        if config.input_dimension <= 0:
            raise ValueError(
                "input_dimension must be positive."
            )

        if config.num_classes <= 1:
            raise ValueError(
                "num_classes must be greater than 1."
            )

        if config.hidden_dimension <= 0:
            raise ValueError(
                "hidden_dimension must be positive."
            )

        if not 0.0 <= config.dropout < 1.0:
            raise ValueError(
                "dropout must be in the range [0, 1)."
            )

        self.config = config

        self.network = nn.Sequential(
            nn.Linear(
                config.input_dimension,
                config.hidden_dimension,
            ),

            nn.ReLU(),

            nn.Dropout(
                config.dropout
            ),

            nn.Linear(
                config.hidden_dimension,
                config.num_classes,
            ),
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        representation: Tensor,
    ) -> Tensor:
        """
        Produce class logits from multimodal representation.

        Parameters
        ----------
        representation:
            Tensor with shape:

                (batch, input_dimension)

        Returns
        -------
        Tensor
            Class logits with shape:

                (batch, num_classes)
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
                "representation must have shape "
                "(batch, features)."
            )

        if (
            representation.shape[1]
            != self.config.input_dimension
        ):
            raise ValueError(
                "Representation dimension does not "
                "match classifier configuration."
            )

        if not torch.isfinite(
            representation
        ).all():

            raise ValueError(
                "Representation contains "
                "NaN or infinite values."
            )

        logits = self.network(
            representation
        )

        if not torch.isfinite(
            logits
        ).all():

            raise ValueError(
                "Classifier output contains "
                "NaN or infinite values."
            )

        return logits


# ============================================================
# PREDICTION
# ============================================================


@torch.no_grad()
def predict_classes(
    model: MultimodalClassifier,
    representation: Tensor,
) -> Tensor:
    """
    Predict the class index for each multimodal representation.

    Returns:
        Tensor with shape:

            (batch,)
    """

    if not isinstance(
        model,
        MultimodalClassifier,
    ):
        raise TypeError(
            "model must be a MultimodalClassifier."
        )

    was_training = model.training

    model.eval()

    try:

        logits = model(
            representation
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
# PROBABILITIES
# ============================================================


@torch.no_grad()
def predict_probabilities(
    model: MultimodalClassifier,
    representation: Tensor,
) -> Tensor:
    """
    Return class probabilities.

    Output shape:

        (batch, num_classes)
    """

    if not isinstance(
        model,
        MultimodalClassifier,
    ):
        raise TypeError(
            "model must be a MultimodalClassifier."
        )

    was_training = model.training

    model.eval()

    try:

        logits = model(
            representation
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

    finally:

        if was_training:
            model.train()

    if not torch.isfinite(
        probabilities
    ).all():

        raise ValueError(
            "Prediction probabilities contain "
            "NaN or infinite values."
        )

    return probabilities


# ============================================================
# LOSS
# ============================================================


def classification_loss(
    logits: Tensor,
    targets: Tensor,
) -> Tensor:
    """
    Calculate cross-entropy classification loss.

    Parameters
    ----------
    logits:
        Shape:

            (batch, num_classes)

    targets:
        Shape:

            (batch,)

        Integer class indices.
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
            "Number of logits and targets must match."
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
# PUBLIC API
# ============================================================


__all__ = [
    "ClassifierConfig",
    "MultimodalClassifier",
    "predict_classes",
    "predict_probabilities",
    "classification_loss",
]
