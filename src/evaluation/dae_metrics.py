"""
Neuro-Symphony Phase 2 — DAE Evaluation Metrics.

Metrics used to compare a baseline signal against a DAE reconstruction.
"""

from __future__ import annotations

import numpy as np


def _validate_pair(
    reference: np.ndarray,
    estimate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    reference = np.asarray(
        reference,
        dtype=np.float64,
    )

    estimate = np.asarray(
        estimate,
        dtype=np.float64,
    )

    if reference.shape != estimate.shape:
        raise ValueError(
            "Reference and estimate must have identical shapes."
        )

    if reference.size == 0:
        raise ValueError(
            "Evaluation arrays cannot be empty."
        )

    if not np.isfinite(reference).all():
        raise ValueError(
            "Reference contains non-finite values."
        )

    if not np.isfinite(estimate).all():
        raise ValueError(
            "Estimate contains non-finite values."
        )

    return reference, estimate


def mse(
    reference: np.ndarray,
    estimate: np.ndarray,
) -> float:

    reference, estimate = _validate_pair(
        reference,
        estimate,
    )

    return float(
        np.mean(
            (reference - estimate) ** 2
        )
    )


def rmse(
    reference: np.ndarray,
    estimate: np.ndarray,
) -> float:

    return float(
        np.sqrt(
            mse(
                reference,
                estimate,
            )
        )
    )


def snr_db(
    reference: np.ndarray,
    estimate: np.ndarray,
) -> float:

    reference, estimate = _validate_pair(
        reference,
        estimate,
    )

    error = (
        reference - estimate
    )

    signal_power = float(
        np.mean(reference ** 2)
    )

    noise_power = float(
        np.mean(error ** 2)
    )

    if noise_power == 0.0:
        return float("inf")

    if signal_power == 0.0:
        return float("-inf")

    return float(
        10.0
        * np.log10(
            signal_power
            / noise_power
        )
    )


def correlation(
    reference: np.ndarray,
    estimate: np.ndarray,
) -> float:

    reference, estimate = _validate_pair(
        reference,
        estimate,
    )

    reference_flat = reference.reshape(-1)
    estimate_flat = estimate.reshape(-1)

    reference_centered = (
        reference_flat
        - reference_flat.mean()
    )

    estimate_centered = (
        estimate_flat
        - estimate_flat.mean()
    )

    denominator = (
        np.linalg.norm(reference_centered)
        * np.linalg.norm(estimate_centered)
    )

    if denominator == 0.0:
        return 0.0

    return float(
        np.dot(
            reference_centered,
            estimate_centered,
        )
        / denominator
    )


def evaluate_reconstruction(
    reference: np.ndarray,
    estimate: np.ndarray,
) -> dict[str, float]:

    return {
        "mse": mse(
            reference,
            estimate,
        ),
        "rmse": rmse(
            reference,
            estimate,
        ),
        "snr_db": snr_db(
            reference,
            estimate,
        ),
        "correlation": correlation(
            reference,
            estimate,
        ),
    }


__all__ = [
    "mse",
    "rmse",
    "snr_db",
    "correlation",
    "evaluate_reconstruction",
]
