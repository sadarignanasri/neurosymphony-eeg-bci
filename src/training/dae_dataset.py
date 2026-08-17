"""
Neuro-Symphony Phase 2 — DAE Dataset Pipeline.

Creates deterministic temporal windows and noisy/clean pairs
for denoising autoencoder training.

Input:
    NumPy array with shape (channels, samples)

Output:
    noisy windows and corresponding clean windows
    with shape (channels, window_size)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DAESplit:
    """Indices defining train/validation/test separation."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def create_windows(
    data: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """
    Convert a continuous signal into non-overlapping windows.

    Parameters
    ----------
    data:
        Signal with shape (channels, samples).

    window_size:
        Number of samples in each temporal window.

    Returns
    -------
    np.ndarray
        Windows with shape:
        (windows, channels, window_size)
    """

    data = np.asarray(
        data,
        dtype=np.float32,
    )

    if data.ndim != 2:
        raise ValueError(
            "DAE input must have shape (channels, samples)."
        )

    if window_size <= 0:
        raise ValueError(
            "window_size must be positive."
        )

    channels, samples = data.shape

    number_of_windows = samples // window_size

    if number_of_windows == 0:
        raise ValueError(
            "Signal is shorter than the requested window size."
        )

    usable_samples = (
        number_of_windows * window_size
    )

    trimmed = data[:, :usable_samples]

    windows = trimmed.reshape(
        channels,
        number_of_windows,
        window_size,
    )

    windows = np.transpose(
        windows,
        (1, 0, 2),
    )

    return windows.copy()


def split_indices(
    number_of_windows: int,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    random_seed: int = 42,
) -> DAESplit:
    """
    Deterministically split windows into train/validation/test sets.

    The split is performed at the window level.
    """

    if number_of_windows < 3:
        raise ValueError(
            "At least 3 windows are required."
        )

    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError(
            "validation_fraction must be in [0, 1)."
        )

    if not 0.0 <= test_fraction < 1.0:
        raise ValueError(
            "test_fraction must be in [0, 1)."
        )

    if (
        validation_fraction
        + test_fraction
        >= 1.0
    ):
        raise ValueError(
            "Validation and test fractions must sum to less than 1."
        )

    rng = np.random.default_rng(
        random_seed
    )

    indices = np.arange(
        number_of_windows
    )

    rng.shuffle(indices)

    test_count = max(
        1,
        int(
            number_of_windows
            * test_fraction
        ),
    )

    validation_count = max(
        1,
        int(
            number_of_windows
            * validation_fraction
        ),
    )

    if (
        test_count
        + validation_count
        >= number_of_windows
    ):
        raise ValueError(
            "Not enough windows for train/validation/test separation."
        )

    test_indices = indices[
        :test_count
    ]

    validation_indices = indices[
        test_count:
        test_count + validation_count
    ]

    train_indices = indices[
        test_count + validation_count:
    ]

    return DAESplit(
        train=np.sort(train_indices),
        validation=np.sort(validation_indices),
        test=np.sort(test_indices),
    )


def add_noise(
    clean_windows: np.ndarray,
    noise_std: float,
    random_seed: int,
) -> np.ndarray:
    """
    Add deterministic Gaussian noise to clean windows.

    The clean target is never modified.
    """

    clean_windows = np.asarray(
        clean_windows,
        dtype=np.float32,
    )

    if clean_windows.ndim != 3:
        raise ValueError(
            "Expected windows with shape "
            "(windows, channels, samples)."
        )

    if noise_std < 0:
        raise ValueError(
            "noise_std must be non-negative."
        )

    rng = np.random.default_rng(
        random_seed
    )

    noise = rng.normal(
        loc=0.0,
        scale=noise_std,
        size=clean_windows.shape,
    ).astype(
        np.float32
    )

    noisy_windows = (
        clean_windows + noise
    )

    return noisy_windows.astype(
        np.float32,
        copy=False,
    )


class DAEDataset(Dataset):
    """
    PyTorch dataset containing noisy input / clean target pairs.
    """

    def __init__(
        self,
        clean_windows: np.ndarray,
        noise_std: float,
        random_seed: int,
    ) -> None:

        clean_windows = np.asarray(
            clean_windows,
            dtype=np.float32,
        )

        if clean_windows.ndim != 3:
            raise ValueError(
                "Expected clean windows with shape "
                "(windows, channels, samples)."
            )

        noisy_windows = add_noise(
            clean_windows,
            noise_std=noise_std,
            random_seed=random_seed,
        )

        self.noisy = torch.from_numpy(
            noisy_windows
        )

        self.clean = torch.from_numpy(
            clean_windows.copy()
        )

    def __len__(self) -> int:
        return self.clean.shape[0]

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        return (
            self.noisy[index],
            self.clean[index],
        )


__all__ = [
    "DAESplit",
    "DAEDataset",
    "add_noise",
    "create_windows",
    "split_indices",
]
