"""
Universal EEG preprocessing configuration.

This file contains preprocessing settings that are
independent of any particular dataset.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EEGPreprocessingConfig:
    """
    Configuration for the universal EEG preprocessing pipeline.
    """

    # --------------------------------------------------------
    # Frequency filtering
    # --------------------------------------------------------

    low_frequency: float = 1.0
    high_frequency: float = 40.0
    notch_frequency: float = 50.0

    # --------------------------------------------------------
    # Bad-channel repair
    # --------------------------------------------------------

    interpolation_method: str = "spline"

    # --------------------------------------------------------
    # EEG reference
    # --------------------------------------------------------

    reference: str = "average"

    # --------------------------------------------------------
    # Quality control
    # --------------------------------------------------------

    detect_bad_channels: bool = True
    interpolate_bad_channels: bool = True