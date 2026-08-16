"""
Universal Phase 2 EEG preprocessing pipeline.

Dataset-independent preprocessing with dataset-specific
channel typing and montage configuration.

Supports:
    1. MNE Raw
    2. Phase 1 EEGData-like objects
"""

from dataclasses import is_dataclass, replace

import numpy as np
import mne

from src.preprocessing.config import EEGPreprocessingConfig
from src.preprocessing.full_recording_quality import (
    assess_full_recording_quality,
)
from src.preprocessing.montage import apply_eeg_montage


# ============================================================
# PHASE 1 EEGData -> MNE Raw
# ============================================================

def _to_raw(eeg_input):

    if isinstance(eeg_input, mne.io.BaseRaw):
        return eeg_input.copy(), "raw"

    required_attributes = [
        "data",
        "channel_names",
        "sampling_rate",
    ]

    missing_attributes = [
        attribute
        for attribute in required_attributes
        if not hasattr(eeg_input, attribute)
    ]

    if missing_attributes:
        raise TypeError(
            "EEG input is missing required "
            f"attributes: {missing_attributes}"
        )

    data = np.asarray(eeg_input.data)

    channel_names = tuple(eeg_input.channel_names)

    sampling_rate = float(eeg_input.sampling_rate)

    if data.ndim != 2:
        raise ValueError(
            "EEGData.data must have shape "
            "(channels, samples)."
        )

    if data.shape[0] != len(channel_names):
        raise ValueError(
            "Number of channel names does not "
            "match the number of EEG data channels."
        )

    channel_types = getattr(
        eeg_input,
        "channel_types",
        None,
    )

    if (
        channel_types is None
        or len(channel_types) != len(channel_names)
    ):
        channel_types = [
            "eeg"
            for _ in channel_names
        ]

    info = mne.create_info(
        ch_names=list(channel_names),
        sfreq=sampling_rate,
        ch_types=list(channel_types),
    )

    raw = mne.io.RawArray(
        data,
        info,
    )

    return raw, "eeg_data"


# ============================================================
# MNE Raw -> Phase 1 EEGData
# ============================================================

def _restore_output(
    processed,
    original,
    input_kind,
):

    if input_kind == "raw":
        return processed

    if not is_dataclass(original):
        return processed

    metadata = dict(
        getattr(
            original,
            "metadata",
            {},
        )
        or {}
    )

    metadata["phase2"] = {
        "bad_channels": list(
            processed.info["bads"]
        ),
        "preprocessing": {
            "bandpass": "1-40 Hz",
            "notch": "50 Hz",
            "interpolation": "spherical spline",
            "reference": "average",
        },
    }

    return replace(
        original,
        data=np.asarray(
            processed.get_data()
        ),
        channel_names=tuple(
            processed.ch_names
        ),
        sampling_rate=float(
            processed.info["sfreq"]
        ),
        channel_types=tuple(
            processed.get_channel_types()
        ),
        metadata=metadata,
    )


# ============================================================
# VALID EEG SENSOR POSITION CHECK
# ============================================================

def _valid_eeg_position_channels(
    raw,
    eeg_channel_names,
):
    """
    Return EEG channels that have finite, usable
    3-D sensor positions.

    This prevents MNE interpolation from receiving
    NaN/Inf sensor coordinates.
    """

    valid = []

    for channel in eeg_channel_names:

        if channel not in raw.ch_names:
            continue

        channel_index = raw.ch_names.index(channel)

        position = raw.info["chs"][channel_index]["loc"][:3]

        if position is None:
            continue

        position = np.asarray(
            position,
            dtype=float,
        )

        if position.shape != (3,):
            continue

        if not np.isfinite(position).all():
            continue

        if np.allclose(position, 0.0):
            continue

        valid.append(channel)

    return valid


# ============================================================
# SAFE EEG INTERPOLATION
# ============================================================

def _interpolate_bad_eeg_channels(
    raw,
    eeg_channel_names,
    interpolation_method,
):
    """
    Safely interpolate bad EEG channels.

    Only EEG channels with valid finite sensor
    positions participate in interpolation.

    Auxiliary and trigger channels are never
    interpolated.
    """

    if not raw.info["bads"]:
        return

    # --------------------------------------------------------
    # Only current EEG channels may be interpolated.
    # --------------------------------------------------------

    bad_eeg = [
        channel
        for channel in raw.info["bads"]
        if channel in eeg_channel_names
    ]

    if not bad_eeg:
        raw.info["bads"] = []
        return

    # --------------------------------------------------------
    # Find EEG channels with valid positions.
    # --------------------------------------------------------

    valid_position_channels = (
        _valid_eeg_position_channels(
            raw,
            eeg_channel_names,
        )
    )

    # --------------------------------------------------------
    # Only bad EEG channels with valid positions can
    # actually be interpolated.
    # --------------------------------------------------------

    interpolatable_bad = [
        channel
        for channel in bad_eeg
        if channel in valid_position_channels
    ]

    # --------------------------------------------------------
    # Bad channels with invalid positions cannot be
    # mathematically interpolated. Remove them from the
    # interpolation list rather than allowing MNE to fail.
    # --------------------------------------------------------

    non_interpolatable_bad = [
        channel
        for channel in bad_eeg
        if channel not in valid_position_channels
    ]

    if non_interpolatable_bad:
        print(
            "Skipping interpolation for EEG channels "
            "without valid sensor positions:"
        )

        for channel in non_interpolatable_bad:
            print(f"  - {channel}")

    if not interpolatable_bad:
        raw.info["bads"] = []
        return

    # --------------------------------------------------------
    # Create a temporary Raw object containing ONLY EEG
    # channels with valid sensor positions.
    #
    # This guarantees that MNE's interpolation matrix
    # cannot contain auxiliary channels or invalid positions.
    # --------------------------------------------------------

    interpolation_channels = list(
        valid_position_channels
    )

    temp = raw.copy().pick(
        interpolation_channels
    )

    temp.info["bads"] = [
        channel
        for channel in interpolatable_bad
        if channel in temp.ch_names
    ]

    # --------------------------------------------------------
    # Perform spherical-spline interpolation.
    # Explicit origin avoids dependence on digitization.
    # --------------------------------------------------------

    temp.interpolate_bads(
        reset_bads=False,
        origin=(0.0, 0.0, 0.0),
        method={
            "eeg": interpolation_method
        },
    )

    # --------------------------------------------------------
    # Copy interpolated data back into original Raw.
    # --------------------------------------------------------

    for channel in interpolatable_bad:

        if channel not in temp.ch_names:
            continue

        original_index = raw.ch_names.index(
            channel
        )

        temp_index = temp.ch_names.index(
            channel
        )

        raw._data[
            original_index
        ] = temp.get_data(
            picks=[temp_index]
        )[0]

    # --------------------------------------------------------
    # Interpolated channels are no longer considered bad.
    # Any non-interpolatable channels are also removed from
    # MNE's bad list so they cannot poison later operations.
    # --------------------------------------------------------

    raw.info["bads"] = [
        channel
        for channel in raw.info["bads"]
        if channel not in interpolatable_bad
        and channel not in non_interpolatable_bad
    ]


# ============================================================
# MAIN PHASE 2 EEG PIPELINE
# ============================================================

def preprocess_eeg(
    eeg_input,
    preprocessing_config=None,
    dataset_config=None,
):

    # ========================================================
    # 1. Universal preprocessing configuration
    # ========================================================

    if preprocessing_config is None:
        preprocessing_config = (
            EEGPreprocessingConfig()
        )

    # ========================================================
    # 2. Convert input to MNE Raw
    # ========================================================

    original = eeg_input

    raw, input_kind = _to_raw(
        eeg_input
    )

    # ========================================================
    # 3. Dataset-specific channel preparation
    # ========================================================

    if dataset_config is not None:

        raw = apply_eeg_montage(
            raw,
            channel_mapping=dataset_config.get(
                "channel_mapping"
            ),
            montage_name=dataset_config.get(
                "montage"
            ),
            eeg_channel_names=dataset_config.get(
                "eeg_channel_names"
            ),
            auxiliary_channel_names=dataset_config.get(
                "auxiliary_channel_names"
            ),
            trigger_channel_names=dataset_config.get(
                "trigger_channel_names"
            ),
        )

    # ========================================================
    # 4. Select ONLY actual EEG channels
    # ========================================================

    eeg_picks = mne.pick_types(
        raw.info,
        eeg=True,
        exclude=[],
    )

    if len(eeg_picks) == 0:
        raise ValueError(
            "No EEG channels were found "
            "after dataset preparation."
        )

    eeg_channel_names = [
        raw.ch_names[index]
        for index in eeg_picks
    ]

    # ========================================================
    # 5. Full-recording quality assessment
    # ========================================================

    if preprocessing_config.detect_bad_channels:

        bad_channels, quality_report = (
            assess_full_recording_quality(
                raw,
                window_seconds=30.0,
                abnormal_window_fraction=0.25,
            )
        )

    else:

        bad_channels = []
        quality_report = None

    # --------------------------------------------------------
    # IMPORTANT:
    # Keep ONLY currently existing EEG channels.
    # Never allow MISC/STIM channels into bad list.
    # --------------------------------------------------------

    raw.info["bads"] = [
        channel
        for channel in bad_channels
        if channel in eeg_channel_names
    ]

    # ========================================================
    # 6. Band-pass filtering
    # ========================================================

    raw.filter(
        l_freq=(
            preprocessing_config.low_frequency
        ),
        h_freq=(
            preprocessing_config.high_frequency
        ),
        picks=eeg_picks,
        fir_design="firwin",
    )

    # ========================================================
    # 7. Power-line notch filtering
    # ========================================================

    raw.notch_filter(
        freqs=[
            preprocessing_config.notch_frequency
        ],
        picks=eeg_picks,
    )

    # ========================================================
    # 8. SAFE BAD EEG INTERPOLATION
    # ========================================================

    if (
        preprocessing_config.interpolate_bad_channels
        and raw.info["bads"]
    ):

        bad_non_eeg = [
            channel
            for channel in raw.info["bads"]
            if channel not in eeg_channel_names
        ]

        if bad_non_eeg:
            raise RuntimeError(
                "Non-EEG channels were included "
                f"in the bad-channel list: {bad_non_eeg}"
            )

        _interpolate_bad_eeg_channels(
            raw=raw,
            eeg_channel_names=eeg_channel_names,
            interpolation_method=(
                preprocessing_config
                .interpolation_method
            ),
        )

    # ========================================================
    # 9. Average reference
    # ========================================================

    reference = (
        preprocessing_config.reference
    )

    if reference == "average":

        raw.set_eeg_reference(
            "average",
            projection=False,
        )

    elif reference not in (
        None,
        "none",
    ):

        raise ValueError(
            f"Unsupported EEG reference: "
            f"{reference}"
        )

    # ========================================================
    # 10. Final validation
    # ========================================================

    final_data = raw.get_data(
        picks=eeg_picks
    )

    if np.isnan(
        final_data
    ).any():

        raise ValueError(
            "Final EEG contains NaN values."
        )

    if np.isinf(
        final_data
    ).any():

        raise ValueError(
            "Final EEG contains infinite values."
        )

    # ========================================================
    # 11. Restore output representation
    # ========================================================

    processed = _restore_output(
        raw,
        original,
        input_kind,
    )

    return (
        processed,
        quality_report,
    )
