

import numpy as np
import mne

from src.preprocessing.quality import assess_eeg_quality


def assess_full_recording_quality(
    raw,
    window_seconds=30.0,
    abnormal_window_fraction=0.25,
):
    """
    Assess EEG channel quality across the full recording.

    The recording is divided into fixed-size windows.
    Each window is evaluated using the channel-level
    quality detector.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        EEG recording.

    window_seconds : float
        Length of each quality-assessment window.

    abnormal_window_fraction : float
        Fraction of windows in which a channel must be
        abnormal before it is considered persistently bad.

    Returns
    -------
    bad_channels : list[str]
        Channels that are persistently abnormal.

    report : dict
        Full-recording quality information.
    """

    # ---------------------------------------------------------
    # 1. Validate input
    # ---------------------------------------------------------

    if raw is None:
        raise ValueError(
            "EEG input cannot be None."
        )

    if not isinstance(
        raw,
        mne.io.BaseRaw
    ):
        raise TypeError(
            "EEG input must be an MNE Raw object."
        )

    if window_seconds <= 0:
        raise ValueError(
            "window_seconds must be greater than zero."
        )

    if not 0 < abnormal_window_fraction <= 1:
        raise ValueError(
            "abnormal_window_fraction must be "
            "between 0 and 1."
        )

    # ---------------------------------------------------------
    # 2. Work on a copy
    # ---------------------------------------------------------

    recording = raw.copy()

    duration = recording.times[-1]

    # Number of complete windows.
    n_windows = int(
        np.floor(
            duration / window_seconds
        )
    )

    # Include a final partial window when needed.
    if duration % window_seconds > 0:
        n_windows += 1

    if n_windows == 0:
        raise ValueError(
            "Recording is too short for quality assessment."
        )

    # ---------------------------------------------------------
    # 3. Initialize per-channel tracking
    # ---------------------------------------------------------

    eeg_picks = mne.pick_types(
        recording.info,
        eeg=True,
        exclude=[]
    )

    if len(eeg_picks) == 0:
        raise ValueError(
            "No EEG channels found."
        )

    channel_names = [
        recording.ch_names[index]
        for index in eeg_picks
    ]

    abnormal_counts = {
        channel: 0
        for channel in channel_names
    }

    window_records = []

    # ---------------------------------------------------------
    # 4. Analyze each window
    # ---------------------------------------------------------

    for window_index in range(n_windows):

        start_time = (
            window_index *
            window_seconds
        )

        end_time = min(
            start_time +
            window_seconds,
            duration
        )

        segment = recording.copy()

        segment.crop(
            tmin=start_time,
            tmax=end_time
        )

        bad_channels, quality_report = (
            assess_eeg_quality(segment)
        )

        # Count abnormal windows.
        for channel in bad_channels:
            abnormal_counts[channel] += 1

        window_records.append(
            {
                "window_index": window_index + 1,
                "start": float(start_time),
                "end": float(end_time),
                "bad_channels": list(
                    bad_channels
                ),
            }
        )

    # ---------------------------------------------------------
    # 5. Calculate persistence
    # ---------------------------------------------------------

    final_bad_channels = []

    channel_summary = {}

    for channel in channel_names:

        abnormal_windows = abnormal_counts[
            channel
        ]

        abnormal_fraction = (
            abnormal_windows /
            n_windows
        )

        is_persistent = (
            abnormal_fraction >=
            abnormal_window_fraction
        )

        if is_persistent:
            final_bad_channels.append(
                channel
            )

        channel_summary[channel] = {
            "abnormal_windows": int(
                abnormal_windows
            ),
            "total_windows": int(
                n_windows
            ),
            "abnormal_fraction": float(
                abnormal_fraction
            ),
            "persistent_bad": bool(
                is_persistent
            ),
        }

    # ---------------------------------------------------------
    # 6. Return complete report
    # ---------------------------------------------------------

    report = {
        "window_seconds": float(
            window_seconds
        ),
        "total_windows": int(
            n_windows
        ),
        "abnormal_window_fraction": float(
            abnormal_window_fraction
        ),
        "channels": channel_summary,
        "windows": window_records,
    }

    return (
        final_bad_channels,
        report
    )