import numpy as np
import mne


def assess_eeg_quality(raw):
    """
    Dataset-independent EEG channel quality assessment.

    A channel is considered suspicious when multiple
    independent quality indicators support the decision.

    Returns
    -------
    bad_channels : list[str]
        Channels identified as bad.

    quality_report : list[dict]
        Quality information for every EEG channel.
    """

    if raw is None:
        raise ValueError(
            "EEG input cannot be None."
        )

    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError(
            "EEG input must be an MNE Raw object."
        )

    eeg = raw.copy()

    # ---------------------------------------------------------
    # 1. Select EEG channels
    # ---------------------------------------------------------

    eeg_picks = mne.pick_types(
        eeg.info,
        eeg=True,
        exclude=[]
    )

    if len(eeg_picks) == 0:
        raise ValueError(
            "No EEG channels found."
        )

    data = eeg.get_data(
        picks=eeg_picks
    )

    channel_names = [
        eeg.ch_names[index]
        for index in eeg_picks
    ]

    # ---------------------------------------------------------
    # 2. Per-channel basic statistics
    # ---------------------------------------------------------

    std_values = np.std(
        data,
        axis=1
    )

    range_values = np.ptp(
        data,
        axis=1
    )

    # ---------------------------------------------------------
    # 3. Robust population references
    # ---------------------------------------------------------

    valid_std = std_values[
        np.isfinite(std_values)
    ]

    valid_range = range_values[
        np.isfinite(range_values)
    ]

    if len(valid_std) == 0:
        raise ValueError(
            "No valid EEG standard deviations found."
        )

    if len(valid_range) == 0:
        raise ValueError(
            "No valid EEG ranges found."
        )

    median_std = np.median(
        valid_std
    )

    median_range = np.median(
        valid_range
    )

    std_reference = max(
        median_std,
        1e-12
    )

    range_reference = max(
        median_range,
        1e-12
    )

    quality_report = []
    bad_channels = []

    # ---------------------------------------------------------
    # 4. Analyze each channel
    # ---------------------------------------------------------

    for index, channel in enumerate(
        channel_names
    ):

        signal = data[index]

        mean_value = np.mean(
            signal
        )

        std_value = np.std(
            signal
        )

        signal_range = np.ptp(
            signal
        )

        nan_count = int(
            np.isnan(signal).sum()
        )

        inf_count = int(
            np.isinf(signal).sum()
        )

        # -----------------------------------------------------
        # Quality indicators
        # -----------------------------------------------------

        is_flat = (
            std_value < 1e-12
        )

        has_invalid_values = (
            nan_count > 0
            or
            inf_count > 0
        )

        excessive_std = (
            std_value
            >
            10.0 * std_reference
        )

        excessive_range = (
            signal_range
            >
            10.0 * range_reference
        )

        # -----------------------------------------------------
        # Evidence score
        # -----------------------------------------------------

        score = 0
        reasons = []

        if is_flat:
            score += 2
            reasons.append("flat")

        if has_invalid_values:
            score += 2
            reasons.append("invalid_values")

        if excessive_std:
            score += 1
            reasons.append("excessive_std")

        if excessive_range:
            score += 1
            reasons.append("excessive_range")

        # -----------------------------------------------------
        # Final decision
        #
        # Invalid/flat channels are automatically bad.
        # Otherwise require >= 2 independent indicators.
        # -----------------------------------------------------

        is_bad = (
            is_flat
            or
            has_invalid_values
            or
            score >= 2
        )

        if is_bad:
            bad_channels.append(
                channel
            )

        quality_report.append(
            {
                "channel": channel,
                "mean": float(mean_value),
                "std": float(std_value),
                "range": float(signal_range),
                "nan_count": nan_count,
                "inf_count": inf_count,
                "score": int(score),
                "bad": bool(is_bad),
                "reason": (
                    ",".join(reasons)
                    if reasons
                    else "acceptable"
                ),
            }
        )

    return (
        bad_channels,
        quality_report
    )