"""
Step 53 — Extract Band-Power Features for All Filter Conditions

Neuro-Symphony
EEG filter comparison experiment

Purpose
-------
For sub-01, extract 64-channel × 5-band band-power features
for every filter condition defined in Step 51.

Experimental design
-------------------
- Subject: sub-01
- Original BDF
- 720 task trials
- 64 EEG channels
- Same representative/event timing framework as Steps 44/51
- Filtering is performed on continuous EEG with context
- Trial is extracted AFTER filtering
- Welch PSD
- Five bands:
    Delta: 1–4 Hz
    Theta: 4–8 Hz
    Alpha: 8–13 Hz
    Beta: 13–30 Hz
    Gamma: 30–40 Hz
- 320 features:
    64 channels × 5 bands

No ML is performed in this step.
No test-set information is used.
Original BDF is never modified.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import mne
from scipy import signal


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sub-01"
    / "sub-01_task-eeg_eeg.bdf"
)

EVENTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sub-01"
    / "sub-01_task-eeg_events.tsv"
)

FILTER_CONDITIONS_PATH = (
    PROJECT_ROOT
    / "results"
    / "eeg"
    / "filter_analysis"
    / "sub01"
    / "step51"
    / "step51_filter_conditions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "eeg"
    / "filter_analysis"
    / "sub01"
    / "step53"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CONSTANTS
# =============================================================================

FS = 2048.0

TRIAL_DURATION = 3.0
TRIAL_SAMPLES = 6142

CONTEXT_SECONDS = 15.0
CONTEXT_SAMPLES = int(CONTEXT_SECONDS * FS)

BUTTER_ORDER = 4
FIR_NUMTAPS = 4097

NOTCH_FREQ = 50.0

EEG_CHANNELS = [
    f"A{i}" for i in range(1, 33)
] + [
    f"B{i}" for i in range(1, 33)
]

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 40.0),
}


# =============================================================================
# HELPERS
# =============================================================================

def print_header(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def butterworth_filter(data, low_hz, high_hz):
    """
    Zero-phase Butterworth band-pass filter.

    data shape:
        (n_channels, n_samples)
    """

    nyquist = FS / 2.0

    low = low_hz / nyquist
    high = high_hz / nyquist

    sos = signal.butter(
        BUTTER_ORDER,
        [low, high],
        btype="bandpass",
        output="sos",
    )

    return signal.sosfiltfilt(
        sos,
        data,
        axis=-1,
    )


def fir_filter(data, low_hz, high_hz):
    """
    Zero-phase FIR band-pass filter.

    scipy.signal.firwin + filtfilt is used here so that
    the filtering is explicitly performed on the continuous
    context before extracting the trial.
    """

    taps = signal.firwin(
        FIR_NUMTAPS,
        [low_hz, high_hz],
        pass_zero=False,
        fs=FS,
        window="hamming",
    )

    return signal.filtfilt(
        taps,
        [1.0],
        data,
        axis=-1,
        method="pad",
    )


def notch_filter(data):
    """
    Zero-phase 50 Hz notch filter.
    """

    b, a = signal.iirnotch(
        w0=NOTCH_FREQ,
        Q=30.0,
        fs=FS,
    )

    return signal.filtfilt(
        b,
        a,
        data,
        axis=-1,
    )


def apply_filter_condition(data, filter_type, low_hz, high_hz):
    """
    Apply exactly one Step-51 filter condition.

    Supported types:
        butterworth
        fir
        notch+butterworth
        notch+fir
    """

    if filter_type == "butterworth":
        return butterworth_filter(
            data,
            low_hz,
            high_hz,
        )

    if filter_type == "fir":
        return fir_filter(
            data,
            low_hz,
            high_hz,
        )

    if filter_type == "notch+butterworth":

        filtered = notch_filter(data)

        return butterworth_filter(
            filtered,
            low_hz,
            high_hz,
        )

    if filter_type == "notch+fir":

        filtered = notch_filter(data)

        return fir_filter(
            filtered,
            low_hz,
            high_hz,
        )

    raise ValueError(
        f"Unknown filter type: {filter_type}"
    )


def compute_band_power(trial_data):
    """
    Compute Welch band power.

    Parameters
    ----------
    trial_data:
        ndarray of shape
        (64, n_samples)

    Returns
    -------
    ndarray:
        shape (64, 5)
    """

    n_samples = trial_data.shape[1]

    # Use the same PSD strategy as Step 44.
    n_fft = min(4096, n_samples)

    n_overlap = min(
        2048,
        n_fft // 2,
    )

    # scipy Welch PSD
    freqs, psd = signal.welch(
        trial_data,
        fs=FS,
        nperseg=n_fft,
        noverlap=n_overlap,
        nfft=n_fft,
        axis=-1,
        scaling="density",
    )

    band_features = []

    for band_name, (low, high) in BANDS.items():

        mask = (
            (freqs >= low)
            & (freqs <= high)
        )

        if not np.any(mask):
            raise RuntimeError(
                f"No frequency bins found for {band_name}"
            )

        band_power = np.trapezoid(
            psd[:, mask],
            freqs[mask],
            axis=-1,
        )

        band_features.append(
            band_power
        )

    return np.stack(
        band_features,
        axis=1,
    )


def get_eeg_channel_indices(raw):
    """
    Return the 64 EEG channel indices.

    Step 51 uses the BioSemi A1-A32/B1-B32 channels.
    """

    missing = [
        ch for ch in EEG_CHANNELS
        if ch not in raw.ch_names
    ]

    if missing:
        raise RuntimeError(
            f"Missing EEG channels: {missing}"
        )

    return [
        raw.ch_names.index(ch)
        for ch in EEG_CHANNELS
    ]


def load_task_events():
    """
    Load and validate the 720 task trials.
    """

    events_df = pd.read_csv(
        EVENTS_PATH,
        sep="\t",
    )

    required_columns = {
        "sample",
        "trial_type",
    }

    missing = required_columns - set(
        events_df.columns
    )

    if missing:
        raise RuntimeError(
            f"Missing event columns: {sorted(missing)}"
        )

    task_events = events_df[
        events_df["trial_type"].astype(str).str.endswith(
            "_task"
        )
    ].copy()

    if len(task_events) != 720:
        raise RuntimeError(
            f"Expected 720 task trials, "
            f"found {len(task_events)}"
        )

    task_events = task_events.reset_index(
        drop=False
    )

    task_events["label_name"] = (
        task_events["trial_type"].astype(str)
    )

    return task_events


def make_feature_name(channel, band):
    return f"{channel}_{band}_power"


# =============================================================================
# MAIN
# =============================================================================

def main():

    print_header(
        "STEP 53 — FILTERED BAND-POWER FEATURE EXTRACTION"
    )

    # -------------------------------------------------------------------------
    # Load filter conditions
    # -------------------------------------------------------------------------

    if not FILTER_CONDITIONS_PATH.exists():
        raise FileNotFoundError(
            f"Filter conditions file not found:\n"
            f"{FILTER_CONDITIONS_PATH}"
        )

    conditions_df = pd.read_csv(
        FILTER_CONDITIONS_PATH
    )

    expected_columns = {
        "condition",
        "filter_type",
        "band",
        "low_hz",
        "high_hz",
    }

    missing = expected_columns - set(
        conditions_df.columns
    )

    if missing:
        raise RuntimeError(
            f"Missing filter-condition columns: {sorted(missing)}"
        )

    print(
        f"Filter conditions loaded: "
        f"{len(conditions_df)}"
    )

    # -------------------------------------------------------------------------
    # Load events
    # -------------------------------------------------------------------------

    print()
    print("Loading task events...")

    task_events = load_task_events()

    print(
        f"✓ Task trials: {len(task_events)}"
    )

    print(
        f"✓ Unique labels: "
        f"{task_events['label_name'].nunique()}"
    )

    print(
        task_events["label_name"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # -------------------------------------------------------------------------
    # Load BDF
    # -------------------------------------------------------------------------

    print()
    print("Loading original BDF...")

    raw = mne.io.read_raw_bdf(
        BDF_PATH,
        preload=False,
        verbose=False,
    )

    print(raw)

    eeg_indices = get_eeg_channel_indices(
        raw
    )

    print(
        f"✓ EEG channels: {len(eeg_indices)}"
    )

    if len(eeg_indices) != 64:
        raise RuntimeError(
            "Expected exactly 64 EEG channels."
        )

    # -------------------------------------------------------------------------
    # Validate sampling rate
    # -------------------------------------------------------------------------

    actual_fs = raw.info["sfreq"]

    if not np.isclose(
        actual_fs,
        FS,
    ):
        raise RuntimeError(
            f"Unexpected sampling rate: "
            f"{actual_fs}"
        )

    # -------------------------------------------------------------------------
    # Generate feature names
    # -------------------------------------------------------------------------

    feature_names = [
        make_feature_name(
            channel,
            band,
        )
        for channel in EEG_CHANNELS
        for band in BANDS.keys()
    ]

    if len(feature_names) != 320:
        raise RuntimeError(
            f"Expected 320 feature names, "
            f"got {len(feature_names)}"
        )

    # -------------------------------------------------------------------------
    # Process every filter condition
    # -------------------------------------------------------------------------

    all_condition_summaries = []

    for condition_number, condition_row in conditions_df.iterrows():

        condition = str(
            condition_row["condition"]
        )

        filter_type = str(
            condition_row["filter_type"]
        )

        band_name = str(
            condition_row["band"]
        )

        low_hz = float(
            condition_row["low_hz"]
        )

        high_hz = float(
            condition_row["high_hz"]
        )

        print()
        print("-" * 80)
        print(
            f"CONDITION "
            f"{condition_number + 1}/{len(conditions_df)}"
        )
        print("-" * 80)

        print(
            f"Condition : {condition}"
        )
        print(
            f"Filter    : {filter_type}"
        )
        print(
            f"Band      : {band_name}"
        )
        print(
            f"Range     : {low_hz:g}–{high_hz:g} Hz"
        )

        condition_rows = []

        # ---------------------------------------------------------------------
        # Process all 720 trials
        # ---------------------------------------------------------------------

        for trial_position, event in task_events.iterrows():

            start_sample = int(
                event["sample"]
            )

            # Context around trial
            context_start = max(
                0,
                start_sample - CONTEXT_SAMPLES,
            )

            context_stop = min(
                raw.n_times,
                start_sample
                + TRIAL_SAMPLES
                + CONTEXT_SAMPLES,
            )

            context_data = raw.get_data(
                picks=eeg_indices,
                start=context_start,
                stop=context_stop,
            )

            if context_data.shape[0] != 64:
                raise RuntimeError(
                    "Unexpected EEG channel count."
                )

            # ---------------------------------------------------------------
            # Filter continuous context
            # ---------------------------------------------------------------

            filtered_context = apply_filter_condition(
                context_data,
                filter_type,
                low_hz,
                high_hz,
            )

            if not np.isfinite(
                filtered_context
            ).all():
                raise RuntimeError(
                    f"NaN/Inf detected after filtering "
                    f"condition={condition}, "
                    f"trial={trial_position}"
                )

            # ---------------------------------------------------------------
            # Locate trial inside filtered context
            # ---------------------------------------------------------------

            trial_start = (
                start_sample
                - context_start
            )

            trial_stop = (
                trial_start
                + TRIAL_SAMPLES
            )

            if trial_stop > filtered_context.shape[1]:

                raise RuntimeError(
                    f"Trial exceeds filtered context: "
                    f"condition={condition}, "
                    f"trial={trial_position}"
                )

            trial_data = filtered_context[
                :,
                trial_start:trial_stop,
            ]

            if trial_data.shape != (
                64,
                TRIAL_SAMPLES,
            ):
                raise RuntimeError(
                    f"Unexpected trial shape: "
                    f"{trial_data.shape}"
                )

            # ---------------------------------------------------------------
            # Welch band power
            # ---------------------------------------------------------------

            band_power = compute_band_power(
                trial_data
            )

            if band_power.shape != (
                64,
                5,
            ):
                raise RuntimeError(
                    f"Unexpected band-power shape: "
                    f"{band_power.shape}"
                )

            # ---------------------------------------------------------------
            # Flatten channel × band
            # ---------------------------------------------------------------

            features = band_power.reshape(
                -1
            )

            row = {
                "subject": "sub-01",
                "trial": int(
                    trial_position
                ),
                "label": int(
                    event["index"]
                ),
                "label_name": str(
                    event["label_name"]
                ),
                "event_row": int(
                    event["index"]
                ),
                "start_sample": start_sample,
                "duration_seconds": TRIAL_DURATION,
                "filter_condition": condition,
                "filter_type": filter_type,
                "filter_band": band_name,
                "filter_low_hz": low_hz,
                "filter_high_hz": high_hz,
            }

            row.update(
                dict(
                    zip(
                        feature_names,
                        features,
                    )
                )
            )

            condition_rows.append(
                row
            )

            if (
                (trial_position + 1) % 100 == 0
                or trial_position == 0
                or trial_position == len(task_events) - 1
            ):
                print(
                    f"  Trial "
                    f"{trial_position + 1:3d}/"
                    f"{len(task_events)}"
                )

        # ---------------------------------------------------------------------
        # Save condition
        # ---------------------------------------------------------------------

        condition_df = pd.DataFrame(
            condition_rows
        )

        expected_rows = 720

        if len(condition_df) != expected_rows:
            raise RuntimeError(
                f"Expected {expected_rows} rows, "
                f"got {len(condition_df)}"
            )

        feature_only = condition_df[
            feature_names
        ]

        if not np.isfinite(
            feature_only.to_numpy()
        ).all():
            raise RuntimeError(
                f"NaN/Inf detected in final "
                f"features for {condition}"
            )

        condition_file = (
            OUTPUT_DIR
            / f"{condition}_band_power.csv"
        )

        condition_df.to_csv(
            condition_file,
            index=False,
        )

        print(
            f"✓ Saved: {condition_file}"
        )

        all_condition_summaries.append(
            {
                "condition": condition,
                "filter_type": filter_type,
                "filter_band": band_name,
                "low_hz": low_hz,
                "high_hz": high_hz,
                "n_trials": len(condition_df),
                "n_features": len(feature_names),
                "file": str(condition_file),
            }
        )

    # =========================================================================
    # Combined metadata/summary
    # =========================================================================

    summary_df = pd.DataFrame(
        all_condition_summaries
    )

    summary_path = (
        OUTPUT_DIR
        / "step53_condition_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # =========================================================================
    # Experiment information
    # =========================================================================

    experiment_info = {
        "step": 53,
        "subject": "sub-01",
        "sampling_rate_hz": FS,
        "n_trials": 720,
        "n_eeg_channels": 64,
        "n_frequency_bands": 5,
        "n_features_per_trial": 320,
        "bands": BANDS,
        "context_seconds": CONTEXT_SECONDS,
        "trial_samples": TRIAL_SAMPLES,
        "trial_duration_seconds": TRIAL_DURATION,
        "welch_n_fft": 4096,
        "welch_overlap": 2048,
        "n_filter_conditions": len(
            conditions_df
        ),
        "filter_conditions_file": str(
            FILTER_CONDITIONS_PATH
        ),
        "original_bdf_modified": False,
        "machine_learning_performed": False,
        "data_leakage_introduced": False,
    }

    info_path = (
        OUTPUT_DIR
        / "step53_experiment_info.json"
    )

    with open(
        info_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            experiment_info,
            f,
            indent=2,
        )

    # =========================================================================
    # Final validation
    # =========================================================================

    print_header(
        "STEP 53 COMPLETE"
    )

    print(
        f"✓ Conditions processed : "
        f"{len(conditions_df)}"
    )

    print(
        f"✓ Trials per condition : 720"
    )

    print(
        f"✓ EEG channels         : 64"
    )

    print(
        f"✓ Bands per trial      : 5"
    )

    print(
        f"✓ Features per trial   : 320"
    )

    print()
    print(
        f"✓ Summary saved:"
    )
    print(
        f"  {summary_path}"
    )

    print()
    print(
        f"✓ Original BDF was NOT modified."
    )

    print(
        f"✓ No ML was performed."
    )

    print(
        f"✓ No train/validation/test information "
        f"was used."
    )

    print()
    print(
        f"Output directory:"
    )
    print(
        f"  {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()