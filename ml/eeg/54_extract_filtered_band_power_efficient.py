

"""
Step 54 — Efficient Filtered Band-Power Extraction
===================================================

Purpose
-------
Efficiently test filtered EEG feature extraction using ONLY 10 trials.

This step is a new implementation and DOES NOT modify Step 53.

Main idea
---------
Instead of:

    read trial -> filter trial -> PSD -> features
    read trial -> filter trial -> PSD -> features
    ...

we:

    read one continuous EEG segment
        -> filter once
        -> extract the required trials
        -> PSD
        -> band power

This is a benchmark/validation step.

IMPORTANT
---------
- Step 53 is NOT modified.
- Only 10 trials are processed.
- No ML is performed.
- Original BDF is never modified.
- Same event timings and EEG channels are used.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy import signal


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = PROJECT_ROOT / "data" / "raw"
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "eeg"
    / "filter_analysis"
    / "sub01"
    / "step54"
)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

SUBJECT = "sub-01"

BDF_PATH = DATA_ROOT / SUBJECT / f"{SUBJECT}_task-eeg_eeg.bdf"

EVENTS_PATH = DATA_ROOT / SUBJECT / f"{SUBJECT}_task-eeg_events.tsv"

# EXACTLY 10 TRIALS FOR THIS BENCHMARK
N_TRIALS = 10

RANDOM_SEED = 42

# EEG channels used in the project
EEG_CHANNELS = (
    [f"A{i}" for i in range(1, 33)]
    + [f"B{i}" for i in range(1, 33)]
)

# Sampling frequency expected from DS004514
EXPECTED_SFREQ = 2048.0

# Trial duration
TRIAL_DURATION_SECONDS = 3.0

# Context around trials
CONTEXT_SECONDS = 15.0

# Welch parameters
FMIN = 1.0
FMAX = 40.0
N_FFT = 4096
N_OVERLAP = 2048

# Frequency bands
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 40.0),
}

EPSILON = 1e-20


# ============================================================
# FILTER CONDITIONS
# ============================================================

FILTER_CONDITIONS = [
    {
        "condition": "butterworth_broad_1_45Hz",
        "filter_type": "butterworth",
        "band": "broad_1_45Hz",
        "low_hz": 1.0,
        "high_hz": 45.0,
    },
    {
        "condition": "fir_broad_1_45Hz",
        "filter_type": "fir",
        "band": "broad_1_45Hz",
        "low_hz": 1.0,
        "high_hz": 45.0,
    },
    {
        "condition": "butterworth_delta_1_4Hz",
        "filter_type": "butterworth",
        "band": "delta_1_4Hz",
        "low_hz": 1.0,
        "high_hz": 4.0,
    },
    {
        "condition": "fir_delta_1_4Hz",
        "filter_type": "fir",
        "band": "delta_1_4Hz",
        "low_hz": 1.0,
        "high_hz": 4.0,
    },
    {
        "condition": "notch_butterworth_delta_1_4Hz",
        "filter_type": "notch+butterworth",
        "band": "delta_1_4Hz",
        "low_hz": 1.0,
        "high_hz": 4.0,
    },
    {
        "condition": "notch_fir_delta_1_4Hz",
        "filter_type": "notch+fir",
        "band": "delta_1_4Hz",
        "low_hz": 1.0,
        "high_hz": 4.0,
    },
    {
        "condition": "butterworth_theta_4_8Hz",
        "filter_type": "butterworth",
        "band": "theta_4_8Hz",
        "low_hz": 4.0,
        "high_hz": 8.0,
    },
    {
        "condition": "fir_theta_4_8Hz",
        "filter_type": "fir",
        "band": "theta_4_8Hz",
        "low_hz": 4.0,
        "high_hz": 8.0,
    },
    {
        "condition": "notch_butterworth_theta_4_8Hz",
        "filter_type": "notch+butterworth",
        "band": "theta_4_8Hz",
        "low_hz": 4.0,
        "high_hz": 8.0,
    },
    {
        "condition": "notch_fir_theta_4_8Hz",
        "filter_type": "notch+fir",
        "band": "theta_4_8Hz",
        "low_hz": 4.0,
        "high_hz": 8.0,
    },
    {
        "condition": "butterworth_alpha_8_13Hz",
        "filter_type": "butterworth",
        "band": "alpha_8_13Hz",
        "low_hz": 8.0,
        "high_hz": 13.0,
    },
    {
        "condition": "fir_alpha_8_13Hz",
        "filter_type": "fir",
        "band": "alpha_8_13Hz",
        "low_hz": 8.0,
        "high_hz": 13.0,
    },
    {
        "condition": "notch_butterworth_alpha_8_13Hz",
        "filter_type": "notch+butterworth",
        "band": "alpha_8_13Hz",
        "low_hz": 8.0,
        "high_hz": 13.0,
    },
    {
        "condition": "notch_fir_alpha_8_13Hz",
        "filter_type": "notch+fir",
        "band": "alpha_8_13Hz",
        "low_hz": 8.0,
        "high_hz": 13.0,
    },
    {
        "condition": "butterworth_beta_13_30Hz",
        "filter_type": "butterworth",
        "band": "beta_13_30Hz",
        "low_hz": 13.0,
        "high_hz": 30.0,
    },
    {
        "condition": "fir_beta_13_30Hz",
        "filter_type": "fir",
        "band": "beta_13_30Hz",
        "low_hz": 13.0,
        "high_hz": 30.0,
    },
    {
        "condition": "notch_butterworth_beta_13_30Hz",
        "filter_type": "notch+butterworth",
        "band": "beta_13_30Hz",
        "low_hz": 13.0,
        "high_hz": 30.0,
    },
    {
        "condition": "notch_fir_beta_13_30Hz",
        "filter_type": "notch+fir",
        "band": "beta_13_30Hz",
        "low_hz": 13.0,
        "high_hz": 30.0,
    },
    {
        "condition": "butterworth_gamma_30_45Hz",
        "filter_type": "butterworth",
        "band": "gamma_30_45Hz",
        "low_hz": 30.0,
        "high_hz": 45.0,
    },
    {
        "condition": "fir_gamma_30_45Hz",
        "filter_type": "fir",
        "band": "gamma_30_45Hz",
        "low_hz": 30.0,
        "high_hz": 45.0,
    },
    {
        "condition": "notch_butterworth_gamma_30_45Hz",
        "filter_type": "notch+butterworth",
        "band": "gamma_30_45Hz",
        "low_hz": 30.0,
        "high_hz": 45.0,
    },
    {
        "condition": "notch_fir_gamma_30_45Hz",
        "filter_type": "notch+fir",
        "band": "gamma_30_45Hz",
        "low_hz": 30.0,
        "high_hz": 45.0,
    },
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_task_events(events_path: Path) -> pd.DataFrame:
    """Load and validate the task events."""

    events = pd.read_csv(events_path, sep="\t")

    required_columns = {"trial_type", "sample"}

    missing = required_columns - set(events.columns)

    if missing:
        raise ValueError(
            f"Missing required event columns: {sorted(missing)}"
        )

    task_events = events[
        events["trial_type"].astype(str).str.endswith("_task")
    ].copy()

    task_events = task_events.reset_index(drop=True)

    print(f"Total event rows: {len(events)}")
    print(f"Task trials found: {len(task_events)}")

    if len(task_events) != 720:
        raise ValueError(
            f"Expected 720 task trials, found {len(task_events)}"
        )

    return task_events


def prepare_butterworth(
    data: np.ndarray,
    sfreq: float,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:
    """Apply zero-phase Butterworth band-pass filtering."""

    nyquist = sfreq / 2.0

    low = low_hz / nyquist
    high = high_hz / nyquist

    sos = signal.butter(
        N=4,
        Wn=[low, high],
        btype="bandpass",
        output="sos",
    )

    return signal.sosfiltfilt(
        sos,
        data,
        axis=-1,
    )


def prepare_fir(
    data: np.ndarray,
    sfreq: float,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:
    """
    Apply zero-phase FIR filtering.

    The filter is applied to the CONTINUOUS context segment,
    not to the 3-second trial itself.
    """

    nyquist = sfreq / 2.0

    low = low_hz / nyquist
    high = high_hz / nyquist

    numtaps = 4097

    taps = signal.firwin(
        numtaps=numtaps,
        cutoff=[low, high],
        pass_zero=False,
        window="hamming",
    )

    return signal.filtfilt(
        taps,
        [1.0],
        data,
        axis=-1,
        method="pad",
    )


def apply_notch(
    data: np.ndarray,
    sfreq: float,
) -> np.ndarray:
    """Apply a 50 Hz notch filter."""

    nyquist = sfreq / 2.0

    low = 49.0 / nyquist
    high = 51.0 / nyquist

    sos = signal.butter(
        N=4,
        Wn=[low, high],
        btype="bandstop",
        output="sos",
    )

    return signal.sosfiltfilt(
        sos,
        data,
        axis=-1,
    )


def filter_continuous_data(
    data: np.ndarray,
    sfreq: float,
    condition: dict,
) -> np.ndarray:
    """
    Filter a continuous EEG segment according to one condition.
    """

    filter_type = condition["filter_type"]
    low_hz = condition["low_hz"]
    high_hz = condition["high_hz"]

    filtered = data

    # --------------------------------------------------------
    # Notch first
    # --------------------------------------------------------

    if filter_type.startswith("notch+"):
        filtered = apply_notch(
            filtered,
            sfreq,
        )

    # --------------------------------------------------------
    # Main band-pass
    # --------------------------------------------------------

    if "butterworth" in filter_type:
        filtered = prepare_butterworth(
            filtered,
            sfreq,
            low_hz,
            high_hz,
        )

    elif "fir" in filter_type:
        filtered = prepare_fir(
            filtered,
            sfreq,
            low_hz,
            high_hz,
        )

    else:
        raise ValueError(
            f"Unknown filter type: {filter_type}"
        )

    return filtered


def calculate_band_power(
    trial_data: np.ndarray,
    sfreq: float,
) -> dict:
    """
    Calculate 5-band power for all 64 EEG channels.

    Returns:
        320 features
        64 channels × 5 bands
    """

    n_channels, n_samples = trial_data.shape

    if n_samples < N_FFT:
        n_fft = n_samples
        n_overlap = min(
            N_OVERLAP,
            n_fft // 2,
        )
    else:
        n_fft = N_FFT
        n_overlap = N_OVERLAP

    freqs, psd = signal.welch(
        trial_data,
        fs=sfreq,
        window="hann",
        nperseg=n_fft,
        noverlap=n_overlap,
        axis=-1,
        scaling="density",
    )

    features = {}

    for channel_idx, channel_name in enumerate(EEG_CHANNELS):

        channel_psd = psd[channel_idx]

        for band_name, (low_hz, high_hz) in BANDS.items():

            mask = (
                (freqs >= low_hz)
                & (freqs <= high_hz)
            )

            if not np.any(mask):
                power = np.nan
            else:
                power = np.trapezoid(
                    channel_psd[mask],
                    freqs[mask],
                )

            feature_name = (
                f"{channel_name}_{band_name}_power"
            )

            features[feature_name] = power

    return features


def select_ten_trials(
    task_events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select exactly 10 trials deterministically.

    We take the first 10 task trials so that the benchmark
    is completely reproducible.
    """

    selected = task_events.iloc[:N_TRIALS].copy()

    selected["benchmark_order"] = np.arange(
        1,
        len(selected) + 1,
    )

    return selected


def load_continuous_context(
    raw: mne.io.BaseRaw,
    start_sample: int,
    stop_sample: int,
) -> tuple[np.ndarray, int, int]:
    """
    Read one continuous segment with context.

    Returns:
        eeg_data
        actual_start_sample
        actual_stop_sample
    """

    sfreq = raw.info["sfreq"]

    context_samples = int(
        CONTEXT_SECONDS * sfreq
    )

    actual_start = max(
        0,
        start_sample - context_samples,
    )

    actual_stop = min(
        raw.n_times,
        stop_sample + context_samples,
    )

    data = raw.get_data(
        picks=EEG_CHANNELS,
        start=actual_start,
        stop=actual_stop,
    )

    return (
        data,
        actual_start,
        actual_stop,
    )


def extract_trial_from_filtered_segment(
    filtered_segment: np.ndarray,
    segment_start: int,
    trial_start: int,
    trial_stop: int,
) -> np.ndarray:
    """
    Extract the exact 3-second trial from already-filtered
    continuous data.
    """

    relative_start = trial_start - segment_start
    relative_stop = trial_stop - segment_start

    trial = filtered_segment[
        :,
        relative_start:relative_stop,
    ]

    return trial


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("STEP 54 — EFFICIENT FILTERED BAND-POWER EXTRACTION")
    print("=" * 80)

    print()
    print("This is a NEW implementation.")
    print("Step 53 will NOT be modified.")
    print(f"Processing exactly {N_TRIALS} trials.")
    print()

    # --------------------------------------------------------
    # Validate BDF
    # --------------------------------------------------------

    if not BDF_PATH.exists():
        raise FileNotFoundError(
            f"BDF not found:\n{BDF_PATH}"
        )

    if not EVENTS_PATH.exists():
        raise FileNotFoundError(
            f"Events file not found:\n{EVENTS_PATH}"
        )

    print(f"BDF:    {BDF_PATH}")
    print(f"Events:  {EVENTS_PATH}")
    print(f"Output:  {OUTPUT_ROOT}")
    print()

    # --------------------------------------------------------
    # Load events
    # --------------------------------------------------------

    task_events = get_task_events(
        EVENTS_PATH
    )

    selected_trials = select_ten_trials(
        task_events
    )

    print()
    print("Selected 10 benchmark trials:")
    print("-" * 80)

    for _, row in selected_trials.iterrows():
        print(
            f"{int(row['benchmark_order']):2d} | "
            f"{row['trial_type']} | "
            f"sample={int(row['sample'])}"
        )

    print()

    # --------------------------------------------------------
    # Load BDF WITHOUT preload
    # --------------------------------------------------------

    print("Opening BDF with preload=False...")

    raw = mne.io.read_raw_bdf(
        BDF_PATH,
        preload=False,
        verbose=False,
    )

    print("BDF opened successfully.")

    sfreq = raw.info["sfreq"]

    print(f"Sampling frequency: {sfreq} Hz")
    print(f"Channels in BDF: {len(raw.ch_names)}")
    print(f"Samples in BDF: {raw.n_times}")

    if not np.isclose(
        sfreq,
        EXPECTED_SFREQ,
    ):
        raise ValueError(
            f"Expected {EXPECTED_SFREQ} Hz, "
            f"found {sfreq} Hz"
        )

    # --------------------------------------------------------
    # Verify EEG channels
    # --------------------------------------------------------

    missing_channels = [
        ch
        for ch in EEG_CHANNELS
        if ch not in raw.ch_names
    ]

    if missing_channels:
        raise ValueError(
            "Missing EEG channels:\n"
            + "\n".join(missing_channels)
        )

    print(
        f"Verified {len(EEG_CHANNELS)} EEG channels."
    )

    # --------------------------------------------------------
    # Determine one continuous segment containing
    # all 10 trials.
    # --------------------------------------------------------

    first_trial_sample = int(
        selected_trials.iloc[0]["sample"]
    )

    last_trial_sample = int(
        selected_trials.iloc[-1]["sample"]
    )

    trial_samples = int(
        TRIAL_DURATION_SECONDS * sfreq
    )

    first_start = first_trial_sample

    last_stop = (
        last_trial_sample
        + trial_samples
    )

    context_samples = int(
        CONTEXT_SECONDS * sfreq
    )

    segment_start = max(
        0,
        first_start - context_samples,
    )

    segment_stop = min(
        raw.n_times,
        last_stop + context_samples,
    )

    print()
    print("=" * 80)
    print("CONTINUOUS SEGMENT")
    print("=" * 80)

    print(
        f"Segment start sample: {segment_start}"
    )

    print(
        f"Segment stop sample:  {segment_stop}"
    )

    print(
        f"Segment duration: "
        f"{(segment_stop - segment_start) / sfreq:.3f} sec"
    )

    # --------------------------------------------------------
    # Read the continuous segment ONCE
    # --------------------------------------------------------

    print()
    print("Reading continuous EEG segment once...")

    read_start = time.perf_counter()

    continuous_data = raw.get_data(
        picks=EEG_CHANNELS,
        start=segment_start,
        stop=segment_stop,
    )

    read_time = (
        time.perf_counter()
        - read_start
    )

    print(
        f"Continuous data shape: "
        f"{continuous_data.shape}"
    )

    print(
        f"Read time: {read_time:.3f} sec"
    )

    # --------------------------------------------------------
    # Store all outputs
    # --------------------------------------------------------

    all_results = []

    timing_results = []

    # --------------------------------------------------------
    # Process each filter condition
    # --------------------------------------------------------

    for condition_index, condition in enumerate(
        FILTER_CONDITIONS,
        start=1,
    ):

        condition_name = condition["condition"]

        print()
        print("=" * 80)
        print(
            f"CONDITION {condition_index}/"
            f"{len(FILTER_CONDITIONS)}"
        )
        print(condition_name)
        print("=" * 80)

        print(
            f"Type: {condition['filter_type']}"
        )

        print(
            f"Band: {condition['band']}"
        )

        print(
            f"Range: "
            f"{condition['low_hz']}–"
            f"{condition['high_hz']} Hz"
        )

        # ----------------------------------------------------
        # Filter continuous data ONCE
        # ----------------------------------------------------

        filter_start = time.perf_counter()

        filtered_continuous = (
            filter_continuous_data(
                continuous_data,
                sfreq,
                condition,
            )
        )

        filter_time = (
            time.perf_counter()
            - filter_start
        )

        print(
            f"Continuous filtering time: "
            f"{filter_time:.3f} sec"
        )

        # ----------------------------------------------------
        # Validate filtered data
        # ----------------------------------------------------

        if np.isnan(
            filtered_continuous
        ).any():

            raise ValueError(
                f"NaN detected after filtering: "
                f"{condition_name}"
            )

        if np.isinf(
            filtered_continuous
        ).any():

            raise ValueError(
                f"Inf detected after filtering: "
                f"{condition_name}"
            )

        # ----------------------------------------------------
        # Extract the 10 trials from the SAME filtered
        # continuous signal
        # ----------------------------------------------------

        condition_start = time.perf_counter()

        for _, row in selected_trials.iterrows():

            trial_number = int(
                row["benchmark_order"]
            )

            trial_start = int(
                row["sample"]
            )

            trial_stop = (
                trial_start
                + trial_samples
            )

            trial_data = (
                extract_trial_from_filtered_segment(
                    filtered_continuous,
                    segment_start,
                    trial_start,
                    trial_stop,
                )
            )

            expected_shape = (
                len(EEG_CHANNELS),
                trial_samples,
            )

            if trial_data.shape != expected_shape:

                raise ValueError(
                    f"Unexpected trial shape "
                    f"for {condition_name}, "
                    f"trial {trial_number}: "
                    f"{trial_data.shape}; "
                    f"expected {expected_shape}"
                )

            feature_start = time.perf_counter()

            features = calculate_band_power(
                trial_data,
                sfreq,
            )

            feature_time = (
                time.perf_counter()
                - feature_start
            )

            # ------------------------------------------------
            # Metadata
            # ------------------------------------------------

            result = {
                "subject": SUBJECT,
                "benchmark_trial": trial_number,
                "original_task_trial_index": int(
                    row.name
                ),
                "label": int(
                    task_events[
                        task_events.index == row.name
                    ]["trial_type"].iloc[0]
                    .split("_")[0] == "tool"
                ),
                "label_name": row["trial_type"],
                "event_row": int(row.name),
                "start_sample": trial_start,
                "duration_seconds": (
                    trial_samples / sfreq
                ),
                "condition": condition_name,
                "filter_type": condition[
                    "filter_type"
                ],
                "filter_band": condition[
                    "band"
                ],
                "low_hz": condition[
                    "low_hz"
                ],
                "high_hz": condition[
                    "high_hz"
                ],
            }

            result.update(features)

            all_results.append(result)

            print(
                f"  Trial {trial_number:2d}/"
                f"{N_TRIALS} | "
                f"features={len(features)} | "
                f"PSD={feature_time:.4f}s"
            )

        condition_time = (
            time.perf_counter()
            - condition_start
        )

        timing_results.append(
            {
                "condition": condition_name,
                "filter_type": condition[
                    "filter_type"
                ],
                "band": condition["band"],
                "low_hz": condition["low_hz"],
                "high_hz": condition["high_hz"],
                "filter_time_seconds": filter_time,
                "feature_extraction_time_seconds": (
                    condition_time
                    - filter_time
                ),
                "total_condition_time_seconds": (
                    condition_time
                ),
                "trials_processed": N_TRIALS,
            }
        )

        print(
            f"Condition total time: "
            f"{condition_time:.3f} sec"
        )

        # Free filtered array before next condition
        del filtered_continuous

    # --------------------------------------------------------
    # Build final dataframe
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("BUILDING OUTPUT")
    print("=" * 80)

    results_df = pd.DataFrame(
        all_results
    )

    timing_df = pd.DataFrame(
        timing_results
    )

    # --------------------------------------------------------
    # Identify feature columns
    # --------------------------------------------------------

    metadata_columns = [
        "subject",
        "benchmark_trial",
        "original_task_trial_index",
        "label",
        "label_name",
        "event_row",
        "start_sample",
        "duration_seconds",
        "condition",
        "filter_type",
        "filter_band",
        "low_hz",
        "high_hz",
    ]

    feature_columns = [
        col
        for col in results_df.columns
        if col not in metadata_columns
    ]

    print(
        f"Result shape: {results_df.shape}"
    )

    print(
        f"Number of feature columns: "
        f"{len(feature_columns)}"
    )

    # --------------------------------------------------------
    # Validate feature count
    # --------------------------------------------------------

    expected_features = (
        64 * 5
    )

    if len(feature_columns) != expected_features:

        raise ValueError(
            f"Expected {expected_features} "
            f"features, found "
            f"{len(feature_columns)}"
        )

    # --------------------------------------------------------
    # NaN / Inf validation
    # --------------------------------------------------------

    feature_values = results_df[
        feature_columns
    ].to_numpy()

    nan_count = int(
        np.isnan(feature_values).sum()
    )

    inf_count = int(
        np.isinf(feature_values).sum()
    )

    print(
        f"NaN values: {nan_count}"
    )

    print(
        f"Inf values: {inf_count}"
    )

    if nan_count != 0:
        raise ValueError(
            "NaN values found in final features."
        )

    if inf_count != 0:
        raise ValueError(
            "Inf values found in final features."
        )

    # --------------------------------------------------------
    # Expected rows
    # --------------------------------------------------------

    expected_rows = (
        N_TRIALS
        * len(FILTER_CONDITIONS)
    )

    if len(results_df) != expected_rows:

        raise ValueError(
            f"Expected {expected_rows} rows, "
            f"found {len(results_df)}"
        )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    output_csv = (
        OUTPUT_ROOT
        / "step54_10trial_filtered_band_power.csv"
    )

    timing_csv = (
        OUTPUT_ROOT
        / "step54_10trial_timing.csv"
    )

    condition_csv = (
        OUTPUT_ROOT
        / "step54_filter_conditions.csv"
    )

    info_json = (
        OUTPUT_ROOT
        / "step54_experiment_info.json"
    )

    results_df.to_csv(
        output_csv,
        index=False,
    )

    timing_df.to_csv(
        timing_csv,
        index=False,
    )

    pd.DataFrame(
        FILTER_CONDITIONS
    ).to_csv(
        condition_csv,
        index=False,
    )

    experiment_info = {
        "step": 54,
        "subject": SUBJECT,
        "trials_processed": N_TRIALS,
        "filter_conditions": len(
            FILTER_CONDITIONS
        ),
        "expected_output_rows": expected_rows,
        "actual_output_rows": len(
            results_df
        ),
        "eeg_channels": len(
            EEG_CHANNELS
        ),
        "features_per_trial": expected_features,
        "sampling_frequency_hz": sfreq,
        "trial_duration_seconds": (
            TRIAL_DURATION_SECONDS
        ),
        "context_seconds": CONTEXT_SECONDS,
        "welch_n_fft": N_FFT,
        "welch_n_overlap": N_OVERLAP,
        "bands": BANDS,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "original_bdf_modified": False,
        "step53_modified": False,
        "implementation": (
            "continuous_segment_filter_then_trial_extract"
        ),
    }

    with open(
        info_json,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            experiment_info,
            f,
            indent=4,
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("STEP 54 COMPLETE")
    print("=" * 80)

    print(
        f"Trials processed: "
        f"{N_TRIALS}"
    )

    print(
        f"Filter conditions: "
        f"{len(FILTER_CONDITIONS)}"
    )

    print(
        f"Expected rows: "
        f"{expected_rows}"
    )

    print(
        f"Actual rows: "
        f"{len(results_df)}"
    )

    print(
        f"Features per row: "
        f"{len(feature_columns)}"
    )

    print(
        f"NaN: {nan_count}"
    )

    print(
        f"Inf: {inf_count}"
    )

    print()
    print("Saved:")
    print(output_csv)
    print(timing_csv)
    print(condition_csv)
    print(info_json)

    print()
    print("=" * 80)
    print("TIMING SUMMARY")
    print("=" * 80)

    print(
        timing_df[
            [
                "condition",
                "filter_time_seconds",
                "feature_extraction_time_seconds",
                "total_condition_time_seconds",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("Original BDF was NOT modified.")
    print("Step 53 was NOT modified.")
    print("No ML was performed.")


if __name__ == "__main__":
    main()