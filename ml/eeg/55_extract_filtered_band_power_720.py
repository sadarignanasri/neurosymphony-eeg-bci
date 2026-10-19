"""
Step 55 — Efficient 720-Trial Filtered Band-Power Extraction
==============================================================

Purpose
-------
Scale the validated Step 54 continuous-filtering approach from
10 benchmark trials to all 720 task trials for sub-01.

IMPORTANT
---------
Step 53 is NOT modified.
Step 54 is NOT modified.
Original BDF is NOT modified.

Approach
--------
Instead of filtering every 3-second trial separately:

    read trial
    filter trial
    calculate PSD
    repeat...

we process manageable continuous chunks:

    read continuous chunk + context
        ↓
    filter continuous chunk once
        ↓
    extract all trials inside the chunk
        ↓
    calculate band power
        ↓
    save results

This reduces repeated filtering while avoiding loading the
entire BDF into memory.

Output
------
One CSV per filter condition.

Each completed condition contains:

    720 trials
    64 EEG channels
    5 frequency bands
    320 band-power features

8-class labels are explicitly encoded.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy import signal


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = PROJECT_ROOT / "data" / "raw"

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "eeg"
    / "filter_analysis"
    / "sub01"
    / "step55"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# DATASET CONFIGURATION
# ============================================================

SUBJECT = "sub-01"

BDF_PATH = (
    DATA_ROOT
    / SUBJECT
    / f"{SUBJECT}_task-eeg_eeg.bdf"
)

EVENTS_PATH = (
    DATA_ROOT
    / SUBJECT
    / f"{SUBJECT}_task-eeg_events.tsv"
)

EXPECTED_SFREQ = 2048.0

# ------------------------------------------------------------
# Trial configuration
# ------------------------------------------------------------

TRIAL_DURATION_SECONDS = 3.0

TRIAL_SAMPLES = int(
    TRIAL_DURATION_SECONDS
    * EXPECTED_SFREQ
)

# ------------------------------------------------------------
# Chunk configuration
# ------------------------------------------------------------

CORE_CHUNK_SECONDS = 120.0

CONTEXT_SECONDS = 15.0

CORE_CHUNK_SAMPLES = int(
    CORE_CHUNK_SECONDS
    * EXPECTED_SFREQ
)

CONTEXT_SAMPLES = int(
    CONTEXT_SECONDS
    * EXPECTED_SFREQ
)


# ============================================================
# EEG CHANNELS
# ============================================================

EEG_CHANNELS = (
    [f"A{i}" for i in range(1, 33)]
    + [f"B{i}" for i in range(1, 33)]
)


# ============================================================
# 8-CLASS LABEL MAPPING
# ============================================================

LABEL_MAP = {
    "animal_tactile_imagery_task": 0,
    "animal_visual_imagery_task": 1,
    "animal_auditory_imagery_task": 2,
    "animal_silent_naming_task": 3,

    "tool_tactile_imagery_task": 4,
    "tool_visual_imagery_task": 5,
    "tool_auditory_imagery_task": 6,
    "tool_silent_naming_task": 7,
}


# ============================================================
# FREQUENCY BANDS
# ============================================================

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 40.0),
}


# ============================================================
# WELCH PARAMETERS
# ============================================================

N_FFT = 4096
N_OVERLAP = 2048


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
# EVENT LOADING
# ============================================================

def load_task_events(
    events_path: Path,
) -> pd.DataFrame:

    events = pd.read_csv(
        events_path,
        sep="\t",
    )

    required = {
        "trial_type",
        "sample",
    }

    missing = required - set(
        events.columns
    )

    if missing:
        raise ValueError(
            f"Missing event columns: {missing}"
        )

    task_events = events[
        events["trial_type"]
        .astype(str)
        .str.endswith("_task")
    ].copy()

    task_events = task_events.reset_index(
        drop=True
    )

    if len(task_events) != 720:
        raise ValueError(
            f"Expected 720 task trials, "
            f"found {len(task_events)}"
        )

    # --------------------------------------------------------
    # Validate all labels
    # --------------------------------------------------------

    unknown_labels = sorted(
        set(task_events["trial_type"])
        - set(LABEL_MAP)
    )

    if unknown_labels:
        raise ValueError(
            "Unknown task labels found:\n"
            + "\n".join(unknown_labels)
        )

    task_events["label"] = (
        task_events["trial_type"]
        .map(LABEL_MAP)
        .astype(int)
    )

    task_events["trial_index"] = np.arange(
        len(task_events)
    )

    return task_events


# ============================================================
# FILTER FUNCTIONS
# ============================================================

def butterworth_filter(
    data: np.ndarray,
    sfreq: float,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:

    nyquist = sfreq / 2.0

    sos = signal.butter(
        N=4,
        Wn=[
            low_hz / nyquist,
            high_hz / nyquist,
        ],
        btype="bandpass",
        output="sos",
    )

    return signal.sosfiltfilt(
        sos,
        data,
        axis=-1,
    )


def fir_filter(
    data: np.ndarray,
    sfreq: float,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:

    nyquist = sfreq / 2.0

    taps = signal.firwin(
        numtaps=4097,
        cutoff=[
            low_hz / nyquist,
            high_hz / nyquist,
        ],
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


def notch_filter(
    data: np.ndarray,
    sfreq: float,
) -> np.ndarray:

    nyquist = sfreq / 2.0

    sos = signal.butter(
        N=4,
        Wn=[
            49.0 / nyquist,
            51.0 / nyquist,
        ],
        btype="bandstop",
        output="sos",
    )

    return signal.sosfiltfilt(
        sos,
        data,
        axis=-1,
    )


def apply_filter(
    data: np.ndarray,
    sfreq: float,
    condition: dict,
) -> np.ndarray:

    filter_type = condition[
        "filter_type"
    ]

    low_hz = condition[
        "low_hz"
    ]

    high_hz = condition[
        "high_hz"
    ]

    filtered = data

    # --------------------------------------------------------
    # Notch first
    # --------------------------------------------------------

    if filter_type.startswith(
        "notch+"
    ):

        filtered = notch_filter(
            filtered,
            sfreq,
        )

    # --------------------------------------------------------
    # Main filter
    # --------------------------------------------------------

    if "butterworth" in filter_type:

        filtered = butterworth_filter(
            filtered,
            sfreq,
            low_hz,
            high_hz,
        )

    elif "fir" in filter_type:

        filtered = fir_filter(
            filtered,
            sfreq,
            low_hz,
            high_hz,
        )

    else:

        raise ValueError(
            f"Unknown filter type: "
            f"{filter_type}"
        )

    return filtered


# ============================================================
# BAND POWER
# ============================================================

def calculate_band_power(
    trial_data: np.ndarray,
    sfreq: float,
) -> dict:

    n_samples = trial_data.shape[-1]

    n_fft = min(
        N_FFT,
        n_samples,
    )

    n_overlap = min(
        N_OVERLAP,
        n_fft // 2,
    )

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

    for channel_idx, channel_name in enumerate(
        EEG_CHANNELS
    ):

        channel_psd = psd[
            channel_idx
        ]

        for band_name, (
            low_hz,
            high_hz,
        ) in BANDS.items():

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

            features[
                f"{channel_name}_{band_name}_power"
            ] = power

    return features


# ============================================================
# CHUNK GENERATION
# ============================================================

def generate_chunks(
    raw: mne.io.BaseRaw,
):

    total_samples = raw.n_times

    core_start = 0

    chunk_id = 0

    while core_start < total_samples:

        core_stop = min(
            core_start
            + CORE_CHUNK_SAMPLES,
            total_samples,
        )

        read_start = max(
            0,
            core_start
            - CONTEXT_SAMPLES,
        )

        read_stop = min(
            total_samples,
            core_stop
            + CONTEXT_SAMPLES,
        )

        yield {
            "chunk_id": chunk_id,
            "core_start": core_start,
            "core_stop": core_stop,
            "read_start": read_start,
            "read_stop": read_stop,
        }

        core_start = core_stop
        chunk_id += 1


# ============================================================
# TRIALS BELONGING TO CHUNK
# ============================================================

def get_trials_for_chunk(
    task_events: pd.DataFrame,
    core_start: int,
    core_stop: int,
) -> pd.DataFrame:

    # Trial onset must belong to the core.
    mask = (
        (task_events["sample"] >= core_start)
        &
        (task_events["sample"] < core_stop)
    )

    return task_events.loc[
        mask
    ].copy()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "STEP 55 — EFFICIENT 720-TRIAL "
        "FILTERED BAND-POWER EXTRACTION"
    )
    print("=" * 80)

    print()
    print("Step 53: UNTOUCHED")
    print("Step 54: UNTOUCHED")
    print("Original BDF: UNTOUCHED")
    print()

    # --------------------------------------------------------
    # Validate files
    # --------------------------------------------------------

    if not BDF_PATH.exists():

        raise FileNotFoundError(
            f"BDF not found:\n{BDF_PATH}"
        )

    if not EVENTS_PATH.exists():

        raise FileNotFoundError(
            f"Events file not found:\n"
            f"{EVENTS_PATH}"
        )

    # --------------------------------------------------------
    # Events
    # --------------------------------------------------------

    task_events = load_task_events(
        EVENTS_PATH
    )

    print(
        f"Task trials: {len(task_events)}"
    )

    print(
        f"Classes: "
        f"{task_events['label'].nunique()}"
    )

    print()
    print("Class distribution:")

    print(
        task_events[
            ["label", "trial_type"]
        ]
        .drop_duplicates()
        .sort_values("label")
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Open BDF
    # --------------------------------------------------------

    print()
    print(
        "Opening BDF with preload=False..."
    )

    raw = mne.io.read_raw_bdf(
        BDF_PATH,
        preload=False,
        verbose=False,
    )

    sfreq = raw.info["sfreq"]

    print(
        f"Sampling frequency: {sfreq} Hz"
    )

    print(
        f"Total BDF samples: "
        f"{raw.n_times}"
    )

    print(
        f"Total duration: "
        f"{raw.n_times / sfreq / 60:.2f} minutes"
    )

    if not np.isclose(
        sfreq,
        EXPECTED_SFREQ,
    ):

        raise ValueError(
            f"Expected {EXPECTED_SFREQ} Hz, "
            f"found {sfreq} Hz"
        )

    # --------------------------------------------------------
    # Channel validation
    # --------------------------------------------------------

    missing = [
        ch
        for ch in EEG_CHANNELS
        if ch not in raw.ch_names
    ]

    if missing:

        raise ValueError(
            "Missing EEG channels:\n"
            + "\n".join(missing)
        )

    print(
        f"Verified {len(EEG_CHANNELS)} EEG channels."
    )

    # --------------------------------------------------------
    # Generate chunk list
    # --------------------------------------------------------

    chunks = list(
        generate_chunks(raw)
    )

    print(
        f"Core chunk size: "
        f"{CORE_CHUNK_SECONDS} seconds"
    )

    print(
        f"Context: "
        f"{CONTEXT_SECONDS} seconds each side"
    )

    print(
        f"Total chunks: {len(chunks)}"
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata_columns = [
        "subject",
        "trial_index",
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

    expected_features = (
        64 * 5
    )

    # --------------------------------------------------------
    # Process each filter condition
    # --------------------------------------------------------

    overall_start = time.perf_counter()

    completed_conditions = []

    for condition_index, condition in enumerate(
        FILTER_CONDITIONS,
        start=1,
    ):

        condition_name = condition[
            "condition"
        ]

        output_csv = (
            OUTPUT_ROOT
            / f"step55_{condition_name}.csv"
        )

        timing_csv = (
            OUTPUT_ROOT
            / f"step55_{condition_name}_timing.csv"
        )

        # ----------------------------------------------------
        # Skip only if an already complete file exists
        # ----------------------------------------------------

        if output_csv.exists():

            try:

                existing = pd.read_csv(
                    output_csv,
                    usecols=[
                        "trial_index"
                    ],
                )

                if len(existing) == 720:

                    print()
                    print(
                        f"[{condition_index}/"
                        f"{len(FILTER_CONDITIONS)}] "
                        f"{condition_name}"
                    )

                    print(
                        "Complete output already "
                        "exists — skipping."
                    )

                    completed_conditions.append(
                        condition_name
                    )

                    continue

            except Exception:

                pass

        print()
        print("=" * 80)

        print(
            f"CONDITION "
            f"{condition_index}/"
            f"{len(FILTER_CONDITIONS)}"
        )

        print(
            condition_name
        )

        print("=" * 80)

        print(
            f"Type: "
            f"{condition['filter_type']}"
        )

        print(
            f"Band: "
            f"{condition['band']}"
        )

        condition_start = (
            time.perf_counter()
        )

        all_results = []
        timing_rows = []

        # ----------------------------------------------------
        # Process chunks
        # ----------------------------------------------------

        for chunk in chunks:

            chunk_id = chunk[
                "chunk_id"
            ]

            core_start = chunk[
                "core_start"
            ]

            core_stop = chunk[
                "core_stop"
            ]

            read_start = chunk[
                "read_start"
            ]

            read_stop = chunk[
                "read_stop"
            ]

            chunk_trials = (
                get_trials_for_chunk(
                    task_events,
                    core_start,
                    core_stop,
                )
            )

            # No task trials in this chunk
            if chunk_trials.empty:
                continue

            print()
            print(
                f"Chunk {chunk_id + 1}/"
                f"{len(chunks)} | "
                f"samples "
                f"{core_start}–"
                f"{core_stop} | "
                f"trials="
                f"{len(chunk_trials)}"
            )

            # ------------------------------------------------
            # Read continuous EEG chunk
            # ------------------------------------------------

            read_timer = (
                time.perf_counter()
            )

            continuous_data = raw.get_data(
                picks=EEG_CHANNELS,
                start=read_start,
                stop=read_stop,
            )

            read_time = (
                time.perf_counter()
                - read_timer
            )

            # ------------------------------------------------
            # Filter continuous chunk ONCE
            # ------------------------------------------------

            filter_timer = (
                time.perf_counter()
            )

            filtered_data = (
                apply_filter(
                    continuous_data,
                    sfreq,
                    condition,
                )
            )

            filter_time = (
                time.perf_counter()
                - filter_timer
            )

            # ------------------------------------------------
            # Validate filtering
            # ------------------------------------------------

            if np.isnan(
                filtered_data
            ).any():

                raise ValueError(
                    f"NaN after filtering "
                    f"in condition "
                    f"{condition_name}, "
                    f"chunk {chunk_id}"
                )

            if np.isinf(
                filtered_data
            ).any():

                raise ValueError(
                    f"Inf after filtering "
                    f"in condition "
                    f"{condition_name}, "
                    f"chunk {chunk_id}"
                )

            # ------------------------------------------------
            # Extract trials
            # ------------------------------------------------

            feature_timer = (
                time.perf_counter()
            )

            for _, row in (
                chunk_trials.iterrows()
            ):

                trial_index = int(
                    row["trial_index"]
                )

                trial_start = int(
                    row["sample"]
                )

                trial_stop = (
                    trial_start
                    + TRIAL_SAMPLES
                )

                # Relative to chunk read start
                relative_start = (
                    trial_start
                    - read_start
                )

                relative_stop = (
                    trial_stop
                    - read_start
                )

                trial_data = (
                    filtered_data[
                        :,
                        relative_start:
                        relative_stop,
                    ]
                )

                expected_shape = (
                    64,
                    TRIAL_SAMPLES,
                )

                if (
                    trial_data.shape
                    != expected_shape
                ):

                    raise ValueError(
                        f"Unexpected trial "
                        f"shape: "
                        f"{trial_data.shape}; "
                        f"expected "
                        f"{expected_shape}; "
                        f"trial="
                        f"{trial_index}"
                    )

                features = (
                    calculate_band_power(
                        trial_data,
                        sfreq,
                    )
                )

                result = {
                    "subject": SUBJECT,
                    "trial_index": trial_index,
                    "label": int(
                        row["label"]
                    ),
                    "label_name": row[
                        "trial_type"
                    ],
                    "event_row": trial_index,
                    "start_sample": trial_start,
                    "duration_seconds": (
                        TRIAL_SAMPLES
                        / sfreq
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

                result.update(
                    features
                )

                all_results.append(
                    result
                )

            feature_time = (
                time.perf_counter()
                - feature_timer
            )

            timing_rows.append(
                {
                    "condition":
                        condition_name,
                    "chunk_id":
                        chunk_id,
                    "trials_in_chunk":
                        len(chunk_trials),
                    "read_time_seconds":
                        read_time,
                    "filter_time_seconds":
                        filter_time,
                    "feature_time_seconds":
                        feature_time,
                    "chunk_total_seconds":
                        read_time
                        + filter_time
                        + feature_time,
                }
            )

            print(
                f"  read={read_time:.3f}s | "
                f"filter={filter_time:.3f}s | "
                f"features={feature_time:.3f}s"
            )

            # Free memory before next chunk
            del continuous_data
            del filtered_data

            gc.collect()

        # ----------------------------------------------------
        # Build dataframe
        # ----------------------------------------------------

        results_df = pd.DataFrame(
            all_results
        )

        timing_df = pd.DataFrame(
            timing_rows
        )

        # ----------------------------------------------------
        # Validate row count
        # ----------------------------------------------------

        if len(results_df) != 720:

            raise ValueError(
                f"Condition "
                f"{condition_name} produced "
                f"{len(results_df)} rows; "
                f"expected 720."
            )

        # ----------------------------------------------------
        # Validate trial uniqueness
        # ----------------------------------------------------

        if (
            results_df[
                "trial_index"
            ].nunique()
            != 720
        ):

            raise ValueError(
                f"Duplicate or missing "
                f"trial indices in "
                f"{condition_name}"
            )

        # ----------------------------------------------------
        # Validate class count
        # ----------------------------------------------------

        class_counts = (
            results_df[
                "label"
            ]
            .value_counts()
            .sort_index()
        )

        if not all(
            class_counts.get(
                class_id,
                0,
            )
            == 90
            for class_id in range(8)
        ):

            raise ValueError(
                "Class distribution "
                "is not 90 per class:\n"
                f"{class_counts}"
            )

        # ----------------------------------------------------
        # Feature columns
        # ----------------------------------------------------

        feature_columns = [
            col
            for col in results_df.columns
            if col not in metadata_columns
        ]

        if len(feature_columns) != (
            expected_features
        ):

            raise ValueError(
                f"Expected "
                f"{expected_features} "
                f"features, found "
                f"{len(feature_columns)}"
            )

        # ----------------------------------------------------
        # NaN / Inf
        # ----------------------------------------------------

        feature_values = (
            results_df[
                feature_columns
            ].to_numpy()
        )

        nan_count = int(
            np.isnan(
                feature_values
            ).sum()
        )

        inf_count = int(
            np.isinf(
                feature_values
            ).sum()
        )

        if nan_count != 0:

            raise ValueError(
                f"NaN count = "
                f"{nan_count}"
            )

        if inf_count != 0:

            raise ValueError(
                f"Inf count = "
                f"{inf_count}"
            )

        # ----------------------------------------------------
        # Sort by original trial order
        # ----------------------------------------------------

        results_df = (
            results_df
            .sort_values(
                "trial_index"
            )
            .reset_index(
                drop=True
            )
        )

        # ----------------------------------------------------
        # Save condition
        # ----------------------------------------------------

        results_df.to_csv(
            output_csv,
            index=False,
        )

        timing_df.to_csv(
            timing_csv,
            index=False,
        )

        condition_total = (
            time.perf_counter()
            - condition_start
        )

        print()
        print(
            f"Completed: "
            f"{condition_name}"
        )

        print(
            f"Rows: "
            f"{len(results_df)}"
        )

        print(
            f"Features: "
            f"{len(feature_columns)}"
        )

        print(
            f"NaN: {nan_count}"
        )

        print(
            f"Inf: {inf_count}"
        )

        print(
            f"Condition time: "
            f"{condition_total / 60:.2f} min"
        )

        print(
            f"Saved: "
            f"{output_csv}"
        )

        completed_conditions.append(
            condition_name
        )

        del results_df
        del timing_df
        del all_results
        del timing_rows

        gc.collect()

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    total_time = (
        time.perf_counter()
        - overall_start
    )

    experiment_info = {
        "step": 55,
        "subject": SUBJECT,
        "task_trials": 720,
        "classes": 8,
        "trials_per_class": 90,
        "eeg_channels": 64,
        "features_per_trial": 320,
        "filter_conditions": len(
            FILTER_CONDITIONS
        ),
        "completed_conditions": len(
            completed_conditions
        ),
        "core_chunk_seconds":
            CORE_CHUNK_SECONDS,
        "context_seconds":
            CONTEXT_SECONDS,
        "sampling_frequency_hz":
            sfreq,
        "trial_duration_seconds":
            TRIAL_DURATION_SECONDS,
        "welch_n_fft":
            N_FFT,
        "welch_n_overlap":
            N_OVERLAP,
        "original_bdf_modified":
            False,
        "step53_modified":
            False,
        "step54_modified":
            False,
        "total_runtime_seconds":
            total_time,
    }

    info_path = (
        OUTPUT_ROOT
        / "step55_experiment_info.json"
    )

    with open(
        info_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            experiment_info,
            f,
            indent=4,
        )

    # --------------------------------------------------------
    # Condition list
    # --------------------------------------------------------

    conditions_path = (
        OUTPUT_ROOT
        / "step55_filter_conditions.csv"
    )

    pd.DataFrame(
        FILTER_CONDITIONS
    ).to_csv(
        conditions_path,
        index=False,
    )

    print()
    print("=" * 80)
    print("STEP 55 COMPLETE")
    print("=" * 80)

    print(
        f"Completed conditions: "
        f"{len(completed_conditions)}/"
        f"{len(FILTER_CONDITIONS)}"
    )

    print(
        f"Total runtime: "
        f"{total_time / 60:.2f} minutes"
    )

    print()
    print(
        "Every completed condition should contain:"
    )

    print(
        "  720 trials"
    )

    print(
        "  8 classes × 90 trials"
    )

    print(
        "  64 EEG channels"
    )

    print(
        "  320 band-power features"
    )

    print()
    print(
        "Original BDF was NOT modified."
    )

    print(
        "Step 53 was NOT modified."
    )

    print(
        "Step 54 was NOT modified."
    )


if __name__ == "__main__":
    main()