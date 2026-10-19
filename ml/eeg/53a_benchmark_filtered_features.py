"""
Step 53A — FIR Runtime Benchmark

Neuro-Symphony
EEG Filter Comparison Experiment

Purpose
-------
Benchmark the FIR 1–45 Hz filtering pipeline on only 10 trials.

This is NOT the full Step 53 experiment.

The purpose is to measure:
    1. FIR filtering runtime
    2. Band-power extraction runtime
    3. Feature dimensions
    4. Numerical validity

Only after this benchmark will we decide whether the
full FIR experiment should be executed.

Experiment:
    Subject       : sub-01
    Trials        : 10
    EEG channels  : 64
    Filter        : FIR 1–45 Hz
    FIR taps      : 4097
    Context       : ±15 seconds
    Sampling rate : 2048 Hz

No ML is performed.
No files are modified.
The original BDF is never modified.
"""

from pathlib import Path
import time

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


# =============================================================================
# EXPERIMENT SETTINGS
# =============================================================================

FS = 2048.0

TEST_TRIALS = 10

TRIAL_SAMPLES = 6142

CONTEXT_SECONDS = 15.0

CONTEXT_SAMPLES = int(
    CONTEXT_SECONDS * FS
)

FIR_NUMTAPS = 4097

LOW_HZ = 1.0

HIGH_HZ = 45.0


# =============================================================================
# EEG CHANNELS
# =============================================================================

EEG_CHANNELS = (
    [f"A{i}" for i in range(1, 33)]
    +
    [f"B{i}" for i in range(1, 33)]
)


# =============================================================================
# FREQUENCY BANDS
# =============================================================================

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 40.0),
}


# =============================================================================
# FIR FILTER
# =============================================================================

def fir_filter(data):
    """
    Apply a zero-phase FIR band-pass filter.

    Parameters
    ----------
    data : ndarray
        Shape:
            (64, n_samples)

    Returns
    -------
    ndarray
        Same shape as input.
    """

    taps = signal.firwin(
        FIR_NUMTAPS,
        [LOW_HZ, HIGH_HZ],
        pass_zero=False,
        fs=FS,
        window="hamming",
    )

    filtered = signal.filtfilt(
        taps,
        [1.0],
        data,
        axis=-1,
        method="pad",
    )

    return filtered


# =============================================================================
# BAND-POWER EXTRACTION
# =============================================================================

def compute_band_power(trial_data):
    """
    Compute Welch band power.

    Parameters
    ----------
    trial_data : ndarray
        Shape:
            (64, 6142)

    Returns
    -------
    ndarray
        Shape:
            (64, 5)

    Meaning:
        64 EEG channels × 5 frequency bands
    """

    n_fft = min(
        4096,
        trial_data.shape[1],
    )

    n_overlap = min(
        2048,
        n_fft // 2,
    )

    freqs, psd = signal.welch(
        trial_data,
        fs=FS,
        nperseg=n_fft,
        noverlap=n_overlap,
        nfft=n_fft,
        axis=-1,
        scaling="density",
    )

    band_power = []

    for band_name, (low, high) in BANDS.items():

        frequency_mask = (
            (freqs >= low)
            &
            (freqs <= high)
        )

        if not np.any(frequency_mask):
            raise RuntimeError(
                f"No frequency bins found for "
                f"{band_name}."
            )

        power = np.trapezoid(
            psd[:, frequency_mask],
            freqs[frequency_mask],
            axis=-1,
        )

        band_power.append(power)

    result = np.stack(
        band_power,
        axis=1,
    )

    return result


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("STEP 53A — FIR RUNTIME BENCHMARK")
    print("=" * 80)

    print()
    print("Test configuration:")
    print(
        f"  Filter        : FIR "
        f"{LOW_HZ:g}–{HIGH_HZ:g} Hz"
    )
    print(
        f"  FIR taps      : {FIR_NUMTAPS}"
    )
    print(
        f"  Trials        : {TEST_TRIALS}"
    )
    print(
        f"  EEG channels  : {len(EEG_CHANNELS)}"
    )
    print(
        f"  Sampling rate : {FS:g} Hz"
    )
    print(
        f"  Context       : ±{CONTEXT_SECONDS:g} seconds"
    )
    print(
        f"  Trial samples : {TRIAL_SAMPLES}"
    )
    print(
        f"  Bands         : {len(BANDS)}"
    )
    print()

    # =========================================================================
    # LOAD EVENTS
    # =========================================================================

    print("Loading task events...")

    events = pd.read_csv(
        EVENTS_PATH,
        sep="\t",
    )

    required_columns = {
        "sample",
        "trial_type",
    }

    missing_columns = (
        required_columns
        -
        set(events.columns)
    )

    if missing_columns:
        raise RuntimeError(
            f"Missing event columns: "
            f"{sorted(missing_columns)}"
        )

    task_events = events[
        events["trial_type"]
        .astype(str)
        .str.endswith("_task")
    ].copy()

    task_events = task_events.reset_index(
        drop=False
    )

    if len(task_events) != 720:
        raise RuntimeError(
            f"Expected 720 task trials, "
            f"found {len(task_events)}."
        )

    print(
        f"✓ Task trials: {len(task_events)}"
    )

    # =========================================================================
    # LOAD BDF
    # =========================================================================

    print()
    print("Loading BDF metadata...")

    raw = mne.io.read_raw_bdf(
        BDF_PATH,
        preload=False,
        verbose=False,
    )

    print(raw)

    # =========================================================================
    # VALIDATE SAMPLING RATE
    # =========================================================================

    if not np.isclose(
        raw.info["sfreq"],
        FS,
    ):
        raise RuntimeError(
            f"Unexpected sampling rate: "
            f"{raw.info['sfreq']}"
        )

    # =========================================================================
    # FIND EEG CHANNELS
    # =========================================================================

    missing_eeg = [
        channel
        for channel in EEG_CHANNELS
        if channel not in raw.ch_names
    ]

    if missing_eeg:
        raise RuntimeError(
            f"Missing EEG channels: "
            f"{missing_eeg}"
        )

    eeg_indices = [
        raw.ch_names.index(channel)
        for channel in EEG_CHANNELS
    ]

    if len(eeg_indices) != 64:
        raise RuntimeError(
            f"Expected 64 EEG channels, "
            f"found {len(eeg_indices)}."
        )

    print(
        f"✓ Found {len(eeg_indices)} EEG channels"
    )

    # =========================================================================
    # START BENCHMARK
    # =========================================================================

    print()
    print("Starting FIR benchmark...")
    print()

    total_start = time.perf_counter()

    feature_shapes = []

    filter_times = []

    psd_times = []

    for trial_number in range(TEST_TRIALS):

        event = task_events.iloc[
            trial_number
        ]

        start_sample = int(
            event["sample"]
        )

        print(
            f"Processing trial "
            f"{trial_number + 1}/{TEST_TRIALS}..."
        )

        # =====================================================================
        # READ CONTINUOUS CONTEXT
        # =====================================================================

        context_start = max(
            0,
            start_sample
            - CONTEXT_SAMPLES,
        )

        context_stop = min(
            raw.n_times,
            start_sample
            + TRIAL_SAMPLES
            + CONTEXT_SAMPLES,
        )

        data = raw.get_data(
            picks=eeg_indices,
            start=context_start,
            stop=context_stop,
        )

        if data.shape[0] != 64:
            raise RuntimeError(
                f"Unexpected EEG shape: "
                f"{data.shape}"
            )

        # =====================================================================
        # FIR FILTER
        # =====================================================================

        filter_start = time.perf_counter()

        filtered = fir_filter(
            data
        )

        filter_elapsed = (
            time.perf_counter()
            - filter_start
        )

        filter_times.append(
            filter_elapsed
        )

        # =====================================================================
        # NUMERICAL VALIDATION
        # =====================================================================

        if not np.isfinite(
            filtered
        ).all():
            raise RuntimeError(
                "NaN/Inf detected after FIR filtering."
            )

        # =====================================================================
        # EXTRACT ORIGINAL TRIAL FROM FILTERED CONTEXT
        # =====================================================================

        trial_start = (
            start_sample
            - context_start
        )

        trial_stop = (
            trial_start
            + TRIAL_SAMPLES
        )

        trial = filtered[
            :,
            trial_start:trial_stop,
        ]

        if trial.shape != (
            64,
            TRIAL_SAMPLES,
        ):
            raise RuntimeError(
                f"Unexpected trial shape: "
                f"{trial.shape}"
            )

        # =====================================================================
        # BAND POWER
        # =====================================================================

        psd_start = time.perf_counter()

        band_power = compute_band_power(
            trial
        )

        psd_elapsed = (
            time.perf_counter()
            - psd_start
        )

        psd_times.append(
            psd_elapsed
        )

        # =====================================================================
        # VALIDATE FEATURES
        # =====================================================================

        if band_power.shape != (
            64,
            5,
        ):
            raise RuntimeError(
                f"Unexpected band-power shape: "
                f"{band_power.shape}"
            )

        features = band_power.reshape(
            -1
        )

        if features.shape != (
            320,
        ):
            raise RuntimeError(
                f"Unexpected feature shape: "
                f"{features.shape}"
            )

        if not np.isfinite(
            features
        ).all():
            raise RuntimeError(
                "NaN/Inf detected in features."
            )

        feature_shapes.append(
            features.shape
        )

        elapsed = (
            time.perf_counter()
            - total_start
        )

        print(
            f"  ✓ Filter: "
            f"{filter_elapsed:.3f} s"
        )

        print(
            f"  ✓ Band power: "
            f"{psd_elapsed:.3f} s"
        )

        print(
            f"  ✓ Total elapsed: "
            f"{elapsed:.3f} s"
        )

        print()

    # =========================================================================
    # FINAL TIMING
    # =========================================================================

    total_time = (
        time.perf_counter()
        - total_start
    )

    average_time = (
        total_time
        / TEST_TRIALS
    )

    average_filter_time = (
        np.mean(filter_times)
    )

    average_psd_time = (
        np.mean(psd_times)
    )

    estimated_720_time = (
        average_time
        * 720
    )

    # =========================================================================
    # RESULTS
    # =========================================================================

    print("=" * 80)
    print("STEP 53A COMPLETE")
    print("=" * 80)

    print()
    print(
        f"Trials completed          : "
        f"{TEST_TRIALS}"
    )

    print(
        f"Total benchmark time      : "
        f"{total_time:.2f} seconds"
    )

    print(
        f"Average per trial        : "
        f"{average_time:.3f} seconds"
    )

    print(
        f"Average FIR time/trial   : "
        f"{average_filter_time:.3f} seconds"
    )

    print(
        f"Average PSD time/trial   : "
        f"{average_psd_time:.3f} seconds"
    )

    print()

    print(
        f"Estimated 720-trial time : "
        f"{estimated_720_time / 60:.2f} minutes"
    )

    print()

    print(
        "Feature shape             : "
        "(320,)"
    )

    print(
        "Band-power shape          : "
        "(64, 5)"
    )

    print(
        "NaN/Inf check             : PASSED"
    )

    print()

    print(
        "Original BDF modified     : NO"
    )

    print(
        "Machine learning          : NOT PERFORMED"
    )

    print()

    print("=" * 80)
    print("IMPORTANT")
    print("=" * 80)

    print(
        "This was ONLY a 10-trial benchmark."
    )

    print(
        "Do NOT start the full experiment yet."
    )

    print(
        "Use the measured FIR runtime to decide "
        "whether the full FIR experiment is practical."
    )

    print("=" * 80)


if __name__ == "__main__":
    main()