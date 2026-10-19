"""
STEP 51 — SUB-01 FILTER DESIGN VALIDATION

Purpose
-------
Validate the different EEG filtering strategies on sub-01
without repeatedly processing the complete 4,350-second recording.

This is a FILTER VALIDATION step.
It does NOT perform machine learning.

Filter strategies:
    1. Raw
    2. Butterworth band-pass
    3. FIR band-pass
    4. Notch + Butterworth
    5. Notch + FIR

Frequency bands:
    Broad : 1–45 Hz
    Delta : 1–4 Hz
    Theta : 4–8 Hz
    Alpha : 8–13 Hz
    Beta  : 13–30 Hz
    Gamma : 30–45 Hz

Important
---------
Filtering is performed on a continuous window around each
representative trial, NOT directly on the isolated 3-second
trial.

This avoids the short-trial FIR problem encountered earlier.

The original BDF is never modified.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import mne

from scipy.signal import butter, sosfiltfilt, firwin


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sub-01"
    / "sub-01_task-eeg_eeg.bdf"
)

EVENT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sub-01"
    / "sub-01_task-eeg_events.tsv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "eeg"
    / "filter_analysis"
    / "sub01"
    / "step51"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

SUBJECT = "sub-01"

EXPECTED_SFREQ = 2048.0
EXPECTED_EEG_CHANNELS = 64
EXPECTED_TRIALS = 720

# Trial duration used throughout the project.
TRIAL_SAMPLES = 6142

# 3 seconds at 2048 Hz is approximately 6144 samples.
# The actual project epoch length is 6142 samples.
TRIAL_DURATION_SECONDS = (
    TRIAL_SAMPLES / EXPECTED_SFREQ
)

# Context around each representative trial.
# This allows the filter to operate on a continuous segment
# rather than a short isolated 3-second signal.
CONTEXT_SECONDS = 15.0

NOTCH_FREQUENCY = 50.0

BUTTERWORTH_ORDER = 4

# FIR length chosen for this validation experiment.
FIR_NUMTAPS = 4097


# ============================================================
# FREQUENCY BANDS
# ============================================================

BANDS = {
    "broad_1_45Hz": (1.0, 45.0),
    "delta_1_4Hz": (1.0, 4.0),
    "theta_4_8Hz": (4.0, 8.0),
    "alpha_8_13Hz": (8.0, 13.0),
    "beta_13_30Hz": (13.0, 30.0),
    "gamma_30_45Hz": (30.0, 45.0),
}


# ============================================================
# REPRESENTATIVE TRIALS
# ============================================================
#
# These are the SAME first-occurrence representative trials
# used in Steps 36–43.
#
# We use their event labels to locate the trials from the
# original events.tsv instead of hard-coding sample positions.
# ============================================================

REPRESENTATIVE_LABELS = [
    "animal_auditory_imagery_task",
    "animal_silent_naming_task",
    "animal_tactile_imagery_task",
    "animal_visual_imagery_task",
    "tool_auditory_imagery_task",
    "tool_silent_naming_task",
    "tool_tactile_imagery_task",
    "tool_visual_imagery_task",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_separator():
    print("\n" + "=" * 80)


def validate_files():

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Missing BDF file:\n{RAW_FILE}"
        )

    if not EVENT_FILE.exists():
        raise FileNotFoundError(
            f"Missing events file:\n{EVENT_FILE}"
        )


def get_eeg_channels(raw):

    eeg_channels = [
        ch
        for ch in raw.ch_names
        if ch.startswith("A")
        or ch.startswith("B")
    ]

    if len(eeg_channels) != EXPECTED_EEG_CHANNELS:

        raise ValueError(
            f"Expected {EXPECTED_EEG_CHANNELS} EEG "
            f"channels, found {len(eeg_channels)}."
        )

    return eeg_channels


def design_butterworth(
    low_hz,
    high_hz,
    sfreq,
    order=4,
):

    nyquist = sfreq / 2.0

    low = low_hz / nyquist
    high = high_hz / nyquist

    sos = butter(
        order,
        [low, high],
        btype="bandpass",
        output="sos",
    )

    return sos


def design_fir(
    low_hz,
    high_hz,
    sfreq,
):

    taps = firwin(
        numtaps=FIR_NUMTAPS,
        cutoff=[
            low_hz,
            high_hz,
        ],
        pass_zero=False,
        fs=sfreq,
    )

    return taps


def apply_butterworth(
    data,
    low_hz,
    high_hz,
    sfreq,
):

    sos = design_butterworth(
        low_hz,
        high_hz,
        sfreq,
        BUTTERWORTH_ORDER,
    )

    filtered = sosfiltfilt(
        sos,
        data,
        axis=-1,
    )

    return filtered, sos


def apply_fir(
    data,
    low_hz,
    high_hz,
    sfreq,
):

    """
    Apply zero-phase FIR filtering using MNE.

    The input is a continuous context window, not an isolated
    3-second trial.
    """

    n_channels = data.shape[0]

    info = mne.create_info(
        ch_names=[
            f"EEG_{i}"
            for i in range(n_channels)
        ],
        sfreq=sfreq,
        ch_types="eeg",
    )

    temp_raw = mne.io.RawArray(
        data,
        info,
        verbose=False,
    )

    filtered_raw = temp_raw.filter(
        l_freq=low_hz,
        h_freq=high_hz,
        method="fir",
        phase="zero",
        fir_design="firwin",
        verbose=False,
    )

    return filtered_raw.get_data()


def apply_notch(
    data,
    sfreq,
):

    """
    Apply 50 Hz zero-phase FIR notch filtering.
    """

    n_channels = data.shape[0]

    info = mne.create_info(
        ch_names=[
            f"EEG_{i}"
            for i in range(n_channels)
        ],
        sfreq=sfreq,
        ch_types="eeg",
    )

    temp_raw = mne.io.RawArray(
        data,
        info,
        verbose=False,
    )

    filtered_raw = temp_raw.notch_filter(
        freqs=[NOTCH_FREQUENCY],
        method="fir",
        phase="zero",
        fir_design="firwin",
        verbose=False,
    )

    return filtered_raw.get_data()


def calculate_statistics(signal):

    return {
        "mean": float(
            np.mean(signal)
        ),
        "std": float(
            np.std(signal)
        ),
        "rms": float(
            np.sqrt(
                np.mean(
                    signal ** 2
                )
            )
        ),
        "min": float(
            np.min(signal)
        ),
        "max": float(
            np.max(signal)
        ),
        "peak_to_peak": float(
            np.ptp(signal)
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print_separator()
    print(
        "STEP 51 — SUB-01 FILTER DESIGN VALIDATION"
    )
    print_separator()

    validate_files()

    # --------------------------------------------------------
    # LOAD BDF HEADER FIRST
    # --------------------------------------------------------

    print("Loading original BDF...")

    raw = mne.io.read_raw_bdf(
        RAW_FILE,
        preload=False,
        verbose=False,
    )

    sfreq = raw.info["sfreq"]

    print(raw)

    print(
        f"Sampling frequency : {sfreq} Hz"
    )

    print(
        f"Total channels     : {len(raw.ch_names)}"
    )

    if not np.isclose(
        sfreq,
        EXPECTED_SFREQ,
    ):

        raise ValueError(
            f"Expected {EXPECTED_SFREQ} Hz, "
            f"found {sfreq} Hz."
        )

    eeg_channels = get_eeg_channels(raw)

    print(
        f"EEG channels       : "
        f"{len(eeg_channels)}"
    )

    # --------------------------------------------------------
    # LOAD EVENTS
    # --------------------------------------------------------

    events_df = pd.read_csv(
        EVENT_FILE,
        sep="\t",
    )

    task_labels = [
        "animal_auditory_imagery_task",
        "animal_silent_naming_task",
        "animal_tactile_imagery_task",
        "animal_visual_imagery_task",
        "tool_auditory_imagery_task",
        "tool_silent_naming_task",
        "tool_tactile_imagery_task",
        "tool_visual_imagery_task",
    ]

    task_events = events_df[
        events_df["trial_type"].isin(
            task_labels
        )
    ].copy()

    print(
        f"Task trials found  : "
        f"{len(task_events)}"
    )

    if len(task_events) != EXPECTED_TRIALS:

        raise ValueError(
            f"Expected {EXPECTED_TRIALS} task trials, "
            f"found {len(task_events)}."
        )

    # --------------------------------------------------------
    # SELECT FIRST OCCURRENCE OF EACH CLASS
    # --------------------------------------------------------

    representative_events = []

    for label in REPRESENTATIVE_LABELS:

        matching = task_events[
            task_events["trial_type"] == label
        ]

        if matching.empty:

            raise ValueError(
                f"No event found for {label}"
            )

        first_event = matching.iloc[0]

        representative_events.append(
            {
                "label": label,
                "start_sample": int(
                    first_event["sample"]
                ),
                "onset_seconds": float(
                    first_event["sample"]
                ) / sfreq,
            }
        )

    representative_df = pd.DataFrame(
        representative_events
    )

    representative_df.to_csv(
        OUTPUT_DIR
        / "step51_representative_trials.csv",
        index=False,
    )

    print_separator()
    print(
        "REPRESENTATIVE TRIALS"
    )
    print_separator()

    print(
        representative_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # LOAD ONLY THE EEG CHANNELS
    # --------------------------------------------------------
    #
    # IMPORTANT:
    # preload=False means we don't load the 5.31 GiB
    # recording into memory.
    # --------------------------------------------------------

    raw_eeg = raw.copy().pick(
        eeg_channels
    )

    # --------------------------------------------------------
    # CONTEXT SIZE
    # --------------------------------------------------------

    context_samples = int(
        CONTEXT_SECONDS * sfreq
    )

    print_separator()
    print(
        "FILTERING CONFIGURATION"
    )
    print_separator()

    print(
        f"Trial duration      : "
        f"{TRIAL_DURATION_SECONDS:.6f} s"
    )

    print(
        f"Context each side   : "
        f"{CONTEXT_SECONDS:.1f} s"
    )

    print(
        f"Context samples     : "
        f"{context_samples}"
    )

    print(
        f"FIR taps            : "
        f"{FIR_NUMTAPS}"
    )

    print(
        f"Butterworth order   : "
        f"{BUTTERWORTH_ORDER}"
    )

    print(
        f"Notch frequency     : "
        f"{NOTCH_FREQUENCY} Hz"
    )

    # --------------------------------------------------------
    # FILTER CONDITIONS
    # --------------------------------------------------------

    conditions = []

    for band_name, (
        low_hz,
        high_hz,
    ) in BANDS.items():

        if band_name == "broad_1_45Hz":

            conditions.append(
                {
                    "condition":
                        "butterworth_broad_1_45Hz",
                    "filter_type":
                        "butterworth",
                    "band":
                        band_name,
                    "low_hz":
                        low_hz,
                    "high_hz":
                        high_hz,
                }
            )

            conditions.append(
                {
                    "condition":
                        "fir_broad_1_45Hz",
                    "filter_type":
                        "fir",
                    "band":
                        band_name,
                    "low_hz":
                        low_hz,
                    "high_hz":
                        high_hz,
                }
            )

            continue

        conditions.append(
            {
                "condition":
                    f"butterworth_{band_name}",
                "filter_type":
                    "butterworth",
                "band":
                    band_name,
                "low_hz":
                    low_hz,
                "high_hz":
                    high_hz,
            }
        )

        conditions.append(
            {
                "condition":
                    f"fir_{band_name}",
                "filter_type":
                    "fir",
                "band":
                    band_name,
                "low_hz":
                    low_hz,
                "high_hz":
                    high_hz,
            }
        )

        conditions.append(
            {
                "condition":
                    f"notch_butterworth_{band_name}",
                "filter_type":
                    "notch+butterworth",
                "band":
                    band_name,
                "low_hz":
                    low_hz,
                "high_hz":
                    high_hz,
            }
        )

        conditions.append(
            {
                "condition":
                    f"notch_fir_{band_name}",
                "filter_type":
                    "notch+fir",
                "band":
                    band_name,
                "low_hz":
                    low_hz,
                "high_hz":
                    high_hz,
            }
        )

    conditions_df = pd.DataFrame(
        conditions
    )

    conditions_df.to_csv(
        OUTPUT_DIR
        / "step51_filter_conditions.csv",
        index=False,
    )

    print(
        f"\nFilter conditions: "
        f"{len(conditions_df)}"
    )

    # --------------------------------------------------------
    # PROCESS EACH REPRESENTATIVE TRIAL
    # --------------------------------------------------------

    results = []

    for trial_index, trial_info in enumerate(
        representative_events,
        start=1,
    ):

        label = trial_info["label"]
        start_sample = trial_info[
            "start_sample"
        ]

        print_separator()

        print(
            f"TRIAL {trial_index}/8"
        )

        print(
            f"Label              : {label}"
        )

        print(
            f"Start sample       : "
            f"{start_sample}"
        )

        print(
            f"Start time         : "
            f"{start_sample / sfreq:.6f} s"
        )

        # ----------------------------------------------------
        # CONTEXT WINDOW
        # ----------------------------------------------------

        context_start = max(
            0,
            start_sample - context_samples,
        )

        context_stop = min(
            raw_eeg.n_times,
            start_sample
            + TRIAL_SAMPLES
            + context_samples,
        )

        print(
            f"Context samples    : "
            f"{context_start} → {context_stop}"
        )

        # ----------------------------------------------------
        # READ ONLY THIS WINDOW
        # ----------------------------------------------------

        context_data = raw_eeg.get_data(
            start=context_start,
            stop=context_stop,
            verbose=False,
        )

        if not np.isfinite(
            context_data
        ).all():

            raise ValueError(
                f"NaN/Inf found in context "
                f"for {label}."
            )

        # Index of trial inside context.
        local_trial_start = (
            start_sample
            - context_start
        )

        local_trial_stop = (
            local_trial_start
            + TRIAL_SAMPLES
        )

        # ----------------------------------------------------
        # RAW TRIAL
        # ----------------------------------------------------

        raw_trial = context_data[
            :,
            local_trial_start:
            local_trial_stop,
        ]

        if raw_trial.shape[1] != TRIAL_SAMPLES:

            raise ValueError(
                f"Unexpected trial shape "
                f"for {label}: "
                f"{raw_trial.shape}"
            )

        # ----------------------------------------------------
        # RAW STATISTICS FOR B6
        # ----------------------------------------------------

        b6_index = eeg_channels.index(
            "B6"
        )

        raw_b6 = raw_trial[
            b6_index
        ]

        stats = calculate_statistics(
            raw_b6
        )

        results.append(
            {
                "trial_index":
                    trial_index,
                "label":
                    label,
                "condition":
                    "raw",
                "filter_type":
                    "none",
                "band":
                    "unfiltered",
                "low_hz":
                    np.nan,
                "high_hz":
                    np.nan,
                **stats,
            }
        )

        # ----------------------------------------------------
        # FILTER CONDITIONS
        # ----------------------------------------------------

        for condition in conditions:

            condition_name = condition[
                "condition"
            ]

            filter_type = condition[
                "filter_type"
            ]

            low_hz = condition[
                "low_hz"
            ]

            high_hz = condition[
                "high_hz"
            ]

            print(
                f"  → {condition_name}"
            )

            # Always start from ORIGINAL context.
            working_data = context_data.copy()

            # -----------------------------------------------
            # NOTCH
            # -----------------------------------------------

            if "notch" in filter_type:

                working_data = apply_notch(
                    working_data,
                    sfreq,
                )

            # -----------------------------------------------
            # BAND-PASS
            # -----------------------------------------------

            if (
                "butterworth"
                in filter_type
            ):

                working_data, _ = (
                    apply_butterworth(
                        working_data,
                        low_hz,
                        high_hz,
                        sfreq,
                    )
                )

            elif filter_type in [
                "fir",
                "notch+fir",
            ]:

                working_data = apply_fir(
                    working_data,
                    low_hz,
                    high_hz,
                    sfreq,
                )

            # -----------------------------------------------
            # VALIDATE FILTERED WINDOW
            # -----------------------------------------------

            if working_data.shape != (
                context_data.shape
            ):

                raise ValueError(
                    f"{condition_name}: "
                    f"shape changed."
                )

            if not np.isfinite(
                working_data
            ).all():

                raise ValueError(
                    f"{condition_name}: "
                    f"NaN/Inf detected."
                )

            # -----------------------------------------------
            # EXTRACT TRIAL AFTER FILTERING
            # -----------------------------------------------

            filtered_trial = working_data[
                :,
                local_trial_start:
                local_trial_stop,
            ]

            if filtered_trial.shape != (
                EXPECTED_EEG_CHANNELS,
                TRIAL_SAMPLES,
            ):

                raise ValueError(
                    f"{condition_name}: "
                    f"unexpected trial shape "
                    f"{filtered_trial.shape}"
                )

            # -----------------------------------------------
            # B6 STATISTICS
            # -----------------------------------------------

            filtered_b6 = filtered_trial[
                b6_index
            ]

            stats = calculate_statistics(
                filtered_b6
            )

            results.append(
                {
                    "trial_index":
                        trial_index,
                    "label":
                        label,
                    "condition":
                        condition_name,
                    "filter_type":
                        filter_type,
                    "band":
                        condition["band"],
                    "low_hz":
                        low_hz,
                    "high_hz":
                        high_hz,
                    **stats,
                }
            )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    results_file = (
        OUTPUT_DIR
        / "step51_filter_validation_results.csv"
    )

    results_df.to_csv(
        results_file,
        index=False,
    )

    # --------------------------------------------------------
    # SAVE EXPERIMENT INFORMATION
    # --------------------------------------------------------

    experiment_info = {
        "step": 51,
        "subject": SUBJECT,
        "sampling_frequency_hz":
            float(sfreq),
        "eeg_channels":
            len(eeg_channels),
        "verified_task_trials":
            len(task_events),
        "representative_trials":
            len(representative_events),
        "trial_samples":
            TRIAL_SAMPLES,
        "trial_duration_seconds":
            TRIAL_DURATION_SECONDS,
        "context_seconds_each_side":
            CONTEXT_SECONDS,
        "butterworth_order":
            BUTTERWORTH_ORDER,
        "fir_numtaps":
            FIR_NUMTAPS,
        "notch_frequency_hz":
            NOTCH_FREQUENCY,
        "filtering_strategy":
            "filter continuous context window, then extract trial",
        "machine_learning":
            False,
        "original_bdf_modified":
            False,
    }

    with open(
        OUTPUT_DIR
        / "step51_experiment_info.json",
        "w",
    ) as f:

        json.dump(
            experiment_info,
            f,
            indent=4,
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print_separator()
    print(
        "STEP 51 COMPLETE"
    )
    print_separator()

    print(
        f"Representative trials processed : "
        f"{len(representative_events)}"
    )

    print(
        f"Filter conditions tested         : "
        f"{len(conditions_df)}"
    )

    print(
        f"Total result rows                : "
        f"{len(results_df)}"
    )

    print(
        "\nOutputs:"
    )

    print(
        f"  {OUTPUT_DIR}"
    )

    print(
        "\n✓ Original BDF was not modified."
    )

    print(
        "✓ 720 task trials were verified."
    )

    print(
        "✓ 8 representative trials were used."
    )

    print(
        "✓ Filtering was performed on continuous "
        "context windows."
    )

    print(
        "✓ Trials were extracted after filtering."
    )

    print(
        "✓ 64 EEG channels were processed."
    )

    print(
        "✓ No ML was performed."
    )

    print(
        "\nNext: inspect the filter-validation results "
        "before moving to visualization."
    )


if __name__ == "__main__":
    main()