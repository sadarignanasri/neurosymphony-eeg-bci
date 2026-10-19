"""
STEP 52 — SUB-01 FILTER COMPARISON VISUALIZATION

Purpose
-------
Visualize raw EEG against different filtering strategies
using the same 8 representative trials from Step 51.

This step is visualization only.
NO machine learning is performed.

For each representative class:
    Raw
    Butterworth
    FIR
    Notch + Butterworth
    Notch + FIR

Frequency bands:
    Broad : 1–45 Hz
    Delta : 1–4 Hz
    Theta : 4–8 Hz
    Alpha : 8–13 Hz
    Beta  : 13–30 Hz
    Gamma : 30–45 Hz

Important
---------
Filtering is performed on a continuous context window
before extracting the 3-second trial.

The original BDF is never modified.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

from scipy.signal import butter, sosfiltfilt


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

STEP51_DIR = (
    PROJECT_ROOT
    / "results"
    / "eeg"
    / "filter_analysis"
    / "sub01"
    / "step51"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "eeg"
    / "filter_analysis"
    / "sub01"
    / "step52"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

SFREQ = 2048.0

TRIAL_SAMPLES = 6142

CONTEXT_SECONDS = 15.0

BUTTERWORTH_ORDER = 4

NOTCH_FREQUENCY = 50.0

REPRESENTATIVE_CHANNEL = "B6"

EXPECTED_EEG_CHANNELS = 64


# ============================================================
# BANDS
# ============================================================

BANDS = {
    "Broad 1–45 Hz": (1.0, 45.0),
    "Delta 1–4 Hz": (1.0, 4.0),
    "Theta 4–8 Hz": (4.0, 8.0),
    "Alpha 8–13 Hz": (8.0, 13.0),
    "Beta 13–30 Hz": (13.0, 30.0),
    "Gamma 30–45 Hz": (30.0, 45.0),
}


# ============================================================
# REPRESENTATIVE LABELS
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
# DISPLAY LABELS
# ============================================================

DISPLAY_LABELS = {
    "animal_auditory_imagery_task":
        "Animal – Auditory",

    "animal_silent_naming_task":
        "Animal – Silent",

    "animal_tactile_imagery_task":
        "Animal – Tactile",

    "animal_visual_imagery_task":
        "Animal – Visual",

    "tool_auditory_imagery_task":
        "Tool – Auditory",

    "tool_silent_naming_task":
        "Tool – Silent",

    "tool_tactile_imagery_task":
        "Tool – Tactile",

    "tool_visual_imagery_task":
        "Tool – Visual",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def design_butterworth(
    low_hz,
    high_hz,
    sfreq,
):

    nyquist = sfreq / 2.0

    low = low_hz / nyquist
    high = high_hz / nyquist

    return butter(
        BUTTERWORTH_ORDER,
        [low, high],
        btype="bandpass",
        output="sos",
    )


def apply_butterworth(
    data,
    low_hz,
    high_hz,
):

    sos = design_butterworth(
        low_hz,
        high_hz,
        SFREQ,
    )

    return sosfiltfilt(
        sos,
        data,
        axis=-1,
    )


def apply_fir(
    data,
    low_hz,
    high_hz,
):

    info = mne.create_info(
        ch_names=[
            f"EEG_{i}"
            for i in range(
                data.shape[0]
            )
        ],
        sfreq=SFREQ,
        ch_types="eeg",
    )

    temp_raw = mne.io.RawArray(
        data,
        info,
        verbose=False,
    )

    filtered = temp_raw.filter(
        l_freq=low_hz,
        h_freq=high_hz,
        method="fir",
        phase="zero",
        fir_design="firwin",
        verbose=False,
    )

    return filtered.get_data()


def apply_notch(data):

    info = mne.create_info(
        ch_names=[
            f"EEG_{i}"
            for i in range(
                data.shape[0]
            )
        ],
        sfreq=SFREQ,
        ch_types="eeg",
    )

    temp_raw = mne.io.RawArray(
        data,
        info,
        verbose=False,
    )

    filtered = temp_raw.notch_filter(
        freqs=[NOTCH_FREQUENCY],
        method="fir",
        phase="zero",
        fir_design="firwin",
        verbose=False,
    )

    return filtered.get_data()


def get_filtered_trial(
    context_data,
    local_trial_start,
    low_hz,
    high_hz,
    filter_type,
):

    working_data = context_data.copy()

    # --------------------------------------------------------
    # Notch first if required
    # --------------------------------------------------------

    if "notch" in filter_type:

        working_data = apply_notch(
            working_data
        )

    # --------------------------------------------------------
    # Band-pass
    # --------------------------------------------------------

    if filter_type == "butterworth":

        working_data = apply_butterworth(
            working_data,
            low_hz,
            high_hz,
        )

    elif filter_type == "fir":

        working_data = apply_fir(
            working_data,
            low_hz,
            high_hz,
        )

    elif filter_type == "notch+butterworth":

        working_data = apply_butterworth(
            working_data,
            low_hz,
            high_hz,
        )

    elif filter_type == "notch+fir":

        working_data = apply_fir(
            working_data,
            low_hz,
            high_hz,
        )

    # --------------------------------------------------------
    # Extract the 3-second trial
    # --------------------------------------------------------

    trial = working_data[
        :,
        local_trial_start:
        local_trial_start + TRIAL_SAMPLES,
    ]

    return trial


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "STEP 52 — SUB-01 FILTER COMPARISON VISUALIZATION"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # LOAD BDF WITHOUT LOADING WHOLE RECORDING
    # --------------------------------------------------------

    print("\nLoading original BDF...")

    raw = mne.io.read_raw_bdf(
        RAW_FILE,
        preload=False,
        verbose=False,
    )

    print(raw)

    if not np.isclose(
        raw.info["sfreq"],
        SFREQ,
    ):

        raise ValueError(
            f"Expected {SFREQ} Hz, "
            f"found {raw.info['sfreq']} Hz."
        )

    # --------------------------------------------------------
    # EEG CHANNELS
    # --------------------------------------------------------

    eeg_channels = [
        ch
        for ch in raw.ch_names
        if ch.startswith("A")
        or ch.startswith("B")
    ]

    if len(eeg_channels) != EXPECTED_EEG_CHANNELS:

        raise ValueError(
            f"Expected 64 EEG channels, "
            f"found {len(eeg_channels)}."
        )

    raw_eeg = raw.copy().pick(
        eeg_channels
    )

    b6_index = eeg_channels.index(
        REPRESENTATIVE_CHANNEL
    )

    # --------------------------------------------------------
    # EVENTS
    # --------------------------------------------------------

    events_df = pd.read_csv(
        EVENT_FILE,
        sep="\t",
    )

    task_events = events_df[
        events_df["trial_type"].isin(
            REPRESENTATIVE_LABELS
        )
    ].copy()

    if len(task_events) != 720:

        raise ValueError(
            f"Expected 720 task events, "
            f"found {len(task_events)}."
        )

    # --------------------------------------------------------
    # SELECT FIRST OCCURRENCE
    # --------------------------------------------------------

    representative_events = []

    for label in REPRESENTATIVE_LABELS:

        matching = task_events[
            task_events["trial_type"] == label
        ]

        first_event = matching.iloc[0]

        representative_events.append(
            {
                "label": label,
                "start_sample": int(
                    first_event["sample"]
                ),
            }
        )

    # --------------------------------------------------------
    # CONTEXT
    # --------------------------------------------------------

    context_samples = int(
        CONTEXT_SECONDS * SFREQ
    )

    # ========================================================
    # CREATE VISUALIZATIONS
    # ========================================================

    for trial_number, event in enumerate(
        representative_events,
        start=1,
    ):

        label = event["label"]

        start_sample = event[
            "start_sample"
        ]

        print(
            f"\nCreating figure "
            f"{trial_number}/8: "
            f"{DISPLAY_LABELS[label]}"
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

        context_data = raw_eeg.get_data(
            start=context_start,
            stop=context_stop,
            verbose=False,
        )

        local_trial_start = (
            start_sample
            - context_start
        )

        # ----------------------------------------------------
        # RAW TRIAL
        # ----------------------------------------------------

        raw_trial = context_data[
            :,
            local_trial_start:
            local_trial_start
            + TRIAL_SAMPLES,
        ]

        # ----------------------------------------------------
        # TIME AXIS
        # ----------------------------------------------------

        time = (
            np.arange(TRIAL_SAMPLES)
            / SFREQ
        )

        # ----------------------------------------------------
        # FIGURE
        # ----------------------------------------------------

        fig, axes = plt.subplots(
            nrows=6,
            ncols=1,
            figsize=(14, 16),
            sharex=True,
        )

        fig.suptitle(
            f"Sub-01 EEG Filter Comparison — "
            f"{DISPLAY_LABELS[label]} — {REPRESENTATIVE_CHANNEL}/Fz",
            fontsize=15,
        )

        # ----------------------------------------------------
        # RAW
        # ----------------------------------------------------

        axes[0].plot(
            time,
            raw_trial[b6_index] * 1e6,
            linewidth=0.8,
        )

        axes[0].set_title(
            "Raw EEG"
        )

        axes[0].set_ylabel(
            "µV"
        )

        # ----------------------------------------------------
        # FILTER GROUPS
        # ----------------------------------------------------

        filter_specs = [
            (
                "Broad 1–45 Hz",
                "butterworth",
                1.0,
                45.0,
            ),
            (
                "Broad 1–45 Hz",
                "fir",
                1.0,
                45.0,
            ),
            (
                "Alpha 8–13 Hz",
                "butterworth",
                8.0,
                13.0,
            ),
            (
                "Alpha 8–13 Hz",
                "fir",
                8.0,
                13.0,
            ),
            (
                "Beta 13–30 Hz",
                "butterworth",
                13.0,
                30.0,
            ),
        ]

        titles = [
            "Butterworth — 1–45 Hz",
            "FIR — 1–45 Hz",
            "Butterworth — Alpha 8–13 Hz",
            "FIR — Alpha 8–13 Hz",
            "Butterworth — Beta 13–30 Hz",
        ]

        for axis_index, (
            spec,
            title,
        ) in enumerate(
            zip(
                filter_specs,
                titles,
            ),
            start=1,
        ):

            _, filter_type, low_hz, high_hz = (
                spec
            )

            filtered_trial = get_filtered_trial(
                context_data,
                local_trial_start,
                low_hz,
                high_hz,
                filter_type,
            )

            axes[axis_index].plot(
                time,
                filtered_trial[
                    b6_index
                ] * 1e6,
                linewidth=0.8,
            )

            axes[axis_index].set_title(
                title
            )

            axes[axis_index].set_ylabel(
                "µV"
            )

        axes[-1].set_xlabel(
            "Time (seconds)"
        )

        for ax in axes:

            ax.grid(
                True,
                alpha=0.25,
            )

        plt.tight_layout(
            rect=[
                0,
                0,
                1,
                0.97,
            ]
        )

        output_file = (
            OUTPUT_DIR
            / (
                f"step52_"
                f"{label}_"
                f"filter_comparison.png"
            )
        )

        fig.savefig(
            output_file,
            dpi=200,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(
            f"  ✓ Saved: {output_file}"
        )

    # ========================================================
    # FINAL
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 52 COMPLETE")
    print("=" * 80)

    print(
        f"\nFigures saved in:\n{OUTPUT_DIR}"
    )

    print(
        "\n✓ Same 8 representative trials used."
    )

    print(
        "✓ Same B6/Fz channel used."
    )

    print(
        "✓ Raw and filtered signals compared."
    )

    print(
        "✓ Original BDF was not modified."
    )

    print(
        "✓ No ML was performed."
    )


if __name__ == "__main__":
    main()