
"""
================================================================================
NEURO-SYMPHONY — STEP 37
CROSS-SUBJECT RAW EEG COMPARISON
================================================================================

Purpose
-------
Compare RAW EEG signals from sub-01 and sub-02 for the same task classes.

Controlled conditions
---------------------
Subjects       : sub-01, sub-02
Channel        : Fz
Original BDF   : Yes
Filtering      : None
FFT            : None
Samples/trial  : 6142
Sampling rate  : 2048 Hz
Duration       : 2.999023 s
Classes        : 8

The same representative-trial selection rule is used for both subjects:
the first occurrence of each class in the events.tsv file.

No preprocessing or ML is performed.
================================================================================
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne


# =============================================================================
# PROJECT ROOT
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# CONFIGURATION
# =============================================================================

SUBJECTS = [
    "sub-01",
    "sub-02",
]

CHANNEL_NAME = "Fz"

# Original BioSemi channel corresponding to Fz
RAW_CHANNEL_NAME = "B6"

SAMPLING_RATE = 2048.0

TRIAL_SAMPLES = 6142

TRIAL_DURATION = (
    TRIAL_SAMPLES / SAMPLING_RATE
)

RAW_ROOT = PROJECT_ROOT / "data" / "raw"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "eeg"
    / "band_analysis"
    / "cross_subject"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# CLASS ORDER
# =============================================================================

CLASS_ORDER = [
    "animal_auditory_imagery_task",
    "animal_silent_naming_task",
    "animal_tactile_imagery_task",
    "animal_visual_imagery_task",
    "tool_auditory_imagery_task",
    "tool_silent_naming_task",
    "tool_tactile_imagery_task",
    "tool_visual_imagery_task",
]


CLASS_SHORT_NAMES = {
    "animal_auditory_imagery_task": "Animal - Auditory",
    "animal_silent_naming_task": "Animal - Silent",
    "animal_tactile_imagery_task": "Animal - Tactile",
    "animal_visual_imagery_task": "Animal - Visual",
    "tool_auditory_imagery_task": "Tool - Auditory",
    "tool_silent_naming_task": "Tool - Silent",
    "tool_tactile_imagery_task": "Tool - Tactile",
    "tool_visual_imagery_task": "Tool - Visual",
}


# =============================================================================
# HEADER
# =============================================================================

print("=" * 90)
print("NEURO-SYMPHONY — STEP 37")
print("CROSS-SUBJECT RAW EEG COMPARISON")
print("=" * 90)

print()
print(f"Subjects       : {', '.join(SUBJECTS)}")
print(f"Channel        : {CHANNEL_NAME}")
print(f"Raw BDF channel: {RAW_CHANNEL_NAME}")
print(f"Sampling rate  : {SAMPLING_RATE} Hz")
print(f"Samples/trial  : {TRIAL_SAMPLES}")
print(f"Duration       : {TRIAL_DURATION:.6f} seconds")
print("Filter         : NONE")
print("FFT            : NONE")


# =============================================================================
# FUNCTION: LOAD ONE SUBJECT
# =============================================================================

def load_subject_raw(subject):
    """
    Load one subject's original BDF without preloading the complete recording.
    """

    bdf_path = (
        RAW_ROOT
        / subject
        / f"{subject}_task-eeg_eeg.bdf"
    )

    events_path = (
        RAW_ROOT
        / subject
        / f"{subject}_task-eeg_events.tsv"
    )

    if not bdf_path.exists():
        raise FileNotFoundError(
            f"BDF not found:\n{bdf_path}"
        )

    if not events_path.exists():
        raise FileNotFoundError(
            f"Events file not found:\n{events_path}"
        )

    print()
    print("=" * 90)
    print(f"LOADING {subject}")
    print("=" * 90)

    # -------------------------------------------------------------------------
    # Events
    # -------------------------------------------------------------------------

    events_df = pd.read_csv(
        events_path,
        sep="\t"
    )

    task_events = events_df[
        events_df["trial_type"].isin(
            CLASS_ORDER
        )
    ].copy()

    task_events = task_events.reset_index(
        drop=True
    )

    if len(task_events) != 720:
        raise ValueError(
            f"{subject}: expected 720 task trials, "
            f"found {len(task_events)}."
        )

    class_counts = (
        task_events["trial_type"]
        .value_counts()
        .reindex(CLASS_ORDER)
    )

    if class_counts.isna().any():
        raise ValueError(
            f"{subject}: one or more classes are missing."
        )

    if not (class_counts == 90).all():
        raise ValueError(
            f"{subject}: expected 90 trials per class."
        )

    print()
    print(f"✓ {subject}: 720 task trials")
    print("✓ All 8 classes contain 90 trials")

    # -------------------------------------------------------------------------
    # BDF
    # -------------------------------------------------------------------------

    raw = mne.io.read_raw_bdf(
        bdf_path,
        preload=False,
        verbose=True
    )

    print()
    print(f"{subject} recording:")
    print(raw)

    if not np.isclose(
        raw.info["sfreq"],
        SAMPLING_RATE
    ):
        raise ValueError(
            f"{subject}: unexpected sampling rate "
            f"{raw.info['sfreq']}"
        )

    if RAW_CHANNEL_NAME not in raw.ch_names:
        raise ValueError(
            f"{subject}: original channel "
            f"{RAW_CHANNEL_NAME} not found."
        )

    print()
    print(
        f"✓ {subject}: {RAW_CHANNEL_NAME} "
        f"available for Fz"
    )

    return raw, task_events


# =============================================================================
# FUNCTION: EXTRACT REPRESENTATIVE TRIALS
# =============================================================================

def extract_subject_signals(
    subject,
    raw,
    task_events
):
    """
    Extract the first trial of each class.
    """

    signals = {}

    metadata = []

    print()
    print("=" * 90)
    print(f"EXTRACTING {subject}")
    print("=" * 90)

    for class_name in CLASS_ORDER:

        class_rows = task_events[
            task_events["trial_type"] == class_name
        ]

        selected_row = class_rows.iloc[0]

        start_sample = int(
            selected_row["sample"]
        )

        stop_sample = (
            start_sample
            + TRIAL_SAMPLES
        )

        if stop_sample > raw.n_times:
            raise ValueError(
                f"{subject} {class_name}: "
                "trial extends beyond recording."
            )

        signal_data = raw.get_data(
            picks=[RAW_CHANNEL_NAME],
            start=start_sample,
            stop=stop_sample
        )

        signal = signal_data[0].astype(
            np.float64
        )

        if len(signal) != TRIAL_SAMPLES:
            raise ValueError(
                f"{subject} {class_name}: "
                f"expected {TRIAL_SAMPLES} samples, "
                f"found {len(signal)}."
            )

        if np.isnan(signal).any():
            raise ValueError(
                f"{subject} {class_name}: NaN found."
            )

        if np.isinf(signal).any():
            raise ValueError(
                f"{subject} {class_name}: Inf found."
            )

        signals[class_name] = signal

        metadata.append(
            {
                "subject": subject,
                "class": class_name,
                "channel": CHANNEL_NAME,
                "original_bdf_channel": RAW_CHANNEL_NAME,
                "trial_row_index": selected_row.name,
                "start_sample": start_sample,
                "samples": TRIAL_SAMPLES,
                "sampling_rate_hz": SAMPLING_RATE,
                "duration_seconds": TRIAL_DURATION,
                "filter": "NONE — RAW",
            }
        )

        print()
        print(f"{class_name}")
        print(
            f"  Start sample : {start_sample}"
        )
        print(
            f"  Onset        : "
            f"{float(selected_row['onset']):.6f} sec"
        )
        print(
            f"  Mean         : "
            f"{signal.mean():.8e}"
        )
        print(
            f"  Std          : "
            f"{signal.std():.8e}"
        )

    return signals, metadata


# =============================================================================
# LOAD BOTH SUBJECTS
# =============================================================================

subject_raw = {}
subject_events = {}

for subject in SUBJECTS:

    raw, events = load_subject_raw(
        subject
    )

    subject_raw[subject] = raw
    subject_events[subject] = events


# =============================================================================
# EXTRACT SIGNALS
# =============================================================================

subject_signals = {}

all_metadata = []

for subject in SUBJECTS:

    signals, metadata = extract_subject_signals(
        subject,
        subject_raw[subject],
        subject_events[subject]
    )

    subject_signals[subject] = signals

    all_metadata.extend(
        metadata
    )


# =============================================================================
# TIME AXIS
# =============================================================================

time_axis = (
    np.arange(TRIAL_SAMPLES)
    / SAMPLING_RATE
)


# =============================================================================
# CREATE ONE FIGURE PER CLASS
# =============================================================================

print()
print("=" * 90)
print("CREATING CROSS-SUBJECT COMPARISON PLOTS")
print("=" * 90)

for class_name in CLASS_ORDER:

    fig, ax = plt.subplots(
        figsize=(15, 6)
    )

    for subject in SUBJECTS:

        signal = (
            subject_signals[subject][class_name]
        )

        ax.plot(
            time_axis,
            signal,
            linewidth=0.8,
            label=subject
        )

    ax.set_title(
        f"{CLASS_SHORT_NAMES[class_name]} — RAW EEG\n"
        f"Channel: {CHANNEL_NAME} | "
        f"sub-01 vs sub-02",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Time (seconds)"
    )

    ax.set_ylabel(
        "Amplitude (V)"
    )

    ax.grid(
        True,
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / (
            f"step37_{class_name}"
            f"_sub01_vs_sub02_raw.png"
        )
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print()
    print(
        f"✓ Saved: {output_path}"
    )


# =============================================================================
# SAVE METADATA
# =============================================================================

metadata_df = pd.DataFrame(
    all_metadata
)

metadata_path = (
    OUTPUT_DIR
    / "step37_cross_subject_metadata.csv"
)

metadata_df.to_csv(
    metadata_path,
    index=False
)


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

summary_rows = []

for class_name in CLASS_ORDER:

    row = {
        "class": class_name
    }

    for subject in SUBJECTS:

        signal = (
            subject_signals[subject][class_name]
        )

        row[
            f"{subject}_mean"
        ] = signal.mean()

        row[
            f"{subject}_std"
        ] = signal.std()

        row[
            f"{subject}_min"
        ] = signal.min()

        row[
            f"{subject}_max"
        ] = signal.max()

    summary_rows.append(row)


summary_df = pd.DataFrame(
    summary_rows
)

summary_path = (
    OUTPUT_DIR
    / "step37_cross_subject_raw_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)


# =============================================================================
# CLEANUP
# =============================================================================

for subject in SUBJECTS:

    subject_raw[subject].close()


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print()
print("=" * 90)
print("STEP 37 COMPLETED SUCCESSFULLY")
print("=" * 90)

print()
print("Comparison:")
print("  sub-01 vs sub-02")
print(f"  Channel       : {CHANNEL_NAME}")
print(f"  Classes       : {len(CLASS_ORDER)}")
print(f"  Samples/trial : {TRIAL_SAMPLES}")
print(f"  Duration      : {TRIAL_DURATION:.6f} sec")
print(f"  Filter        : NONE")
print(f"  FFT           : NONE")

print()
print("Outputs:")
print(
    f"  Cross-subject plots : {OUTPUT_DIR}"
)

print()
print(
    f"  Metadata : {metadata_path}"
)

print(
    f"  Summary  : {summary_path}"
)

print()
print("✓ Same channel used for both subjects.")
print("✓ Same trial-selection rule used.")
print("✓ Same trial duration used.")
print("✓ Original BDF data used.")
print("✓ No filtering performed.")
print("✓ No FFT performed.")
print("✓ No ML performed.")

print()
print("=" * 90)