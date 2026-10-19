"""
Step 38 — Raw EEG visualization for sub-02

Purpose:
    Generate ONE combined figure containing one representative
    raw EEG trial for each of the 8 classes.

Same visualization format as Step 36 (sub-01).

Subject:
    sub-02

Channel:
    Fz = original BioSemi channel B6

Data:
    Original BDF
    No filtering
    No FFT
    No ML
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne


# ============================================================
# CONFIGURATION
# ============================================================

SUBJECT = "sub-02"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / SUBJECT
    / f"{SUBJECT}_task-eeg_eeg.bdf"
)

EVENTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / SUBJECT
    / f"{SUBJECT}_task-eeg_events.tsv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "eeg"
    / "band_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# EEG SETTINGS
# ============================================================

CHANNEL = "B6"
STANDARD_CHANNEL_NAME = "Fz"

SFREQ = 2048
TRIAL_SAMPLES = 6142
TRIAL_DURATION = TRIAL_SAMPLES / SFREQ


# ============================================================
# ACTUAL SUB-02 DATASET LABELS
# ============================================================

CLASS_NAMES = [
    "animal_auditory_imagery_task",
    "animal_silent_naming_task",
    "animal_tactile_imagery_task",
    "animal_visual_imagery_task",
    "tool_auditory_imagery_task",
    "tool_silent_naming_task",
    "tool_tactile_imagery_task",
    "tool_visual_imagery_task",
]


# Labels shown on the figure
DISPLAY_NAMES = {
    "animal_auditory_imagery_task": "Animal - Auditory",
    "animal_silent_naming_task": "Animal - Silent",
    "animal_tactile_imagery_task": "Animal - Tactile",
    "animal_visual_imagery_task": "Animal - Visual",
    "tool_auditory_imagery_task": "Tool - Auditory",
    "tool_silent_naming_task": "Tool - Silent",
    "tool_tactile_imagery_task": "Tool - Tactile",
    "tool_visual_imagery_task": "Tool - Visual",
}


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STEP 38 — RAW EEG VISUALIZATION — SUB-02")
print("=" * 70)

print(f"Subject       : {SUBJECT}")
print(f"BDF           : {BDF_PATH}")
print(f"Events        : {EVENTS_PATH}")
print(f"Channel       : {CHANNEL} ({STANDARD_CHANNEL_NAME})")
print(f"Sampling rate : {SFREQ} Hz")
print(f"Trial samples : {TRIAL_SAMPLES}")
print(f"Duration      : {TRIAL_DURATION:.6f} seconds")
print("Filtering     : NONE")
print("FFT           : NONE")
print("ML            : NONE")
print()


# ============================================================
# CHECK FILES
# ============================================================

if not BDF_PATH.exists():
    raise FileNotFoundError(
        f"BDF file not found:\n{BDF_PATH}"
    )

if not EVENTS_PATH.exists():
    raise FileNotFoundError(
        f"Events file not found:\n{EVENTS_PATH}"
    )


# ============================================================
# LOAD EVENTS
# ============================================================

events_df = pd.read_csv(
    EVENTS_PATH,
    sep="\t"
)

print(f"Total event rows: {len(events_df)}")

print("\nEvent columns:")
print(events_df.columns.tolist())


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

if "trial_type" not in events_df.columns:
    raise ValueError(
        "'trial_type' column not found in events.tsv."
    )

if "onset" not in events_df.columns:
    raise ValueError(
        "'onset' column not found in events.tsv."
    )


# ============================================================
# SELECT THE 8 TASK CLASSES
# ============================================================

task_df = events_df[
    events_df["trial_type"].isin(CLASS_NAMES)
].copy()

task_df = task_df.reset_index(drop=True)

print(f"\nTask trials found: {len(task_df)}")


# ============================================================
# CHECK CLASS COUNTS
# ============================================================

class_counts = task_df["trial_type"].value_counts()

print("\nClass counts:")

for class_name in CLASS_NAMES:

    count = int(
        class_counts.get(class_name, 0)
    )

    print(
        f"  {class_name:<40} {count}"
    )

if len(task_df) != 720:
    raise ValueError(
        f"Expected 720 task trials, found {len(task_df)}."
    )

for class_name in CLASS_NAMES:

    count = int(
        class_counts.get(class_name, 0)
    )

    if count != 90:
        raise ValueError(
            f"{class_name}: expected 90 trials, found {count}."
        )

print("\n✓ 720 task trials confirmed")
print("✓ 8 classes confirmed")
print("✓ 90 trials per class confirmed")


# ============================================================
# LOAD ORIGINAL BDF
# ============================================================

print("\nLoading original BDF...")

raw = mne.io.read_raw_bdf(
    BDF_PATH,
    preload=False,
    verbose="ERROR"
)

print("\nOriginal BDF:")
print(raw)

print(
    f"Sampling frequency: "
    f"{raw.info['sfreq']} Hz"
)

if raw.info["sfreq"] != SFREQ:
    raise ValueError(
        f"Expected {SFREQ} Hz, "
        f"found {raw.info['sfreq']} Hz."
    )

if CHANNEL not in raw.ch_names:
    raise ValueError(
        f"Channel {CHANNEL} not found."
    )


# ============================================================
# SELECT FIRST TRIAL OF EACH CLASS
# ============================================================

selected_trials = []

print("\nSelecting first occurrence of each class...")

for class_name in CLASS_NAMES:

    class_rows = task_df[
        task_df["trial_type"] == class_name
    ]

    first_row = class_rows.iloc[0]

    onset_seconds = float(
        first_row["onset"]
    )

    start_sample = int(
        round(onset_seconds * SFREQ)
    )

    stop_sample = (
        start_sample + TRIAL_SAMPLES
    )

    if stop_sample > raw.n_times:
        raise ValueError(
            f"{class_name}: trial exceeds "
            f"recording length."
        )

    selected_trials.append(
        {
            "class": class_name,
            "display_name": DISPLAY_NAMES[class_name],
            "event_row": int(first_row.name),
            "onset_seconds": onset_seconds,
            "start_sample": start_sample,
            "stop_sample": stop_sample,
        }
    )


# ============================================================
# TIME AXIS
# ============================================================

time = (
    np.arange(TRIAL_SAMPLES)
    / SFREQ
)


# ============================================================
# CREATE ONE COMBINED FIGURE
# ============================================================

print("\nGenerating combined 8-class figure...")

fig, axes = plt.subplots(
    nrows=8,
    ncols=1,
    figsize=(14, 17),
    sharex=True
)


metadata_rows = []


# ============================================================
# EXTRACT + PLOT EACH CLASS
# ============================================================

for ax, trial in zip(
    axes,
    selected_trials
):

    class_name = trial["class"]

    start_sample = trial["start_sample"]
    stop_sample = trial["stop_sample"]

    # --------------------------------------------------------
    # Extract only B6 / Fz
    # --------------------------------------------------------

    channel_index = raw.ch_names.index(
        CHANNEL
    )

    data, _ = raw[
        [channel_index],
        start_sample:stop_sample
    ]

    signal = data[0]

    if len(signal) != TRIAL_SAMPLES:
        raise ValueError(
            f"{class_name}: expected "
            f"{TRIAL_SAMPLES} samples, "
            f"got {len(signal)}."
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    signal_min = float(
        np.min(signal)
    )

    signal_max = float(
        np.max(signal)
    )

    signal_mean = float(
        np.mean(signal)
    )

    signal_std = float(
        np.std(signal)
    )

    print(
        f"{trial['display_name']:<20} | "
        f"trial row: {trial['event_row']:>3} | "
        f"onset: {trial['onset_seconds']:.6f} s | "
        f"std: {signal_std:.6e}"
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    ax.plot(
        time,
        signal,
        linewidth=0.8
    )

    ax.set_ylabel(
        "Amplitude (V)"
    )

    ax.set_title(
        trial["display_name"],
        loc="left",
        fontsize=11,
        fontweight="bold"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    ax.set_xlim(
        0,
        TRIAL_DURATION
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata_rows.append(
        {
            "subject": SUBJECT,
            "class": class_name,
            "display_name": trial["display_name"],
            "channel_original": CHANNEL,
            "channel_standard": STANDARD_CHANNEL_NAME,
            "sampling_rate_hz": SFREQ,
            "trial_samples": TRIAL_SAMPLES,
            "duration_seconds": TRIAL_DURATION,
            "event_row": trial["event_row"],
            "onset_seconds": trial["onset_seconds"],
            "start_sample": start_sample,
            "stop_sample": stop_sample,
            "signal_min": signal_min,
            "signal_max": signal_max,
            "signal_mean": signal_mean,
            "signal_std": signal_std,
            "filter": "none",
            "fft": False,
            "ml": False
        }
    )


# ============================================================
# FIGURE TITLE
# ============================================================

fig.suptitle(
    "RAW EEG — One Representative Trial per Class\n"
    f"Subject: {SUBJECT} | "
    f"Channel: {STANDARD_CHANNEL_NAME} | "
    f"Duration: {TRIAL_DURATION:.3f} s",
    fontsize=16,
    fontweight="bold",
    y=0.995
)


# ============================================================
# X-AXIS
# ============================================================

axes[-1].set_xlabel(
    "Time (seconds)"
)


# ============================================================
# LAYOUT
# ============================================================

plt.tight_layout(
    rect=[0, 0, 1, 0.97]
)


# ============================================================
# SAVE COMBINED FIGURE
# ============================================================

output_path = (
    OUTPUT_DIR
    / "step38_raw_eeg_8_classes_fz_sub02.png"
)

plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SAVE METADATA
# ============================================================

metadata_df = pd.DataFrame(
    metadata_rows
)

metadata_path = (
    OUTPUT_DIR
    / "step38_raw_eeg_8_classes_fz_sub02_metadata.csv"
)

metadata_df.to_csv(
    metadata_path,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 38 COMPLETE")
print("=" * 70)

print(f"Subject       : {SUBJECT}")
print(f"Channel       : {STANDARD_CHANNEL_NAME} ({CHANNEL})")
print(f"Classes       : 8")
print(f"Trials/class  : 90")
print(f"Samples/trial : {TRIAL_SAMPLES}")
print(f"Duration      : {TRIAL_DURATION:.6f} seconds")

print("\nCombined figure:")
print(f"  {output_path}")

print("\nMetadata:")
print(f"  {metadata_path}")

print("\n✓ Original BDF")
print("✓ Raw EEG")
print("✓ No filtering")
print("✓ No FFT")
print("✓ No ML")
print("✓ Same visualization structure as sub-01")