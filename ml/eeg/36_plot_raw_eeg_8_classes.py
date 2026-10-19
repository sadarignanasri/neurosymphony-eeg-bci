
"""
================================================================================
NEURO-SYMPHONY — STEP 36
RAW EEG VISUALIZATION — 8 TASK CLASSES
================================================================================

Purpose
-------
Visualize one representative RAW EEG trial from each of the 8 task classes.

This experiment uses the ORIGINAL BDF recording before the project's
1–40 Hz preprocessing pipeline.

Experimental setup
------------------
Subject       : sub-01
Channel       : Fz
Classes       : 8
Trials/class  : 90
Samples/trial : 6142
Duration      : 2.999023 seconds
Sampling rate : 2048 Hz

Important
---------
- No filtering is performed.
- No FFT is performed.
- No machine-learning model is trained.
- No train/validation/test split is performed.
- The same EEG channel is used for all classes.
- One representative trial is selected from each class.
- The purpose is visual comparison before filtering.
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

SUBJECT = "sub-01"

SAMPLING_RATE = 2048.0

TRIAL_SAMPLES = 6142

TRIAL_DURATION = TRIAL_SAMPLES / SAMPLING_RATE

CHANNEL_NAME = "Fz"

RAW_ROOT = PROJECT_ROOT / "data" / "raw"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "eeg"
    / "band_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


BDF_PATH = (
    RAW_ROOT
    / SUBJECT
    / f"{SUBJECT}_task-eeg_eeg.bdf"
)

EVENTS_PATH = (
    RAW_ROOT
    / SUBJECT
    / f"{SUBJECT}_task-eeg_events.tsv"
)


# =============================================================================
# EXPECTED CLASSES
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


# Short names for plotting
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
print("NEURO-SYMPHONY — STEP 36")
print("RAW EEG VISUALIZATION — 8 TASK CLASSES")
print("=" * 90)

print()
print(f"Subject       : {SUBJECT}")
print(f"Channel       : {CHANNEL_NAME}")
print(f"Sampling rate : {SAMPLING_RATE} Hz")
print(f"Samples/trial : {TRIAL_SAMPLES}")
print(f"Duration      : {TRIAL_DURATION:.6f} seconds")

print()
print(f"BDF:")
print(BDF_PATH)

print()
print(f"Events:")
print(EVENTS_PATH)

print()
print("No filtering will be applied.")
print("No FFT will be applied.")
print("No model will be trained.")


# =============================================================================
# CHECK FILES
# =============================================================================

if not BDF_PATH.exists():
    raise FileNotFoundError(
        f"BDF file not found:\n{BDF_PATH}"
    )

if not EVENTS_PATH.exists():
    raise FileNotFoundError(
        f"Events file not found:\n{EVENTS_PATH}"
    )


# =============================================================================
# LOAD EVENTS
# =============================================================================

print()
print("=" * 90)
print("LOADING EVENTS")
print("=" * 90)

events_df = pd.read_csv(
    EVENTS_PATH,
    sep="\t"
)

print()
print(f"Total event rows: {len(events_df)}")

if "trial_type" not in events_df.columns:
    raise ValueError(
        "Expected 'trial_type' column was not found."
    )

if "sample" not in events_df.columns:
    raise ValueError(
        "Expected 'sample' column was not found."
    )


# Keep only the 8 task classes
task_events = events_df[
    events_df["trial_type"].isin(CLASS_ORDER)
].copy()

task_events = task_events.reset_index(
    drop=True
)

print(f"Task trials found: {len(task_events)}")


# =============================================================================
# VALIDATE CLASS COUNTS
# =============================================================================

print()
print("Class counts:")

class_counts = (
    task_events["trial_type"]
    .value_counts()
    .reindex(CLASS_ORDER)
)

for class_name, count in class_counts.items():
    print(f"  {class_name}: {count}")

if len(task_events) != 720:
    raise ValueError(
        f"Expected 720 task trials, found {len(task_events)}."
    )

if class_counts.isna().any():
    raise ValueError(
        "One or more expected classes are missing."
    )

if not (class_counts == 90).all():
    raise ValueError(
        "Expected exactly 90 trials per class."
    )

print()
print("✓ All 8 classes contain exactly 90 trials.")


# =============================================================================
# SELECT ONE REPRESENTATIVE TRIAL PER CLASS
# =============================================================================

print()
print("=" * 90)
print("SELECTING REPRESENTATIVE TRIALS")
print("=" * 90)

selected_trials = []

for class_name in CLASS_ORDER:

    class_rows = task_events[
        task_events["trial_type"] == class_name
    ]

    # Use the first trial of each class consistently.
    selected_row = class_rows.iloc[0]

    selected_trials.append(
        selected_row
    )

    print()
    print(f"Class: {class_name}")
    print(f"  Trial index : {selected_row.name}")
    print(f"  Sample      : {int(selected_row['sample'])}")
    print(f"  Onset       : {float(selected_row['onset']):.6f} sec")


# =============================================================================
# LOAD ORIGINAL BDF
# =============================================================================

print()
print("=" * 90)
print("LOADING ORIGINAL BDF")
print("=" * 90)

print()
print("Loading BDF header/data access...")
print("The full recording will NOT be loaded into normal RAM.")

raw = mne.io.read_raw_bdf(
    BDF_PATH,
    preload=False,
    verbose=True
)

print()
print("Original recording:")
print(raw)

print()
print(f"Channels        : {len(raw.ch_names)}")
print(f"Sampling rate   : {raw.info['sfreq']} Hz")
print(f"Duration        : {raw.times[-1]:.3f} seconds")


# =============================================================================
# CHECK SAMPLING RATE
# =============================================================================

if not np.isclose(
    raw.info["sfreq"],
    SAMPLING_RATE
):
    raise ValueError(
        f"Expected sampling rate {SAMPLING_RATE}, "
        f"found {raw.info['sfreq']}."
    )

print()
print("✓ Sampling rate verified.")


# =============================================================================
# FIND FZ CHANNEL
# =============================================================================

print()
print("=" * 90)
print("CHECKING EEG CHANNEL")
print("=" * 90)

# Original BDF uses BioSemi A/B channel names.
# In the project montage, B6 corresponds to Fz.

RAW_CHANNEL_NAME = "B6"

if RAW_CHANNEL_NAME not in raw.ch_names:
    raise ValueError(
        f"Original BDF channel {RAW_CHANNEL_NAME} "
        f"was not found."
    )

print()
print(f"Original BDF channel : {RAW_CHANNEL_NAME}")
print(f"Standard EEG name    : {CHANNEL_NAME}")

print()
print(
    "✓ Using the same EEG channel for all 8 classes."
)


# =============================================================================
# EXTRACT SIGNALS
# =============================================================================

print()
print("=" * 90)
print("EXTRACTING RAW EEG SIGNALS")
print("=" * 90)

signals = {}

time_axis = (
    np.arange(TRIAL_SAMPLES)
    / SAMPLING_RATE
)

for class_name, selected_row in zip(
    CLASS_ORDER,
    selected_trials
):

    start_sample = int(
        selected_row["sample"]
    )

    stop_sample = (
        start_sample
        + TRIAL_SAMPLES
    )

    # Check recording boundary
    if stop_sample > raw.n_times:
        raise ValueError(
            f"{class_name}: trial extends beyond "
            "the end of the recording."
        )

    print()
    print(f"Extracting: {class_name}")
    print(
        f"  Samples: "
        f"{start_sample} → {stop_sample - 1}"
    )

    # Load only this small segment from the BDF.
    data = raw.get_data(
        picks=[RAW_CHANNEL_NAME],
        start=start_sample,
        stop=stop_sample
    )

    signal = data[0].astype(
        np.float64
    )

    if len(signal) != TRIAL_SAMPLES:
        raise ValueError(
            f"{class_name}: expected "
            f"{TRIAL_SAMPLES} samples, "
            f"found {len(signal)}."
        )

    if np.isnan(signal).any():
        raise ValueError(
            f"{class_name}: NaN values found."
        )

    if np.isinf(signal).any():
        raise ValueError(
            f"{class_name}: infinite values found."
        )

    signals[class_name] = signal

    print(
        f"  ✓ Extracted {len(signal)} samples"
    )

    print(
        f"  Min: {signal.min():.8e}"
    )

    print(
        f"  Max: {signal.max():.8e}"
    )

    print(
        f"  Mean: {signal.mean():.8e}"
    )

    print(
        f"  Std: {signal.std():.8e}"
    )


# =============================================================================
# PLOT
# =============================================================================

print()
print("=" * 90)
print("CREATING RAW 8-CLASS FIGURE")
print("=" * 90)

fig, axes = plt.subplots(
    nrows=8,
    ncols=1,
    figsize=(15, 20),
    sharex=True
)

for ax, class_name in zip(
    axes,
    CLASS_ORDER
):

    signal = signals[class_name]

    ax.plot(
        time_axis,
        signal,
        linewidth=0.7
    )

    ax.set_ylabel(
        "Amplitude (V)",
        fontsize=9
    )

    ax.set_title(
        CLASS_SHORT_NAMES[class_name],
        loc="left",
        fontsize=11,
        fontweight="bold"
    )

    ax.grid(
        True,
        alpha=0.25
    )


axes[-1].set_xlabel(
    "Time (seconds)",
    fontsize=11
)

fig.suptitle(
    "RAW EEG — One Representative Trial per Class\n"
    f"Subject: {SUBJECT} | Channel: {CHANNEL_NAME} | "
    f"Duration: {TRIAL_DURATION:.3f} s",
    fontsize=15,
    fontweight="bold"
)

fig.tight_layout(
    rect=[0, 0, 1, 0.97]
)


# =============================================================================
# SAVE FIGURE
# =============================================================================

output_path = (
    OUTPUT_DIR
    / "step36_raw_eeg_8_classes_fz.png"
)

fig.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# =============================================================================
# SAVE SELECTED TRIAL METADATA
# =============================================================================

metadata_rows = []

for class_name, selected_row in zip(
    CLASS_ORDER,
    selected_trials
):

    metadata_rows.append(
        {
            "subject": SUBJECT,
            "class": class_name,
            "channel": CHANNEL_NAME,
            "original_bdf_channel": RAW_CHANNEL_NAME,
            "trial_row_index": selected_row.name,
            "start_sample": int(
                selected_row["sample"]
            ),
            "samples": TRIAL_SAMPLES,
            "sampling_rate_hz": SAMPLING_RATE,
            "duration_seconds": TRIAL_DURATION,
            "filter": "NONE — RAW",
        }
    )


metadata_df = pd.DataFrame(
    metadata_rows
)

metadata_path = (
    OUTPUT_DIR
    / "step36_raw_eeg_8_classes_metadata.csv"
)

metadata_df.to_csv(
    metadata_path,
    index=False
)


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print()
print("=" * 90)
print("STEP 36 COMPLETED SUCCESSFULLY")
print("=" * 90)

print()
print("Experimental setup:")
print(f"  Subject       : {SUBJECT}")
print(f"  Channel       : {CHANNEL_NAME}")
print(f"  Classes       : {len(CLASS_ORDER)}")
print(f"  Trials/class  : 1 representative trial")
print(f"  Samples/trial : {TRIAL_SAMPLES}")
print(f"  Duration      : {TRIAL_DURATION:.6f} seconds")
print(f"  Sampling rate : {SAMPLING_RATE} Hz")
print("  Filter        : NONE — ORIGINAL RAW EEG")

print()
print("Output files:")

print()
print("Figure:")
print(output_path)

print()
print("Metadata:")
print(metadata_path)

print()
print("✓ Same channel used for all 8 classes.")
print("✓ Same trial duration used for all 8 classes.")
print("✓ Original BDF data used.")
print("✓ No filtering performed.")
print("✓ No FFT performed.")
print("✓ No ML performed.")

print()
print("=" * 90)