
"""
Neuro-Symphony
Step 30: Extract temporal EEG epochs from original sub-01 BDF

Source:
    Original DS004514 sub-01 BDF

Output:
    720 labelled temporal EEG epochs

Shape:
    (720, 64, 6142)

6142 samples corresponds to:
    6142 / 2048 = 2.999023 seconds

This script performs extraction only.
No model training is performed.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import mne


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SUBJECT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sub-01"
)

BDF_FILE = SUBJECT_DIR / "sub-01_task-eeg_eeg.bdf"

EVENTS_TSV = (
    SUBJECT_DIR
    / "sub-01_task-eeg_events.tsv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "temporal_eeg_sub01"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

SFREQ = 2048.0

N_SAMPLES = 6142

EEG_CHANNELS = [
    f"A{i}" for i in range(1, 33)
] + [
    f"B{i}" for i in range(1, 33)
]

TASK_LABELS = [
    "animal_tactile_imagery_task",
    "animal_visual_imagery_task",
    "animal_auditory_imagery_task",
    "animal_silent_naming_task",
    "tool_tactile_imagery_task",
    "tool_visual_imagery_task",
    "tool_auditory_imagery_task",
    "tool_silent_naming_task",
]


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
# START
# ============================================================

print("=" * 80)
print("NEURO-SYMPHONY — SUB-01 TEMPORAL EEG EXTRACTION")
print("=" * 80)

print("\nBDF:")
print(BDF_FILE)

print("\nOutput directory:")
print(OUTPUT_DIR)


# ============================================================
# LOAD EVENTS
# ============================================================

print("\n" + "=" * 80)
print("LOADING EVENTS")
print("=" * 80)

events_df = pd.read_csv(
    EVENTS_TSV,
    sep="\t"
)

events_df = events_df.sort_values(
    "sample"
).reset_index(drop=True)


# Select only the 8 labelled task events
task_events = events_df[
    events_df["trial_type"].isin(TASK_LABELS)
].copy()

task_events = task_events.reset_index(drop=True)


print(f"\nTask trials found: {len(task_events)}")


if len(task_events) != 720:
    raise RuntimeError(
        f"Expected 720 trials but found {len(task_events)}"
    )


# ============================================================
# LOAD ORIGINAL BDF
# ============================================================

print("\n" + "=" * 80)
print("LOADING ORIGINAL BDF")
print("=" * 80)

raw = mne.io.read_raw_bdf(
    BDF_FILE,
    preload=True,
    verbose=True
)

print("\nRaw information:")
print(raw)


# ============================================================
# VERIFY SAMPLING RATE
# ============================================================

if raw.info["sfreq"] != SFREQ:
    raise RuntimeError(
        f"Expected {SFREQ} Hz but found "
        f"{raw.info['sfreq']} Hz"
    )


# ============================================================
# VERIFY EEG CHANNELS
# ============================================================

missing_channels = [
    ch for ch in EEG_CHANNELS
    if ch not in raw.ch_names
]

if missing_channels:
    raise RuntimeError(
        f"Missing EEG channels: {missing_channels}"
    )

print(
    f"\n✓ All {len(EEG_CHANNELS)} EEG channels found."
)


# ============================================================
# PICK 64 EEG CHANNELS
# ============================================================

raw_eeg = raw.copy().pick(
    EEG_CHANNELS
)

print("\nSelected EEG channels:")
print(len(raw_eeg.ch_names))


# ============================================================
# EXTRACT EPOCHS
# ============================================================

print("\n" + "=" * 80)
print("EXTRACTING TEMPORAL EEG")
print("=" * 80)

epochs = []

metadata_rows = []

for trial_index, row in task_events.iterrows():

    label_name = row["trial_type"]

    start_sample = int(row["sample"])

    end_sample = start_sample + N_SAMPLES

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if end_sample > raw_eeg.n_times:

        raise RuntimeError(
            f"Trial {trial_index} exceeds recording length."
        )

    # --------------------------------------------------------
    # Extract
    # --------------------------------------------------------

    data = raw_eeg.get_data(
        start=start_sample,
        stop=end_sample
    )

    # Expected shape:
    # (64, 6142)

    if data.shape != (64, N_SAMPLES):

        raise RuntimeError(
            f"Unexpected shape for trial "
            f"{trial_index}: {data.shape}"
        )

    epochs.append(
        data.astype(np.float32)
    )

    metadata_rows.append(
        {
            "subject_id": "sub-01",
            "trial_id": trial_index,
            "start_sample": start_sample,
            "end_sample": end_sample,
            "start_time_sec": start_sample / SFREQ,
            "duration_sec": N_SAMPLES / SFREQ,
            "label_name": label_name,
            "label": LABEL_MAP[label_name],
        }
    )

    if (trial_index + 1) % 50 == 0:
        print(
            f"Extracted {trial_index + 1}/720 trials"
        )


# ============================================================
# STACK
# ============================================================

X = np.stack(epochs)

metadata = pd.DataFrame(metadata_rows)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("EXTRACTED DATA VALIDATION")
print("=" * 80)

print("\nShape:")
print(X.shape)

print(
    f"\nExpected shape: "
    f"(720, 64, {N_SAMPLES})"
)

print(
    f"\nData type: {X.dtype}"
)

print(
    f"\nMinimum value: {np.min(X):.6f}"
)

print(
    f"Maximum value: {np.max(X):.6f}"
)

print(
    f"\nNaN count: {np.isnan(X).sum()}"
)

print(
    f"Inf count: {np.isinf(X).sum()}"
)


# ============================================================
# CLASS COUNTS
# ============================================================

print("\nClass counts:")

print(
    metadata["label_name"].value_counts().sort_index()
)


# ============================================================
# SAVE
# ============================================================

X_FILE = OUTPUT_DIR / "sub-01_temporal_eeg.npy"

METADATA_FILE = (
    OUTPUT_DIR
    / "sub-01_temporal_eeg_metadata.csv"
)

CHANNEL_FILE = (
    OUTPUT_DIR
    / "channel_names.txt"
)


print("\n" + "=" * 80)
print("SAVING")
print("=" * 80)

np.save(
    X_FILE,
    X
)

metadata.to_csv(
    METADATA_FILE,
    index=False
)

with open(
    CHANNEL_FILE,
    "w",
    encoding="utf-8"
) as f:

    for channel in raw_eeg.ch_names:
        f.write(channel + "\n")


print(f"\n✓ EEG data saved:")
print(X_FILE)

print(f"\n✓ Metadata saved:")
print(METADATA_FILE)

print(f"\n✓ Channel names saved:")
print(CHANNEL_FILE)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 80)
print("EXTRACTION COMPLETE")
print("=" * 80)

print(
    "\nFinal temporal EEG shape:"
)

print(
    f"Trials  : {X.shape[0]}"
)

print(
    f"Channels: {X.shape[1]}"
)

print(
    f"Samples : {X.shape[2]}"
)

print(
    f"Duration: {X.shape[2] / SFREQ:.6f} seconds"
)

print("\nNo model was trained.")
print("No FFT was performed.")
print("No test split was performed.")