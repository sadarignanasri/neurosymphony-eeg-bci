"""
Neuro-Symphony
Step 31: Preprocess continuous sub-01 EEG and extract clean temporal epochs.

Uses the project's existing production EEG preprocessing pipeline:

    80-channel BDF
        ↓
    64 BioSemi EEG channels
        ↓
    BioSemi-64 montage
        ↓
    1–40 Hz band-pass
        ↓
    50 Hz notch
        ↓
    Iz, TP8, P6 bad-channel handling
        ↓
    Spherical-spline interpolation
        ↓
    Average reference
        ↓
    Select 64 EEG channels
        ↓
    Extract 720 labelled temporal epochs

Output:
    (720, 64, 6142)

No FFT.
No model training.
No train/test split.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import mne


# ============================================================================
# PROJECT IMPORT PATH
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.preprocessing.eeg import purify_eeg


# ============================================================================
# PATHS
# ============================================================================

BDF_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sub-01"
    / "sub-01_task-eeg_eeg.bdf"
)

EVENTS_TSV = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sub-01"
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


# ============================================================================
# DATASET PARAMETERS
# ============================================================================

SFREQ = 2048.0

# All task trials have approximately 3 seconds available.
# 6142 samples gives:
#
# 6142 / 2048 = 2.9990234375 seconds
#
# This avoids crossing into the following event for the 5 trials
# that have slightly less than exactly 3 seconds available.

N_SAMPLES = 6142


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


# ============================================================================
# AUXILIARY CHANNELS
# ============================================================================

AUXILIARY_CHANNELS = [
    "EXG1",
    "EXG2",
    "EXG3",
    "EXG4",
    "EXG5",
    "EXG6",
    "EXG7",
    "EXG8",
    "GSR1",
    "GSR2",
    "Erg1",
    "Erg2",
    "Resp",
    "Plet",
    "Temp",
    "Status",
]


# ============================================================================
# START
# ============================================================================

print("=" * 80)
print("NEURO-SYMPHONY — SUB-01 TEMPORAL EEG PREPROCESSING")
print("=" * 80)

print("\nBDF:")
print(BDF_FILE)

print("\nEvents:")
print(EVENTS_TSV)

print("\nOutput:")
print(OUTPUT_DIR)


# ============================================================================
# VALIDATE INPUT FILES
# ============================================================================

if not BDF_FILE.exists():
    raise FileNotFoundError(
        f"Original BDF file not found:\n{BDF_FILE}"
    )

if not EVENTS_TSV.exists():
    raise FileNotFoundError(
        f"Events TSV file not found:\n{EVENTS_TSV}"
    )


# ============================================================================
# LOAD EVENTS
# ============================================================================

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


task_events = events_df[
    events_df["trial_type"].isin(TASK_LABELS)
].copy()

task_events = task_events.reset_index(drop=True)


print(f"\nTotal events in events.tsv: {len(events_df)}")
print(f"Task trials found: {len(task_events)}")


# ============================================================================
# VALIDATE TASK COUNT
# ============================================================================

if len(task_events) != 720:
    raise RuntimeError(
        f"Expected exactly 720 task trials, "
        f"but found {len(task_events)}."
    )


print("\nTask class counts:")

class_counts = (
    task_events["trial_type"]
    .value_counts()
    .sort_index()
)

print(class_counts)


for label_name in TASK_LABELS:

    count = int(
        class_counts.get(label_name, 0)
    )

    if count != 90:
        raise RuntimeError(
            f"Expected 90 trials for "
            f"{label_name}, found {count}."
        )


print("\n✓ All 8 classes contain exactly 90 trials.")
print("✓ Total task trials = 720.")


# ============================================================================
# LOAD ORIGINAL BDF
# ============================================================================

print("\n" + "=" * 80)
print("LOADING ORIGINAL BDF")
print("=" * 80)

raw = mne.io.read_raw_bdf(
    BDF_FILE,
    preload=True,
    verbose=True
)


print("\nOriginal recording:")
print(raw)


# ============================================================================
# VALIDATE SAMPLING RATE
# ============================================================================

if not np.isclose(
    raw.info["sfreq"],
    SFREQ
):
    raise RuntimeError(
        f"Expected sampling rate {SFREQ} Hz, "
        f"found {raw.info['sfreq']} Hz."
    )


print(
    f"\n✓ Sampling rate verified: "
    f"{raw.info['sfreq']} Hz"
)


# ============================================================================
# APPLY EXISTING PROJECT EEG PREPROCESSING
# ============================================================================

print("\n" + "=" * 80)
print("APPLYING EXISTING EEG PREPROCESSING PIPELINE")
print("=" * 80)

print("\nUsing the project's existing preprocessing pipeline:")

print("  • BioSemi-64 channel identification")
print("  • BioSemi-64 montage")
print("  • 1–40 Hz band-pass")
print("  • 50 Hz notch")
print("  • Iz / TP8 / P6 bad-channel handling")
print("  • Spherical-spline interpolation")
print("  • Average reference")


clean_raw = purify_eeg(raw)


# ============================================================================
# INSPECT PREPROCESSED RECORDING
# ============================================================================

print("\n" + "=" * 80)
print("PREPROCESSED RECORDING")
print("=" * 80)

print(
    f"\nChannels returned by preprocessing: "
    f"{len(clean_raw.ch_names)}"
)

print(
    f"Sampling rate: "
    f"{clean_raw.info['sfreq']} Hz"
)

print("\nChannels returned:")
print(clean_raw.ch_names)

print("\nRemaining bad channels:")
print(clean_raw.info["bads"])


# ============================================================================
# SELECT ONLY THE 64 EEG CHANNELS
# ============================================================================

print("\n" + "=" * 80)
print("SELECTING 64 EEG CHANNELS")
print("=" * 80)

# The existing project preprocessing function keeps the auxiliary
# channels inside the Raw object.
#
# These are NOT EEG channels and must not be included in the
# temporal EEG dataset.

eeg_channel_names = [
    channel
    for channel in clean_raw.ch_names
    if channel not in AUXILIARY_CHANNELS
]


print(
    f"\nEEG channels identified: "
    f"{len(eeg_channel_names)}"
)


if len(eeg_channel_names) != 64:
    raise RuntimeError(
        "Expected exactly 64 EEG channels after removing "
        f"auxiliary channels, but found "
        f"{len(eeg_channel_names)}."
    )


print("\nSelected 64 EEG channels:")

for index, channel in enumerate(
    eeg_channel_names,
    start=1
):
    print(
        f"{index:02d}. {channel}"
    )


# ============================================================================
# KEEP ONLY EEG CHANNELS
# ============================================================================

clean_raw = clean_raw.copy().pick(
    eeg_channel_names
)


print(
    f"\nFinal EEG channel count: "
    f"{len(clean_raw.ch_names)}"
)


if len(clean_raw.ch_names) != 64:
    raise RuntimeError(
        f"Expected 64 final EEG channels, "
        f"found {len(clean_raw.ch_names)}."
    )


print("\n✓ Exactly 64 EEG channels retained.")


# ============================================================================
# EXTRACT CLEAN TEMPORAL EPOCHS
# ============================================================================

print("\n" + "=" * 80)
print("EXTRACTING CLEAN TEMPORAL EPOCHS")
print("=" * 80)

print(
    f"\nSamples per trial: {N_SAMPLES}"
)

print(
    f"Duration per trial: "
    f"{N_SAMPLES / SFREQ:.6f} seconds"
)


epochs = []
metadata_rows = []


for trial_index, row in task_events.iterrows():

    label_name = row["trial_type"]

    start_sample = int(
        row["sample"]
    )

    end_sample = (
        start_sample
        + N_SAMPLES
    )


    # ------------------------------------------------------------------------
    # Validate label
    # ------------------------------------------------------------------------

    if label_name not in LABEL_MAP:
        raise RuntimeError(
            f"Unknown task label: {label_name}"
        )


    # ------------------------------------------------------------------------
    # Validate temporal boundaries
    # ------------------------------------------------------------------------

    if start_sample < 0:
        raise RuntimeError(
            f"Trial {trial_index} has "
            f"negative start sample: "
            f"{start_sample}"
        )


    if end_sample > clean_raw.n_times:
        raise RuntimeError(
            f"Trial {trial_index} exceeds "
            f"recording length."
        )


    # ------------------------------------------------------------------------
    # Extract temporal EEG
    # ------------------------------------------------------------------------

    data = clean_raw.get_data(
        start=start_sample,
        stop=end_sample
    )


    # Expected:
    #
    # (64 channels, 6142 samples)

    expected_trial_shape = (
        64,
        N_SAMPLES
    )


    if data.shape != expected_trial_shape:
        raise RuntimeError(
            f"Unexpected trial shape for "
            f"trial {trial_index}: "
            f"{data.shape}. "
            f"Expected {expected_trial_shape}."
        )


    epochs.append(
        data.astype(
            np.float32
        )
    )


    # ------------------------------------------------------------------------
    # Store metadata
    # ------------------------------------------------------------------------

    metadata_rows.append(
        {
            "subject_id": "sub-01",

            "trial_id": trial_index,

            "start_sample": start_sample,

            "end_sample": end_sample,

            "start_time_sec": (
                start_sample / SFREQ
            ),

            "duration_sec": (
                N_SAMPLES / SFREQ
            ),

            "label_name": label_name,

            "label": LABEL_MAP[label_name],
        }
    )


    # ------------------------------------------------------------------------
    # Progress
    # ------------------------------------------------------------------------

    if (
        trial_index + 1
    ) % 50 == 0:

        print(
            f"Extracted "
            f"{trial_index + 1}/720 trials"
        )


# ============================================================================
# STACK EPOCHS
# ============================================================================

X = np.stack(
    epochs
)


metadata = pd.DataFrame(
    metadata_rows
)


# ============================================================================
# VALIDATE FINAL TEMPORAL DATASET
# ============================================================================

print("\n" + "=" * 80)
print("CLEAN TEMPORAL EEG VALIDATION")
print("=" * 80)


print("\nShape:")
print(X.shape)


expected_shape = (
    720,
    64,
    N_SAMPLES
)


print("\nExpected:")
print(expected_shape)


if X.shape != expected_shape:
    raise RuntimeError(
        f"Unexpected final dataset shape: "
        f"{X.shape}. "
        f"Expected {expected_shape}."
    )


print(
    f"\nData type: "
    f"{X.dtype}"
)

print(
    f"Minimum value: "
    f"{np.min(X):.6f}"
)

print(
    f"Maximum value: "
    f"{np.max(X):.6f}"
)

print(
    f"Mean: "
    f"{np.mean(X):.6f}"
)

print(
    f"Std: "
    f"{np.std(X):.6f}"
)

print(
    f"NaN count: "
    f"{np.isnan(X).sum()}"
)

print(
    f"Inf count: "
    f"{np.isinf(X).sum()}"
)


# ============================================================================
# VALIDATE METADATA
# ============================================================================

print("\nMetadata shape:")
print(metadata.shape)


if len(metadata) != 720:
    raise RuntimeError(
        f"Expected 720 metadata rows, "
        f"found {len(metadata)}."
    )


print("\nClass counts:")

final_class_counts = (
    metadata["label_name"]
    .value_counts()
    .sort_index()
)

print(final_class_counts)


for label_name in TASK_LABELS:

    count = int(
        final_class_counts.get(
            label_name,
            0
        )
    )

    if count != 90:
        raise RuntimeError(
            f"Final dataset contains "
            f"{count} trials for "
            f"{label_name}; expected 90."
        )


# ============================================================================
# CHECK FOR NaN / INF
# ============================================================================

if np.isnan(X).any():
    raise RuntimeError(
        "NaN values detected in final EEG dataset."
    )


if np.isinf(X).any():
    raise RuntimeError(
        "Infinite values detected in final EEG dataset."
    )


print(
    "\n✓ No NaN values."
)

print(
    "✓ No infinite values."
)

print(
    "✓ Shape verified."
)

print(
    "✓ Class balance verified."
)


# ============================================================================
# SAVE DATASET
# ============================================================================

X_FILE = (
    OUTPUT_DIR
    / "sub-01_temporal_eeg_preprocessed.npy"
)

METADATA_FILE = (
    OUTPUT_DIR
    / "sub-01_temporal_eeg_preprocessed_metadata.csv"
)

CHANNEL_FILE = (
    OUTPUT_DIR
    / "preprocessed_channel_names.txt"
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

    for channel in clean_raw.ch_names:

        f.write(
            channel + "\n"
        )


print("\n✓ Saved temporal EEG:")
print(X_FILE)

print("\n✓ Saved metadata:")
print(METADATA_FILE)

print("\n✓ Saved channel names:")
print(CHANNEL_FILE)


# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("STEP 31 COMPLETE")
print("=" * 80)


print("\nFinal dataset:")

print(
    f"Trials   : {X.shape[0]}"
)

print(
    f"Channels : {X.shape[1]}"
)

print(
    f"Samples  : {X.shape[2]}"
)

print(
    f"Duration : "
    f"{X.shape[2] / SFREQ:.6f} sec"
)


print("\nDataset structure:")

print(
    "  720 labelled trials"
)

print(
    "  64 EEG channels"
)

print(
    f"  {N_SAMPLES} temporal samples per trial"
)

print(
    f"  {N_SAMPLES / SFREQ:.6f} seconds per trial"
)

print(
    "  8 balanced task classes"
)


print("\nNo FFT was performed.")
print("No model was trained.")
print("No train/test split was performed.")

print(
    "\nThe preprocessed temporal EEG dataset "
    "is ready for true temporal FFT analysis."
)

print("=" * 80)