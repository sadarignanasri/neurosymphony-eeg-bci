
from pathlib import Path
import json

import mne
import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

PROJECT = Path(__file__).resolve().parents[2]

DATA_ROOT = PROJECT / "major datasets" / "ds004514_real"
OUTPUT_DIR = PROJECT / "experiments" / "verified_eeg_epochs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TASKS = [
    "animal_tactile_imagery_task",
    "animal_visual_imagery_task",
    "animal_auditory_imagery_task",
    "animal_silent_naming_task",
    "tool_tactile_imagery_task",
    "tool_visual_imagery_task",
    "tool_auditory_imagery_task",
    "tool_silent_naming_task",
]

SCALP_CHANNELS = [
    f"A{i}" for i in range(1, 33)
] + [
    f"B{i}" for i in range(1, 33)
]

TARGET_SFREQ = 256
EPOCH_SECONDS = 3.0
EEG_CHANNELS = 64
EXPECTED_SUBJECTS = 12
EXPECTED_TRIALS_PER_SUBJECT = 720

# Keep the original dataset untouched.
# All generated files go into experiments/verified_eeg_epochs.


# --------------------------------------------------
# 2. Validate dataset directory
# --------------------------------------------------

if not DATA_ROOT.exists():
    raise FileNotFoundError(
        f"Dataset directory not found: {DATA_ROOT}"
    )

subject_dirs = sorted(
    path for path in DATA_ROOT.glob("sub-*")
    if path.is_dir()
)

if len(subject_dirs) != EXPECTED_SUBJECTS:
    raise ValueError(
        f"Expected {EXPECTED_SUBJECTS} subjects, "
        f"found {len(subject_dirs)}"
    )

print(f"Dataset: {DATA_ROOT}")
print(f"Subjects found: {len(subject_dirs)}")
print(f"Target sampling rate: {TARGET_SFREQ} Hz")
print(f"Epoch duration: {EPOCH_SECONDS} seconds")


# --------------------------------------------------
# 3. Storage for extracted epochs and audit records
# --------------------------------------------------

all_epochs = []
all_labels = []
all_subjects = []
all_event_samples = []
all_event_names = []
audit_rows = []
alignment_rows = []

samples_per_epoch = int(
    round(EPOCH_SECONDS * TARGET_SFREQ)
)


# --------------------------------------------------
# 4. Process each subject
# --------------------------------------------------

for subject_dir in subject_dirs:
    subject = subject_dir.name
    eeg_dir = subject_dir / "eeg"

    bdf_path = eeg_dir / f"{subject}_task-eeg_eeg.bdf"
    events_path = eeg_dir / f"{subject}_task-eeg_events.tsv"

    if not bdf_path.exists():
        raise FileNotFoundError(
            f"Missing BDF file for {subject}: {bdf_path}"
        )

    if not events_path.exists():
        raise FileNotFoundError(
            f"Missing events file for {subject}: {events_path}"
        )

    print(f"\nLoading {subject}: {bdf_path.name}")

    # Load the original continuous recording.
    raw = mne.io.read_raw_bdf(
        bdf_path,
        preload=True,
        verbose="ERROR",
    )

    original_sfreq = float(raw.info["sfreq"])
    original_first_time = float(raw.first_time)
    original_n_samples = int(raw.n_times)

    print(f"Original sampling rate: {original_sfreq} Hz")
    print(f"Original recording samples: {original_n_samples}")
    print(f"Recording first_time: {original_first_time:.6f} s")

    # Explicitly select the 64 documented scalp electrodes.
    missing_channels = sorted(
        set(SCALP_CHANNELS) - set(raw.ch_names)
    )

    if missing_channels:
        raise ValueError(
            f"{subject}: expected scalp channels are missing: "
            f"{missing_channels}"
        )

    raw.pick(SCALP_CHANNELS)

    if len(raw.ch_names) != EEG_CHANNELS:
        raise ValueError(
            f"{subject}: expected {EEG_CHANNELS} scalp channels, "
            f"found {len(raw.ch_names)}"
        )

    # Confirm channel order is consistent for every subject.
    if raw.ch_names != SCALP_CHANNELS:
        raise ValueError(
            f"{subject}: scalp channel order does not match "
            "the expected order"
        )

    print(f"Selected scalp channels: {len(raw.ch_names)}")

    # Load event metadata.
    events_df = pd.read_csv(
        events_path,
        sep="\t",
    )

    required_columns = {"onset", "trial_type", "sample"}
    missing_columns = required_columns - set(events_df.columns)

    if missing_columns:
        raise ValueError(
            f"{subject}: missing event columns: "
            f"{sorted(missing_columns)}"
        )

    # Keep only the eight semantic imagery task labels.
    task_df = events_df[
        events_df["trial_type"].isin(TASKS)
    ].copy()

    task_df = task_df.reset_index(drop=True)

    if len(task_df) != EXPECTED_TRIALS_PER_SUBJECT:
        raise ValueError(
            f"{subject}: expected "
            f"{EXPECTED_TRIALS_PER_SUBJECT} task events, "
            f"found {len(task_df)}"
        )

    # Validate event fields before using them.
    task_df["onset"] = pd.to_numeric(
        task_df["onset"],
        errors="raise",
    )

    task_df["sample"] = pd.to_numeric(
        task_df["sample"],
        errors="raise",
    )

    if task_df[["onset", "sample"]].isna().any().any():
        raise ValueError(
            f"{subject}: event onset/sample contains missing values"
        )

    if not np.isfinite(
        task_df["onset"].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            f"{subject}: event onset contains invalid values"
        )

    if not np.isfinite(
        task_df["sample"].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            f"{subject}: event sample contains invalid values"
        )

    # Check whether onset and sample describe the same event times.
    # This is an audit only; it does not prove that the BDF timing
    # convention and the TSV timing convention are identical.
    onset_values = task_df["onset"].to_numpy(dtype=float)
    original_event_samples = task_df["sample"].to_numpy(dtype=float)

    sample_times = original_event_samples / original_sfreq
    onset_sample_time_difference = onset_values - sample_times

    alignment_rows.append({
        "subject": subject,
        "original_sampling_rate": original_sfreq,
        "raw_first_time_seconds": original_first_time,
        "task_event_count": len(task_df),
        "max_abs_onset_sample_difference_seconds": float(
            np.max(np.abs(onset_sample_time_difference))
        ),
        "mean_abs_onset_sample_difference_seconds": float(
            np.mean(np.abs(onset_sample_time_difference))
        ),
        "first_event_onset_seconds": float(onset_values[0]),
        "first_event_sample": int(original_event_samples[0]),
        "first_event_sample_time_seconds": float(sample_times[0]),
    })

    # Reject clearly inconsistent event metadata.
    if np.max(np.abs(onset_sample_time_difference)) > (
        1.0 / original_sfreq + 1e-6
    ):
        raise ValueError(
            f"{subject}: events.tsv onset and sample values disagree "
            "by more than one original sampling interval. "
            "Check the dataset timing convention before extraction."
        )

    # Filter the continuous recording before downsampling.
    raw.filter(
        l_freq=1.0,
        h_freq=40.0,
        picks="eeg",
        verbose="ERROR",
    )

    # Resample continuous data to 256 Hz.
    raw.resample(
        TARGET_SFREQ,
        npad="auto",
        verbose="ERROR",
    )

    sfreq = float(raw.info["sfreq"])

    if not np.isclose(sfreq, TARGET_SFREQ):
        raise ValueError(
            f"{subject}: expected {TARGET_SFREQ} Hz after resampling, "
            f"found {sfreq}"
        )

    data = raw.get_data()

    subject_epochs = []
    subject_labels = []
    subject_event_samples = []
    subject_event_names = []

    # Event onset is interpreted as seconds from the start of the
    # recording, matching the events.tsv convention used here.
    # The event/sample audit above checks internal consistency.
    event_samples_resampled = np.rint(
        onset_values * sfreq
    ).astype(np.int64)

    # Extract exactly 3 seconds beginning at each task event.
    for row_index, event_row in task_df.iterrows():
        start = int(event_samples_resampled[row_index])
        stop = start + samples_per_epoch

        if start < 0 or stop > data.shape[1]:
            raise ValueError(
                f"{subject}: event {row_index} produces an epoch "
                f"outside the recording: start={start}, stop={stop}, "
                f"recording_samples={data.shape[1]}"
            )

        epoch = data[:, start:stop]

        if epoch.shape != (EEG_CHANNELS, samples_per_epoch):
            raise ValueError(
                f"{subject}: unexpected epoch shape at event "
                f"{row_index}: {epoch.shape}"
            )

        if not np.isfinite(epoch).all():
            raise ValueError(
                f"{subject}: non-finite values in epoch {row_index}"
            )

        task_name = event_row["trial_type"]
        class_id = TASKS.index(task_name)

        subject_epochs.append(epoch)
        subject_labels.append(class_id)
        subject_event_samples.append(
            int(event_row["sample"])
        )
        subject_event_names.append(task_name)

    if len(subject_epochs) != EXPECTED_TRIALS_PER_SUBJECT:
        raise ValueError(
            f"{subject}: extracted {len(subject_epochs)} epochs, "
            f"expected {EXPECTED_TRIALS_PER_SUBJECT}"
        )

    epochs_array = np.stack(subject_epochs).astype(
        np.float32,
        copy=False,
    )

    labels_array = np.asarray(
        subject_labels,
        dtype=np.int64,
    )

    # Require 90 trials for each of the eight task classes.
    class_counts = np.bincount(
        labels_array,
        minlength=len(TASKS),
    )

    if not np.all(class_counts == 90):
        raise ValueError(
            f"{subject}: expected 90 trials per class, got "
            f"{dict(zip(TASKS, class_counts.tolist()))}"
        )

    all_epochs.append(epochs_array)
    all_labels.append(labels_array)
    all_subjects.extend([subject] * len(labels_array))
    all_event_samples.extend(subject_event_samples)
    all_event_names.extend(subject_event_names)

    for class_id, task_name in enumerate(TASKS):
        audit_rows.append({
            "subject": subject,
            "class_id": class_id,
            "task": task_name,
            "trial_count": int(class_counts[class_id]),
        })

    print(
        f"{subject}: extracted {len(epochs_array)} epochs, "
        f"shape={epochs_array.shape}"
    )


# --------------------------------------------------
# 5. Combine all subjects
# --------------------------------------------------

X = np.concatenate(all_epochs, axis=0)
y = np.concatenate(all_labels, axis=0)
subjects = np.asarray(all_subjects)
event_samples_original = np.asarray(
    all_event_samples,
    dtype=np.int64,
)
event_names = np.asarray(all_event_names)

expected_total_trials = (
    EXPECTED_SUBJECTS * EXPECTED_TRIALS_PER_SUBJECT
)

if X.shape != (
    expected_total_trials,
    EEG_CHANNELS,
    samples_per_epoch,
):
    raise ValueError(
        f"Unexpected final EEG tensor shape: {X.shape}"
    )

if len(y) != expected_total_trials:
    raise ValueError(
        f"Expected {expected_total_trials} labels, found {len(y)}"
    )

if len(subjects) != len(y):
    raise ValueError(
        "Subject identifiers and labels are misaligned"
    )

if len(event_names) != len(y):
    raise ValueError(
        "Event names and labels are misaligned"
    )

if not np.isfinite(X).all():
    raise ValueError(
        "Final EEG tensor contains non-finite values"
    )

total_class_counts = np.bincount(
    y,
    minlength=len(TASKS),
)

expected_count_per_class = EXPECTED_SUBJECTS * 90

if not np.all(total_class_counts == expected_count_per_class):
    raise ValueError(
        "Final class counts are incorrect: "
        f"{dict(zip(TASKS, total_class_counts.tolist()))}"
    )


# --------------------------------------------------
# 6. Save verified EEG epochs
# --------------------------------------------------

np.savez_compressed(
    OUTPUT_DIR / "eeg_epochs.npz",
    X=X,
    y=y,
    subjects=subjects,
    event_samples_original=event_samples_original,
    event_names=event_names,
    task_names=np.asarray(TASKS),
    sampling_rate=np.asarray(TARGET_SFREQ),
    channel_names=np.asarray(SCALP_CHANNELS),
)

pd.DataFrame(audit_rows).to_csv(
    OUTPUT_DIR / "class_counts.csv",
    index=False,
)

pd.DataFrame(alignment_rows).to_csv(
    OUTPUT_DIR / "event_alignment_audit.csv",
    index=False,
)

metadata = {
    "dataset": "ds004514_real",
    "subjects": int(len(np.unique(subjects))),
    "trials": int(len(y)),
    "channels": int(X.shape[1]),
    "samples_per_trial": int(X.shape[2]),
    "sampling_rate": TARGET_SFREQ,
    "epoch_duration_seconds": EPOCH_SECONDS,
    "task_names": TASKS,
    "class_counts": {
        TASKS[i]: int(np.sum(y == i))
        for i in range(len(TASKS))
    },
    "channel_names": SCALP_CHANNELS,
    "preprocessing": {
        "bandpass_filter_hz": [1.0, 40.0],
        "resampling_hz": TARGET_SFREQ,
        "epoch_start": "events.tsv onset",
        "epoch_duration_seconds": EPOCH_SECONDS,
    },
    "validation": {
        "expected_subjects": EXPECTED_SUBJECTS,
        "expected_trials_per_subject": EXPECTED_TRIALS_PER_SUBJECT,
        "expected_trials_per_class_per_subject": 90,
        "all_subjects_validated": True,
        "all_epochs_have_expected_shape": True,
        "all_values_finite": True,
        "onset_sample_consistency_checked": True,
        "bdf_timing_convention_independently_confirmed": False,
    },
    "note": (
        "The events.tsv onset and sample values are checked for "
        "internal consistency. Before interpreting model results, "
        "independently confirm that event onset is relative to the "
        "start of the loaded BDF recording and that the selected "
        "3-second window matches the task protocol."
    ),
}

(OUTPUT_DIR / "metadata.json").write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8",
)

print("\nExtraction completed successfully.")
print(f"Final tensor shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Unique subjects: {len(np.unique(subjects))}")
print(f"Total trials: {len(y)}")
print("\nClass counts:")

for class_id, task_name in enumerate(TASKS):
    print(f"{class_id}: {task_name}: {total_class_counts[class_id]}")

print(f"\nOutput directory: {OUTPUT_DIR}")
print("Files created:")
print("  eeg_epochs.npz")
print("  class_counts.csv")
print("  event_alignment_audit.csv")
print("  metadata.json")
