
from pathlib import Path
import json

import mne
import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PROJECT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT / "major datasets" / "ds004022"
OUTPUT_DIR = PROJECT / "experiments" / "ds004022_verified_epochs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CUE_TO_LABEL = {
    "S 3": "reaching",
    "S 4": "grasping",
    "S 5": "lifting",
    "S 6": "twisting",
}

CLASS_NAMES = [
    "reaching",
    "grasping",
    "lifting",
    "twisting",
]

CLASS_TO_ID = {
    name: index for index, name in enumerate(CLASS_NAMES)
}

IMAGERY_MARKER = "S 8"
EXPECTED_TRIALS_PER_RUN = 40
EXPECTED_TRIALS_PER_CLASS_PER_RUN = 10
SAMPLING_RATE = 500
IMAGERY_SECONDS = 5.0
CHANNEL_COUNT = 18

SAMPLES_PER_EPOCH = int(
    SAMPLING_RATE * IMAGERY_SECONDS
)

EXPECTED_CHANNELS = [
    "C3", "C4", "Cz", "CP1", "CP2", "CP5", "CP6",
    "FC1", "FC2", "FC5", "FC6", "Fp2",
    "O1", "O2", "Oz", "P3", "P4", "Pz",
]

if not DATA_ROOT.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_ROOT}"
    )

set_files = sorted(DATA_ROOT.rglob("*run-*_eeg.set"))

if not set_files:
    raise FileNotFoundError(
        f"No run EEG files found inside {DATA_ROOT}"
    )

print("=" * 70)
print("DS004022 VERIFIED EEG EPOCH EXTRACTION")
print("=" * 70)
print(f"Recordings found: {len(set_files)}")
print(f"Output directory: {OUTPUT_DIR}")


# --------------------------------------------------
# 2. Extract each run
# --------------------------------------------------

all_X = []
all_y = []
all_subjects = []
all_runs = []
all_onsets = []
all_cues = []
audit_rows = []

for set_path in set_files:
    subject = next(
        part for part in set_path.parts
        if part.startswith("sub-")
    )

    run_name = set_path.stem.split("_run-")[-1]
    run_number = int(run_name.split("_")[0])

    print(f"\nProcessing {subject}, run {run_number}")

    raw = mne.io.read_raw_eeglab(
        set_path,
        preload=True,
        verbose="ERROR",
    )

    try:
        # Normalize channel names and annotation labels.
        raw.rename_channels({
            name: name.strip()
            for name in raw.ch_names
        })

        annotations = sorted(
            [
                {
                    "onset": float(annotation["onset"]),
                    "label": " ".join(
                        str(annotation["description"]).split()
                    ),
                }
                for annotation in raw.annotations
            ],
            key=lambda item: item["onset"],
        )

        if not np.isclose(
            raw.info["sfreq"], SAMPLING_RATE
        ):
            raise ValueError(
                f"{subject} run {run_number}: expected "
                f"{SAMPLING_RATE} Hz, got {raw.info['sfreq']}"
            )

        missing_channels = sorted(
            set(EXPECTED_CHANNELS) - set(raw.ch_names)
        )

        if missing_channels:
            raise ValueError(
                f"{subject} run {run_number}: missing channels "
                f"{missing_channels}"
            )

        # Use the same 18 EEG channels and order for every recording.
        raw.pick(EXPECTED_CHANNELS)

        if len(raw.ch_names) != CHANNEL_COUNT:
            raise ValueError(
                f"Expected {CHANNEL_COUNT} channels, "
                f"got {len(raw.ch_names)}"
            )

        # The imagery window starts at S 8.
        # Its class comes from the most recent S 3-S 6 cue.
        current_cue = None
        trials = []

        for event in annotations:
            label = event["label"]

            if label in CUE_TO_LABEL:
                current_cue = label

            elif label == IMAGERY_MARKER:
                if current_cue is None:
                    raise ValueError(
                        f"{subject} run {run_number}: "
                        f"S 8 at {event['onset']:.3f}s has "
                        "no preceding class cue"
                    )

                class_name = CUE_TO_LABEL[current_cue]

                trials.append({
                    "onset": event["onset"],
                    "cue": current_cue,
                    "class_name": class_name,
                    "class_id": CLASS_TO_ID[class_name],
                })

                # Each cue applies to one imagery onset only.
                current_cue = None

        if len(trials) != EXPECTED_TRIALS_PER_RUN:
            raise ValueError(
                f"{subject} run {run_number}: expected "
                f"{EXPECTED_TRIALS_PER_RUN} imagery trials, "
                f"found {len(trials)}"
            )

        counts = pd.Series(
            [trial["class_name"] for trial in trials]
        ).value_counts()

        for class_name in CLASS_NAMES:
            if int(counts.get(class_name, 0)) != (
                EXPECTED_TRIALS_PER_CLASS_PER_RUN
            ):
                raise ValueError(
                    f"{subject} run {run_number}: expected "
                    f"{EXPECTED_TRIALS_PER_CLASS_PER_RUN} "
                    f"{class_name} trials, got "
                    f"{int(counts.get(class_name, 0))}"
                )

        # Apply the same 1-40 Hz bandpass filter to each recording.
        raw.filter(
            l_freq=1.0,
            h_freq=40.0,
            picks="eeg",
            verbose="ERROR",
        )

        data = raw.get_data()
        run_X = []

        for trial_index, trial in enumerate(trials):
            # Convert event time into an index relative to raw.times.
            start = int(
                raw.time_as_index(
                    [trial["onset"]],
                    use_rounding=True,
                )[0]
            )
            stop = start + SAMPLES_PER_EPOCH

            if start < 0 or stop > data.shape[1]:
                raise ValueError(
                    f"{subject} run {run_number}: trial "
                    f"{trial_index} exceeds recording boundaries"
                )

            epoch = data[:, start:stop]

            if epoch.shape != (
                CHANNEL_COUNT, SAMPLES_PER_EPOCH
            ):
                raise ValueError(
                    f"Unexpected epoch shape: {epoch.shape}"
                )

            if not np.isfinite(epoch).all():
                raise ValueError(
                    f"Non-finite EEG values in "
                    f"{subject} run {run_number}, "
                    f"trial {trial_index}"
                )

            run_X.append(epoch.astype(np.float32))

            all_y.append(trial["class_id"])
            all_subjects.append(subject)
            all_runs.append(run_number)
            all_onsets.append(trial["onset"])
            all_cues.append(trial["cue"])

            audit_rows.append({
                "subject": subject,
                "run": run_number,
                "trial_index": trial_index,
                "imagery_onset_seconds": trial["onset"],
                "cue_marker": trial["cue"],
                "class_name": trial["class_name"],
                "class_id": trial["class_id"],
            })

        run_X = np.stack(run_X)

        all_X.append(run_X)

        print(f"Trials extracted: {len(run_X)}")
        print(f"Epoch shape: {run_X.shape}")
        print(
            "Class counts:",
            {
                name: int(counts.get(name, 0))
                for name in CLASS_NAMES
            },
        )

    finally:
        raw.close()


# --------------------------------------------------
# 3. Combine verified epochs
# --------------------------------------------------

X = np.concatenate(all_X, axis=0)
y = np.asarray(all_y, dtype=np.int64)
subjects = np.asarray(all_subjects)
runs = np.asarray(all_runs, dtype=np.int64)
onsets = np.asarray(all_onsets, dtype=np.float64)
cues = np.asarray(all_cues)

expected_total = (
    len(set_files) * EXPECTED_TRIALS_PER_RUN
)

if X.shape != (
    expected_total,
    CHANNEL_COUNT,
    SAMPLES_PER_EPOCH,
):
    raise ValueError(
        f"Unexpected combined tensor shape: {X.shape}"
    )

if not (
    len(y) == len(subjects) == len(runs)
    == len(onsets) == len(cues) == len(X)
):
    raise ValueError("Combined metadata is misaligned.")

if not np.isfinite(X).all():
    raise ValueError("Combined EEG data contains invalid values.")

# Confirm class balance for every subject and run.
audit_df = pd.DataFrame(audit_rows)

counts_df = (
    audit_df.groupby(
        ["subject", "run", "class_name"]
    )
    .size()
    .unstack(fill_value=0)
    .reindex(columns=CLASS_NAMES, fill_value=0)
)

if not (
    counts_df.to_numpy()
    == EXPECTED_TRIALS_PER_CLASS_PER_RUN
).all():
    raise ValueError(
        "At least one subject/run has unexpected class counts."
    )


# --------------------------------------------------
# 4. Save the dataset and audit reports
# --------------------------------------------------

np.savez_compressed(
    OUTPUT_DIR / "eeg_epochs.npz",
    X=X,
    y=y,
    subjects=subjects,
    runs=runs,
    onsets_seconds=onsets,
    cue_markers=cues,
    class_names=np.asarray(CLASS_NAMES),
    channel_names=np.asarray(EXPECTED_CHANNELS),
    sampling_rate=np.asarray(SAMPLING_RATE),
)

audit_df.to_csv(
    OUTPUT_DIR / "trial_audit.csv",
    index=False,
)

counts_df.to_csv(
    OUTPUT_DIR / "class_counts_by_subject_run.csv"
)

metadata = {
    "dataset": "ds004022",
    "recordings_processed": len(set_files),
    "subjects_present": sorted(
        np.unique(subjects).tolist()
    ),
    "runs_present": sorted(
        np.unique(runs).tolist()
    ),
    "total_trials": int(len(y)),
    "channels": CHANNEL_COUNT,
    "channel_names": EXPECTED_CHANNELS,
    "sampling_rate_hz": SAMPLING_RATE,
    "epoch_duration_seconds": IMAGERY_SECONDS,
    "samples_per_epoch": SAMPLES_PER_EPOCH,
    "filter_hz": [1.0, 40.0],
    "imagery_onset_marker": IMAGERY_MARKER,
    "cue_to_class": CUE_TO_LABEL,
    "class_counts": {
        name: int(np.sum(y == CLASS_TO_ID[name]))
        for name in CLASS_NAMES
    },
    "note": (
        "Class cues are mapped according to the published "
        "MOABB Lee2022 dataset implementation. Epochs start "
        "at S 8 and contain five seconds of EEG."
    ),
}

(OUTPUT_DIR / "metadata.json").write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8",
)

print("\n" + "=" * 70)
print("EXTRACTION COMPLETE")
print("=" * 70)
print(f"EEG tensor shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Unique subjects: {len(np.unique(subjects))}")
print(f"Runs processed: {len(set_files)}")
print(f"Total trials: {len(y)}")

print("\nOverall class counts:")
for name in CLASS_NAMES:
    print(
        f"{name}: "
        f"{int(np.sum(y == CLASS_TO_ID[name]))}"
    )

print(f"\nSaved files in: {OUTPUT_DIR}")
print("  eeg_epochs.npz")
print("  trial_audit.csv")
print("  class_counts_by_subject_run.csv")
print("  metadata.json")
