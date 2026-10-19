"""
NEURO-SYMPHONY
DS004514 — ALL SUBJECT ORIGINAL BDF VERIFICATION

Purpose
-------
Read-only integrity verification of the 12 original EEG BDF recordings.

This script DOES NOT:
- preprocess EEG
- modify BDF files
- modify TSV/JSON files
- create epochs
- create features
- train models
- perform scaling
- modify the dataset

It verifies:
1. BDF file exists
2. Required metadata files exist
3. BDF can be opened by MNE
4. Channel count
5. Sampling frequency
6. Recording duration
7. Presence of BioSemi A/B EEG channels
8. Presence of Status channel
9. Events TSV can be read
10. Number of task trials
11. Number of classes
12. Class balance

Expected DS004514 EEG structure:
- 80 total channels
- 2048 Hz sampling frequency
- 64 EEG channels (A1-A32 + B1-B32)
- Status stimulus channel
- 720 task trials
- 8 task classes
- 90 trials per class

IMPORTANT:
This script does NOT preload the ~2 GB BDF data into RAM.
Only the BDF header is inspected.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import mne


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "dataset_validation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_FILE = (
    OUTPUT_DIR
    / "ds004514_all_subject_bdf_integrity_report.csv"
)


# ============================================================================
# SUBJECTS
# ============================================================================

SUBJECTS = [
    f"sub-{i:02d}"
    for i in range(1, 13)
]


# ============================================================================
# EXPECTED DATASET PROPERTIES
# ============================================================================

EXPECTED_CHANNELS = 80

EXPECTED_SFREQ = 2048.0

EXPECTED_EEG_CHANNELS = 64

EXPECTED_TASK_TRIALS = 720

EXPECTED_CLASSES = 8

EXPECTED_TRIALS_PER_CLASS = 90


# ============================================================================
# EXPECTED BIOSEMI EEG CHANNELS
# ============================================================================

EXPECTED_EEG_NAMES = (
    [f"A{i}" for i in range(1, 33)]
    +
    [f"B{i}" for i in range(1, 33)]
)


# ============================================================================
# REQUIRED FILES
# ============================================================================

def required_files(subject):

    subject_dir = RAW_DIR / subject

    return {
        "bdf": (
            subject_dir
            / f"{subject}_task-eeg_eeg.bdf"
        ),
        "channels_tsv": (
            subject_dir
            / f"{subject}_task-eeg_channels.tsv"
        ),
        "eeg_json": (
            subject_dir
            / f"{subject}_task-eeg_eeg.json"
        ),
        "events_json": (
            subject_dir
            / f"{subject}_task-eeg_events.json"
        ),
        "events_tsv": (
            subject_dir
            / f"{subject}_task-eeg_events.tsv"
        ),
    }


# ============================================================================
# FIND TASK LABEL COLUMN
# ============================================================================

def find_task_label_column(events):

    candidates = [
        "trial_type",
        "event_type",
        "task",
        "condition",
    ]

    for column in candidates:

        if column in events.columns:
            return column

    return None


# ============================================================================
# MAIN VERIFICATION
# ============================================================================

def main():

    print("=" * 100)
    print(
        "NEURO-SYMPHONY — DS004514 "
        "ALL-SUBJECT ORIGINAL BDF VERIFICATION"
    )
    print("=" * 100)

    print()
    print("Project root:")
    print(PROJECT_ROOT)

    print()
    print("Raw dataset:")
    print(RAW_DIR)

    print()
    print("IMPORTANT:")
    print(
        "This script is READ-ONLY and does not modify "
        "the original dataset."
    )

    print()
    print("=" * 100)

    results = []

    passed_subjects = 0
    failed_subjects = 0

    # ========================================================================
    # SUBJECT LOOP
    # ========================================================================

    for subject in SUBJECTS:

        print()
        print("=" * 100)
        print(f"VERIFYING {subject}")
        print("=" * 100)

        files = required_files(subject)

        subject_passed = True
        failure_reasons = []

        # ====================================================================
        # CHECK REQUIRED FILES
        # ====================================================================

        print()
        print("Checking required files...")

        for file_type, path in files.items():

            if path.exists():

                print(
                    f"✓ {file_type}: "
                    f"{path.name}"
                )

            else:

                print(
                    f"✗ {file_type} MISSING: "
                    f"{path}"
                )

                subject_passed = False

                failure_reasons.append(
                    f"{file_type}_missing"
                )

        # ====================================================================
        # BDF VERIFICATION
        # ====================================================================

        n_channels = np.nan
        sfreq = np.nan
        duration = np.nan
        n_eeg_channels = np.nan
        status_present = False

        channel_names = []

        if files["bdf"].exists():

            print()
            print("Opening BDF header with MNE...")
            print(
                "No full data preload is performed."
            )

            try:

                raw = mne.io.read_raw_bdf(
                    files["bdf"],
                    preload=False,
                    verbose="ERROR"
                )

                print(
                    "✓ MNE successfully opened "
                    "the BDF."
                )

                n_channels = len(
                    raw.ch_names
                )

                sfreq = float(
                    raw.info["sfreq"]
                )

                duration = float(
                    raw.n_times
                    / raw.info["sfreq"]
                )

                channel_names = list(
                    raw.ch_names
                )

                print()
                print(
                    f"Channels        : "
                    f"{n_channels}"
                )

                print(
                    f"Sampling rate   : "
                    f"{sfreq:.6f} Hz"
                )

                print(
                    f"Samples         : "
                    f"{raw.n_times}"
                )

                print(
                    f"Duration        : "
                    f"{duration:.3f} seconds"
                )

                # ============================================================
                # CHANNEL COUNT
                # ============================================================

                if n_channels == EXPECTED_CHANNELS:

                    print(
                        f"✓ Channel count = "
                        f"{EXPECTED_CHANNELS}"
                    )

                else:

                    print(
                        f"✗ Unexpected channel count: "
                        f"{n_channels}"
                    )

                    subject_passed = False

                    failure_reasons.append(
                        "unexpected_channel_count"
                    )

                # ============================================================
                # SAMPLING RATE
                # ============================================================

                if np.isclose(
                    sfreq,
                    EXPECTED_SFREQ
                ):

                    print(
                        f"✓ Sampling rate = "
                        f"{EXPECTED_SFREQ} Hz"
                    )

                else:

                    print(
                        f"✗ Unexpected sampling rate: "
                        f"{sfreq}"
                    )

                    subject_passed = False

                    failure_reasons.append(
                        "unexpected_sampling_rate"
                    )

                # ============================================================
                # EEG CHANNELS
                # ============================================================

                eeg_channels_found = [
                    name
                    for name in EXPECTED_EEG_NAMES
                    if name in channel_names
                ]

                n_eeg_channels = len(
                    eeg_channels_found
                )

                print()
                print(
                    f"BioSemi EEG channels found: "
                    f"{n_eeg_channels}"
                )

                if (
                    n_eeg_channels
                    == EXPECTED_EEG_CHANNELS
                ):

                    print(
                        f"✓ Found all "
                        f"{EXPECTED_EEG_CHANNELS} "
                        f"expected A/B EEG channels."
                    )

                else:

                    print(
                        "✗ Expected 64 A/B EEG "
                        f"channels but found "
                        f"{n_eeg_channels}."
                    )

                    subject_passed = False

                    failure_reasons.append(
                        "unexpected_eeg_channel_count"
                    )

                # ============================================================
                # STATUS CHANNEL
                # ============================================================

                status_present = (
                    "Status"
                    in channel_names
                )

                if status_present:

                    print(
                        "✓ Status channel present."
                    )

                else:

                    print(
                        "✗ Status channel missing."
                    )

                    subject_passed = False

                    failure_reasons.append(
                        "status_channel_missing"
                    )

                raw.close()

            except Exception as exc:

                print()
                print(
                    "✗ MNE could not open "
                    "the BDF."
                )

                print(
                    f"Error: {exc}"
                )

                subject_passed = False

                failure_reasons.append(
                    "mne_bdf_read_failure"
                )

        # ====================================================================
        # EVENTS TSV VERIFICATION
        # ====================================================================

        task_trials = 0
        n_classes = 0
        class_distribution = {}

        if files["events_tsv"].exists():

            print()
            print(
                "Reading events TSV..."
            )

            try:

                events = pd.read_csv(
                    files["events_tsv"],
                    sep="\t"
                )

                print(
                    f"✓ Events TSV loaded."
                )

                print(
                    f"Total event rows: "
                    f"{len(events)}"
                )

                print(
                    "Columns:"
                )

                print(
                    list(events.columns)
                )

                label_column = (
                    find_task_label_column(
                        events
                    )
                )

                if label_column is None:

                    print(
                        "✗ Could not identify "
                        "the task-label column."
                    )

                    subject_passed = False

                    failure_reasons.append(
                        "task_label_column_missing"
                    )

                else:

                    print(
                        f"✓ Task label column: "
                        f"{label_column}"
                    )

                    labels = (
                        events[
                            label_column
                        ]
                        .astype(str)
                        .str.strip()
                    )

                    # --------------------------------------------------------
                    # TASK EVENTS
                    # --------------------------------------------------------

                    task_mask = (
                        labels.str.endswith(
                            "_task"
                        )
                    )

                    task_labels = labels[
                        task_mask
                    ]

                    task_trials = len(
                        task_labels
                    )

                    print()
                    print(
                        f"Task trials: "
                        f"{task_trials}"
                    )

                    if (
                        task_trials
                        == EXPECTED_TASK_TRIALS
                    ):

                        print(
                            f"✓ Exactly "
                            f"{EXPECTED_TASK_TRIALS} "
                            "task trials."
                        )

                    else:

                        print(
                            f"✗ Expected "
                            f"{EXPECTED_TASK_TRIALS} "
                            f"task trials but found "
                            f"{task_trials}."
                        )

                        subject_passed = False

                        failure_reasons.append(
                            "unexpected_task_trial_count"
                        )

                    # --------------------------------------------------------
                    # CLASS DISTRIBUTION
                    # --------------------------------------------------------

                    class_distribution = (
                        task_labels.value_counts()
                        .sort_index()
                        .to_dict()
                    )

                    n_classes = len(
                        class_distribution
                    )

                    print()
                    print(
                        f"Number of task classes: "
                        f"{n_classes}"
                    )

                    for label, count in (
                        class_distribution.items()
                    ):

                        print(
                            f"  {label}: "
                            f"{count}"
                        )

                    # --------------------------------------------------------
                    # CLASS COUNT
                    # --------------------------------------------------------

                    if (
                        n_classes
                        == EXPECTED_CLASSES
                    ):

                        print(
                            f"✓ Exactly "
                            f"{EXPECTED_CLASSES} "
                            "task classes."
                        )

                    else:

                        print(
                            f"✗ Expected "
                            f"{EXPECTED_CLASSES} "
                            f"classes but found "
                            f"{n_classes}."
                        )

                        subject_passed = False

                        failure_reasons.append(
                            "unexpected_class_count"
                        )

                    # --------------------------------------------------------
                    # 90 TRIALS PER CLASS
                    # --------------------------------------------------------

                    counts_are_correct = all(
                        count
                        == EXPECTED_TRIALS_PER_CLASS
                        for count
                        in class_distribution.values()
                    )

                    if (
                        n_classes
                        == EXPECTED_CLASSES
                        and counts_are_correct
                    ):

                        print(
                            f"✓ Every class contains "
                            f"{EXPECTED_TRIALS_PER_CLASS} "
                            "trials."
                        )

                    else:

                        print(
                            "✗ Class distribution "
                            "is not 90 trials/class."
                        )

                        subject_passed = False

                        failure_reasons.append(
                            "class_imbalance"
                        )

            except Exception as exc:

                print()
                print(
                    "✗ Failed to read "
                    "events TSV."
                )

                print(
                    f"Error: {exc}"
                )

                subject_passed = False

                failure_reasons.append(
                    "events_tsv_read_failure"
                )

        # ====================================================================
        # SUBJECT RESULT
        # ====================================================================

        if subject_passed:

            print()
            print(
                f"✓ {subject} PASSED "
                "ALL INTEGRITY CHECKS."
            )

            passed_subjects += 1

        else:

            print()
            print(
                f"✗ {subject} FAILED "
                "INTEGRITY CHECKS."
            )

            print(
                "Reasons:"
            )

            for reason in failure_reasons:

                print(
                    f"  - {reason}"
                )

            failed_subjects += 1

        results.append(
            {
                "subject": subject,
                "bdf_exists": files[
                    "bdf"
                ].exists(),
                "mne_readable": (
                    "mne_bdf_read_failure"
                    not in failure_reasons
                    and files[
                        "bdf"
                    ].exists()
                ),
                "channels": n_channels,
                "sampling_frequency_hz": sfreq,
                "duration_seconds": duration,
                "eeg_channels": n_eeg_channels,
                "status_channel": status_present,
                "task_trials": task_trials,
                "classes": n_classes,
                "all_classes_90": (
                    n_classes
                    == EXPECTED_CLASSES
                    and all(
                        count
                        == EXPECTED_TRIALS_PER_CLASS
                        for count
                        in class_distribution.values()
                    )
                    if class_distribution
                    else False
                ),
                "status": (
                    "PASS"
                    if subject_passed
                    else "FAIL"
                ),
                "failure_reasons": (
                    ";".join(
                        failure_reasons
                    )
                    if failure_reasons
                    else ""
                ),
            }
        )

    # =========================================================================
    # SAVE REPORT
    # =========================================================================

    report = pd.DataFrame(
        results
    )

    report.to_csv(
        REPORT_FILE,
        index=False
    )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    print()
    print("=" * 100)
    print(
        "FINAL DS004514 "
        "INTEGRITY SUMMARY"
    )
    print("=" * 100)

    print()

    for result in results:

        print(
            f"{result['subject']}: "
            f"{result['status']}"
        )

    print()
    print("-" * 100)

    print(
        f"Subjects checked: "
        f"{len(SUBJECTS)}"
    )

    print(
        f"Subjects passed:  "
        f"{passed_subjects}"
    )

    print(
        f"Subjects failed:  "
        f"{failed_subjects}"
    )

    print()
    print(
        "✓ Full report saved to:"
    )

    print(
        REPORT_FILE
    )

    print()

    if failed_subjects == 0:

        print("=" * 100)
        print(
            "✓ ALL 12 SUBJECTS PASSED."
        )
        print("=" * 100)

        print()
        print(
            "The original BDF recordings and "
            "task-event metadata are ready "
            "for the temporal EEG pipeline."
        )

    else:

        print("=" * 100)
        print(
            "⚠ ONE OR MORE SUBJECTS FAILED."
        )
        print("=" * 100)

        print()
        print(
            "Do NOT start the full "
            "12-subject preprocessing pipeline "
            "until the failed subjects "
            "have been reviewed."
        )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()