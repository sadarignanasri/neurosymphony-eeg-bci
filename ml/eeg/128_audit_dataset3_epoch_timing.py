
"""
Dataset 3: EEG epoch structure and timing audit.

Dataset:
    major datasets/28955456

Audit outputs:
    experiments/dataset3_28955456_audit/

Original dataset files are read-only.
This script does not train models or modify source files.
"""

from pathlib import Path
import csv
import json
import traceback

import numpy as np
from scipy.io import loadmat


# ============================================================
# 1. PATHS AND EXPECTED DATASET STRUCTURE
# ============================================================

PROJECT_ROOT = Path(r"D:\neurosymphony-eeg-bci")

DATASET_ROOT = PROJECT_ROOT / "major datasets" / "28955456"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "dataset3_28955456_audit"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_FS = 256
EXPECTED_CHANNELS = 32
EXPECTED_TRIALS = 30

EXPECTED_LABEL_COUNTS = {
    0: 15,
    1: 15,
}

# Taken from the original dataset-generation MATLAB source.
CHANNEL_NAMES = [
    "FC1", "AF3", "AF4", "CP1", "CP2", "CP6", "Cz", "C3",
    "C4", "T7", "T8", "FC2", "FC5", "FC6", "Pz", "CP5",
    "PO3", "PO4", "Oz", "FP2", "FP1", "FZ", "F3", "F4",
    "F7", "F8", "P3", "P4", "P7", "P8", "O1", "O2",
]

# Candidate window used by the original MATLAB classifier.
CANDIDATE_START_SECONDS = 13
CANDIDATE_END_SECONDS = 23


# ============================================================
# 2. HELPERS
# ============================================================

def json_safe(value):
    """Convert NumPy values and Paths into JSON-compatible values."""
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]

    return value


def load_epoch_file(path):
    """
    Load a MATLAB .mat file and extract its 'epo' structure.

    Important:
    With struct_as_record=True, epo = mat['epo'][0, 0]
    is a structured record. Its X field already contains
    the complete numeric EEG array.
    """
    mat = loadmat(
        str(path),
        squeeze_me=False,
        struct_as_record=True,
    )

    if "epo" not in mat:
        raise ValueError("MATLAB file does not contain an 'epo' variable.")

    epo_container = mat["epo"]

    if not isinstance(epo_container, np.ndarray):
        raise ValueError("'epo' is not a NumPy array.")

    if epo_container.size != 1:
        raise ValueError(
            f"Expected one 'epo' structure; found {epo_container.shape}."
        )

    epo = epo_container[0, 0]

    if epo.dtype.names is None:
        raise ValueError("'epo' is not a structured MATLAB record.")

    required_fields = {"X", "y", "fs"}

    missing = required_fields - set(epo.dtype.names)

    if missing:
        raise ValueError(f"Missing required fields: {sorted(missing)}")

    # FIX: These fields already contain the numeric arrays.
    # Do not use epo["X"][0, 0] or epo["y"][0, 0].
    X = np.asarray(epo["X"], dtype=np.float64)
    y = np.asarray(epo["y"]).reshape(-1)
    fs = float(np.asarray(epo["fs"]).reshape(-1)[0])

    return X, y, fs


def audit_one_file(path, group):
    """Run structural, label, finite-value, and candidate-window checks."""

    record = {
        "file": str(path),
        "group": group,
        "status": "FAILED",
        "errors": [],
        "warnings": [],
        "eeg_shape": None,
        "label_shape": None,
        "sampling_frequency_hz": None,
        "channel_count": None,
        "trial_count": None,
        "label_counts": None,
        "finite_fraction": None,
        "candidate_window_start_seconds": CANDIDATE_START_SECONDS,
        "candidate_window_end_seconds": CANDIDATE_END_SECONDS,
        "candidate_window_samples": None,
        "candidate_window_finite_fraction": None,
        "candidate_window_within_epoch": False,
        "candidate_window_timing_verified": False,
    }

    try:
        X, y, fs = load_epoch_file(path)

        record["eeg_shape"] = list(X.shape)
        record["label_shape"] = list(y.shape)
        record["sampling_frequency_hz"] = fs

        # ----------------------------------------------------
        # Structural checks
        # ----------------------------------------------------

        if X.ndim != 3:
            record["errors"].append(
                f"Expected 3D EEG array (samples, channels, trials); "
                f"found {X.ndim}D with shape {X.shape}."
            )
        else:
            samples, channels, trials = X.shape

            record["channel_count"] = int(channels)
            record["trial_count"] = int(trials)

            if channels != EXPECTED_CHANNELS:
                record["errors"].append(
                    f"Expected {EXPECTED_CHANNELS} channels; found {channels}."
                )

            if trials != EXPECTED_TRIALS:
                record["errors"].append(
                    f"Expected {EXPECTED_TRIALS} trials; found {trials}."
                )

            if not np.isclose(fs, EXPECTED_FS):
                record["errors"].append(
                    f"Expected sampling frequency {EXPECTED_FS} Hz; found {fs}."
                )

            expected_samples = 3131 + 25 * EXPECTED_FS

            if samples != expected_samples:
                record["warnings"].append(
                    f"Expected {expected_samples} samples based on the "
                    f"source epoch formula; found {samples}."
                )

            finite_fraction = float(np.isfinite(X).mean())
            record["finite_fraction"] = finite_fraction

            if finite_fraction < 1.0:
                record["errors"].append(
                    f"EEG contains non-finite values; finite fraction "
                    f"is {finite_fraction:.6f}."
                )

        # ----------------------------------------------------
        # Label checks
        # ----------------------------------------------------

        if y.size != EXPECTED_TRIALS:
            record["errors"].append(
                f"Expected {EXPECTED_TRIALS} labels; found {y.size}."
            )
        elif not np.all(np.isin(y, [0, 1])):
            record["errors"].append(
                f"Expected binary labels 0 and 1; found "
                f"{np.unique(y).tolist()}."
            )
        else:
            unique, counts = np.unique(y.astype(int), return_counts=True)
            label_counts = {
                str(int(label)): int(count)
                for label, count in zip(unique, counts)
            }
            record["label_counts"] = label_counts

            expected_counts = {
                str(label): count
                for label, count in EXPECTED_LABEL_COUNTS.items()
            }

            if label_counts != expected_counts:
                record["errors"].append(
                    f"Expected 15 trials per class; found {label_counts}."
                )

        # ----------------------------------------------------
        # Candidate-window bounds check
        # ----------------------------------------------------

        if X.ndim == 3:
            samples = X.shape[0]

            # The original MATLAB indexing is 1-based and inclusive:
            # tmp(fs*13 : fs*23, :, :)
            #
            # Equivalent Python indexing:
            # tmp[fs*13 - 1 : fs*23, :, :]
            start_index = int(round(fs * CANDIDATE_START_SECONDS)) - 1
            end_index_exclusive = int(round(fs * CANDIDATE_END_SECONDS))

            window_fits = (
                start_index >= 0
                and end_index_exclusive > start_index
                and end_index_exclusive <= samples
            )

            record["candidate_window_within_epoch"] = bool(window_fits)

            if window_fits:
                candidate = X[start_index:end_index_exclusive, :, :]

                record["candidate_window_samples"] = int(candidate.shape[0])
                record["candidate_window_finite_fraction"] = float(
                    np.isfinite(candidate).mean()
                )

                if not np.isfinite(candidate).all():
                    record["errors"].append(
                        "Candidate EEG window contains non-finite values."
                    )
            else:
                record["errors"].append(
                    "Candidate 13-23 second window exceeds epoch boundaries."
                )

        # ----------------------------------------------------
        # Timing is not considered verified by structural checks.
        # ----------------------------------------------------

        record["candidate_window_timing_verified"] = False

        if not record["errors"]:
            record["status"] = (
                "CHECK" if record["warnings"] else "PASS"
            )

    except Exception as exc:
        record["errors"].append(
            f"{type(exc).__name__}: {exc}"
        )

    return record


# ============================================================
# 3. FIND AND AUDIT FILES
# ============================================================

def main():
    all_records = []
    group_counts = {}

    for group in ("Normals", "Patients"):
        eeg_dir = DATASET_ROOT / "epo_data" / group / "EEG"

        if not eeg_dir.exists():
            print(f"WARNING: EEG directory not found: {eeg_dir}")
            group_counts[group] = 0
            continue

        files = sorted(eeg_dir.rglob("*_epo.mat"))
        group_counts[group] = len(files)

        print(f"{group}: {len(files)} files found")

        for index, path in enumerate(files, start=1):
            record = audit_one_file(path, group)
            all_records.append(record)

            if index % 10 == 0 or index == len(files):
                print(f"  Processed {index}/{len(files)}")

    # --------------------------------------------------------
    # 4. COUNTS AND SUMMARY
    # --------------------------------------------------------

    status_counts = {
        "PASS": sum(r["status"] == "PASS" for r in all_records),
        "CHECK": sum(r["status"] == "CHECK" for r in all_records),
        "FAILED": sum(r["status"] == "FAILED" for r in all_records),
    }

    error_summary = {}

    for record in all_records:
        for error in record["errors"]:
            error_summary[error] = error_summary.get(error, 0) + 1

    summary = {
        "dataset": str(DATASET_ROOT),
        "files_found": len(all_records),
        "status_counts": status_counts,
        "expected_group_counts": group_counts,
        "label_mapping_from_source": {
            "0": "Left motor imagery",
            "1": "Right motor imagery",
        },
        "expected_sampling_frequency_hz": EXPECTED_FS,
        "expected_channels": EXPECTED_CHANNELS,
        "expected_trials_per_file": EXPECTED_TRIALS,
        "channel_order_from_original_source": CHANNEL_NAMES,
        "candidate_window_seconds": [
            CANDIDATE_START_SECONDS,
            CANDIDATE_END_SECONDS,
        ],
        "window_timing_verified": False,
        "error_summary": error_summary,
        "original_files_modified": False,
        "note": (
            "PASS means structural checks passed. It does not prove "
            "that the candidate window contains motor imagery. "
            "Timing must be verified separately against event markers."
        ),
    }

    # --------------------------------------------------------
    # 5. SAVE CSV AND JSON
    # --------------------------------------------------------

    csv_path = OUTPUT_DIR / "epoch_timing_audit.csv"
    json_path = OUTPUT_DIR / "epoch_timing_audit_summary.json"

    fieldnames = [
        "file",
        "group",
        "status",
        "errors",
        "warnings",
        "eeg_shape",
        "label_shape",
        "sampling_frequency_hz",
        "channel_count",
        "trial_count",
        "label_counts",
        "finite_fraction",
        "candidate_window_start_seconds",
        "candidate_window_end_seconds",
        "candidate_window_samples",
        "candidate_window_finite_fraction",
        "candidate_window_within_epoch",
        "candidate_window_timing_verified",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for record in all_records:
            row = record.copy()
            row["errors"] = " | ".join(row["errors"])
            row["warnings"] = " | ".join(row["warnings"])
            row["eeg_shape"] = json.dumps(row["eeg_shape"])
            row["label_shape"] = json.dumps(row["label_shape"])
            row["label_counts"] = json.dumps(row["label_counts"])
            writer.writerow(row)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(json_safe(summary), file, indent=2)

    # --------------------------------------------------------
    # 6. DISPLAY RESULTS
    # --------------------------------------------------------

    print("\n=== DATASET 3 EEG EPOCH AUDIT ===")
    print(json.dumps(summary, indent=2))

    print("\nReports saved:")
    print(csv_path)
    print(json_path)

    print("\nOriginal dataset files were read only.")

    if status_counts["FAILED"] > 0:
        print(
            "\nACTION REQUIRED: Inspect error_summary and the CSV "
            "before proceeding."
        )
    elif status_counts["CHECK"] > 0:
        print(
            "\nSTRUCTURE CHECK: Review warnings in the CSV. "
            "Do not begin training until these are resolved."
        )
    else:
        print(
            "\nSTRUCTURAL AUDIT PASSED. The EEG window's scientific "
            "timing still needs separate verification."
        )


if __name__ == "__main__":
    main()
