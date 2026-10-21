
"""
NeuroSymphony — Step 150: EEG Dataset Inventory

Inspects two visual cognitive motivation EEG datasets.
EEG files only: EDF and MATLAB files under subject EEG folders.
Does not load fNIRS signals or modify source recordings.
"""

from pathlib import Path
import json
import warnings

import numpy as np
from scipy.io import whosmat

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "results" / "eeg" / "dataset_inventory"

DATASETS = {
    "dataset_1": (
        DATA_ROOT
        / "A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults 3"
    ),
    "dataset_2": (
        DATA_ROOT
        / "A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults-2 2"
    ),
}

EEG_EXTENSIONS = {".edf", ".bdf", ".set", ".vhdr", ".cnt", ".mat"}
SUBJECT_PATTERN = "S*"


def inspect_mat_file(path):
    """Report MATLAB variable names, shapes, and data types when available."""
    result = {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "format": "MATLAB",
        "variables": [],
        "inspection_error": None,
    }

    try:
        variables = whosmat(str(path))
        result["variables"] = [
            {
                "name": name,
                "shape": list(shape),
                "matlab_class": matlab_class,
            }
            for name, shape, matlab_class in variables
        ]
    except Exception as exc:
        result["inspection_error"] = (
            f"{type(exc).__name__}: {exc}"
        )

    return result


def inspect_edf_file(path):
    """Inspect EDF metadata without reading signal samples."""
    result = {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "format": "EDF",
        "inspection_error": None,
    }

    try:
        import mne

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_edf(
                str(path),
                preload=False,
                verbose="ERROR",
            )

        result.update({
            "sampling_frequency_hz": float(raw.info["sfreq"]),
            "number_of_channels": len(raw.ch_names),
            "channel_names": raw.ch_names,
            "duration_seconds": float(raw.n_times / raw.info["sfreq"]),
            "number_of_samples": int(raw.n_times),
        })
        raw.close()

    except Exception as exc:
        result["inspection_error"] = (
            f"{type(exc).__name__}: {exc}"
        )

    return result


def inspect_dataset(dataset_name, root):
    print("\n" + "=" * 78)
    print(dataset_name)
    print("=" * 78)

    if not root.is_dir():
        print(f"ERROR: Dataset directory not found: {root}")
        return {
            "dataset": dataset_name,
            "dataset_path": str(root),
            "exists": False,
            "subjects": [],
            "files": [],
        }

    subjects = sorted(
        p for p in root.glob(SUBJECT_PATTERN)
        if p.is_dir() and p.name.upper().startswith("S")
    )

    records = []
    subject_summary = []

    for subject_dir in subjects:
        eeg_dir = subject_dir / "EEG"
        if not eeg_dir.is_dir():
            continue

        subject_files = sorted(
            p for p in eeg_dir.rglob("*")
            if p.is_file()
            and p.suffix.lower() in EEG_EXTENSIONS
        )

        subject_summary.append({
            "subject": subject_dir.name,
            "eeg_file_count": len(subject_files),
            "edf_count": sum(
                p.suffix.lower() == ".edf" for p in subject_files
            ),
            "mat_count": sum(
                p.suffix.lower() == ".mat" for p in subject_files
            ),
        })

        for path in subject_files:
            if path.suffix.lower() == ".mat":
                record = inspect_mat_file(path)
            elif path.suffix.lower() == ".edf":
                record = inspect_edf_file(path)
            else:
                record = {
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "format": path.suffix.lower(),
                    "inspection_note": (
                        "File discovered; format-specific inspection "
                        "not implemented in this inventory step."
                    ),
                }

            record["subject"] = subject_dir.name
            records.append(record)

            print(f"\nSubject: {subject_dir.name}")
            print(f"File: {path.name}")
            if "variables" in record:
                for variable in record["variables"]:
                    print(
                        f"  Variable: {variable['name']}, "
                        f"shape={variable['shape']}, "
                        f"class={variable['matlab_class']}"
                    )
                if record["inspection_error"]:
                    print("  MATLAB inspection error:",
                          record["inspection_error"])
            elif record["format"] == "EDF":
                if record["inspection_error"]:
                    print("  EDF inspection error:",
                          record["inspection_error"])
                else:
                    print(
                        f"  Sampling rate: "
                        f"{record['sampling_frequency_hz']} Hz"
                    )
                    print(
                        f"  Channels: {record['number_of_channels']}"
                    )
                    print(
                        f"  Duration: {record['duration_seconds']:.2f} s"
                    )

    summary = {
        "dataset": dataset_name,
        "dataset_path": str(root),
        "exists": True,
        "subject_folder_count": len(subjects),
        "subjects_with_eeg_files": len(subject_summary),
        "total_eeg_files": len(records),
        "total_edf_files": sum(
            r["format"] == "EDF" for r in records
        ),
        "total_mat_files": sum(
            r["format"] == "MATLAB" for r in records
        ),
        "subjects": subject_summary,
        "files": records,
    }

    print("\nSubject summary:")
    for item in subject_summary:
        print(
            f"  {item['subject']}: "
            f"{item['eeg_file_count']} EEG files, "
            f"{item['edf_count']} EDF, "
            f"{item['mat_count']} MATLAB"
        )

    return summary


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for name, root in DATASETS.items():
        all_results[name] = inspect_dataset(name, root)

    report_path = OUTPUT_DIR / "step150_eeg_inventory.json"
    report_path.write_text(
        json.dumps(all_results, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print("INVENTORY COMPLETE")
    print(f"Report saved to: {report_path}")
    print("Only EEG subdirectories were inspected.")
    print("No original data was modified.")
    print("=" * 78)


if __name__ == "__main__":
    main()
