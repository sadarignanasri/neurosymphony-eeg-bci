
from pathlib import Path
import json
import csv
import sys

import h5py
import numpy as np
from scipy.io import loadmat

ROOT = Path(
    r"D:\neurosymphony-eeg-bci\major datasets\28955456"
)

OUTPUT = Path(
    r"D:\neurosymphony-eeg-bci\experiments"
    r"\dataset3_28955456_audit"
)

# Only inspect EEG files. Original data will never be modified.
EEG_AREAS = [
    ("source_data", "Normals"),
    ("source_data", "Patients"),
    ("epo_data", "Normals"),
    ("epo_data", "Patients"),
]

def describe_hdf5(path):
    result = {"format": "HDF5", "datasets": [], "attributes": {}}

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            result["datasets"].append({
                "name": name,
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "attributes": {
                    str(k): str(v)
                    for k, v in obj.attrs.items()
                },
            })

    with h5py.File(path, "r") as f:
        result["top_level_keys"] = list(f.keys())
        result["attributes"] = {
            str(k): str(v) for k, v in f.attrs.items()
        }
        f.visititems(visitor)

    return result


def describe_mat(path):
    try:
        data = loadmat(path, squeeze_me=False)
        variables = []

        for name, value in data.items():
            if name.startswith("__"):
                continue

            variables.append({
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            })

        return {
            "format": "MATLAB MAT",
            "variables": variables,
            "reader": "scipy.io.loadmat",
        }

    except (NotImplementedError, ValueError, OSError):
        # MATLAB v7.3 files use the HDF5 format.
        result = describe_hdf5(path)
        result["format"] = "MATLAB v7.3 / HDF5"
        return result


def main():
    if not ROOT.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {ROOT}")

    OUTPUT.mkdir(parents=True, exist_ok=True)

    records = []
    subject_summary = []

    for area, group in EEG_AREAS:
        eeg_root = ROOT / area / group / "EEG"

        if not eeg_root.exists():
            print(f"Missing directory: {eeg_root}")
            continue

        subjects = sorted(
            [p for p in eeg_root.iterdir() if p.is_dir()],
            key=lambda p: int(
                "".join(c for c in p.name if c.isdigit()) or 0
            ),
        )

        print(f"\n{area}/{group}: {len(subjects)} subject folders")

        for subject_dir in subjects:
            files = sorted(
                p for p in subject_dir.iterdir()
                if p.is_file() and p.suffix.lower() in {".hdf5", ".mat"}
            )

            subject_summary.append({
                "area": area,
                "group": group,
                "subject": subject_dir.name,
                "recording_file_count": len(files),
                "files": "; ".join(p.name for p in files),
            })

            for path in files:
                row = {
                    "area": area,
                    "group": group,
                    "subject": subject_dir.name,
                    "file": str(path),
                    "extension": path.suffix.lower(),
                    "size_bytes": path.stat().st_size,
                    "inspection": {},
                    "error": None,
                }

                try:
                    if path.suffix.lower() == ".hdf5":
                        row["inspection"] = describe_hdf5(path)
                    else:
                        row["inspection"] = describe_mat(path)

                    print(
                        f"  OK {group}/{subject_dir.name}/{path.name}"
                    )

                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    print(f"  ERROR {path}: {row['error']}")

                records.append(row)

    # Detailed metadata report
    report_path = OUTPUT / "dataset_structure.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, default=str)

    # Flat file inventory
    inventory_path = OUTPUT / "file_inventory.csv"
    with inventory_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "area", "group", "subject", "file",
                "extension", "size_bytes", "error",
            ],
        )
        writer.writeheader()

        for row in records:
            writer.writerow({
                key: row[key]
                for key in writer.fieldnames
            })

    # Subject-level inventory
    subjects_path = OUTPUT / "subject_inventory.csv"
    with subjects_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        fields = [
            "area", "group", "subject",
            "recording_file_count", "files",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(subject_summary)

    summary = {
        "dataset_root": str(ROOT),
        "original_files_modified": False,
        "eeg_files_inspected": len(records),
        "successful_inspections": sum(
            row["error"] is None for row in records
        ),
        "failed_inspections": sum(
            row["error"] is not None for row in records
        ),
        "subject_folder_count": len(subject_summary),
        "outputs": [
            str(report_path),
            str(inventory_path),
            str(subjects_path),
        ],
    }

    with (OUTPUT / "audit_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)

    print("\n=== DATASET 3 AUDIT SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print("\nAudit complete. Original data was only read.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nAUDIT FAILED: {type(exc).__name__}: {exc}")
        sys.exit(1)
