from pathlib import Path
from scipy.io import loadmat
import pandas as pd
import numpy as np
import json

BASE = Path("results/eeg/visual_motivation")
MANIFEST = BASE / "step174_split_manifest.csv"

ROOTS = [
    Path("data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults 3"),
    Path("data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults-2 2"),
]

CLASSES = ["FF", "FR", "RF", "RR"]
EXPECTED_SHAPE = (31, 6500)

if not MANIFEST.exists():
    raise FileNotFoundError(MANIFEST)

manifest = pd.read_csv(MANIFEST)

def locate_file(subject):
    rel = (
        Path(subject) / "EEG" / "preprocessed"
        / f"{subject}_preprocessed.mat"
    )
    for root in ROOTS:
        path = root / rel
        if path.exists():
            return path
    return None

report = {
    "step": 184,
    "subjects": {},
    "total_epochs": 0,
    "split_totals": {},
    "classes": CLASSES,
    "all_arrays_valid": True,
    "source_data_modified": False,
    "models_modified": False,
}

for subject in sorted(manifest["subject"].unique()):
    subject_rows = manifest[manifest["subject"] == subject]
    splits = subject_rows["split"].unique().tolist()

    if len(splits) != 1:
        raise ValueError(f"{subject} has multiple split assignments.")

    split = splits[0]
    path = locate_file(subject)
    if path is None:
        raise FileNotFoundError(f"Preprocessed file missing: {subject}")

    # Load one subject at a time to keep memory use controlled.
    mat = loadmat(path)
    class_counts = {}
    subject_total = 0

    for label in CLASSES:
        if label not in mat:
            raise KeyError(f"{subject}: missing array {label}")

        array = mat[label]
        if array.ndim != 3 or array.shape[:2] != EXPECTED_SHAPE:
            raise ValueError(
                f"{subject}/{label}: invalid shape {array.shape}"
            )

        if not np.isfinite(array).all():
            raise ValueError(f"{subject}/{label}: non-finite values")

        count = int(array.shape[2])
        expected_count = int(
            subject_rows.loc[
                subject_rows["label"] == label, "epoch_count"
            ].iloc[0]
        )

        if count != expected_count:
            raise ValueError(
                f"{subject}/{label}: manifest says {expected_count}, "
                f"array contains {count}"
            )

        class_counts[label] = count
        subject_total += count

    report["subjects"][subject] = {
        "split": split,
        "class_counts": class_counts,
        "total_epochs": subject_total,
        "source_file": str(path),
    }
    report["total_epochs"] += subject_total
    report["split_totals"][split] = (
        report["split_totals"].get(split, 0) + subject_total
    )

    print(
        f"{subject}: PASS | split={split} | "
        f"epochs={subject_total} | classes={class_counts}"
    )

if report["total_epochs"] != 1874:
    raise ValueError(
        f"Expected 1874 epochs; found {report['total_epochs']}"
    )

out = BASE / "step184_dataset_loader_validation.json"
out.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("\nDATASET LOADER VALIDATION: PASS")
print("Total epochs:", report["total_epochs"])
print("Split totals:", report["split_totals"])
print("Report saved:", out)
print("No large tensors saved.")
print("No source data or existing models modified.")
