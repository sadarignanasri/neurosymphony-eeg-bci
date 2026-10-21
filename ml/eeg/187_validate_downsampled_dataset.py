from pathlib import Path
import numpy as np
import pandas as pd
import json

ROOT = Path("data/processed/visual_motivation_125hz")
MANIFEST = ROOT / "dataset_manifest.csv"
CLASSES = ["FF", "FR", "RF", "RR"]

if not MANIFEST.exists():
    raise FileNotFoundError(MANIFEST)

df = pd.read_csv(MANIFEST)

required = {
    "subject", "split", "label", "label_id",
    "epoch_count", "output_file",
    "channels", "timepoints", "sampling_rate_hz",
}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Manifest columns missing: {sorted(missing)}")

if len(df) != 40:
    raise ValueError(f"Expected 40 subject-class entries; found {len(df)}")

if df.duplicated(["subject", "label"]).any():
    raise ValueError("Duplicate subject/class manifest entries")

if set(df["label"]) != set(CLASSES):
    raise ValueError("Unexpected class labels")

if df["split"].isna().any():
    raise ValueError("Missing split assignment")

report = {
    "step": 187,
    "entries_checked": 0,
    "total_epochs": 0,
    "split_totals": {},
    "class_totals": {},
    "all_arrays_valid": True,
}

for row in df.itertuples(index=False):
    path = Path(row.output_file)
    if not path.exists():
        raise FileNotFoundError(f"Missing saved array: {path}")

    arr = np.load(path, allow_pickle=False, mmap_mode="r")
    expected_count = int(row.epoch_count)
    expected_shape = (expected_count, 31, 1625)

    if arr.shape != expected_shape:
        raise ValueError(
            f"{path}: shape {arr.shape}, expected {expected_shape}"
        )

    if arr.dtype != np.float32:
        raise ValueError(f"{path}: expected float32, found {arr.dtype}")

    if not np.isfinite(arr).all():
        raise ValueError(f"{path}: non-finite values found")

    if int(row.label_id) != CLASSES.index(row.label):
        raise ValueError(f"{path}: inconsistent label ID")

    report["entries_checked"] += 1
    report["total_epochs"] += expected_count
    report["split_totals"][row.split] = (
        report["split_totals"].get(row.split, 0) + expected_count
    )
    report["class_totals"][row.label] = (
        report["class_totals"].get(row.label, 0) + expected_count
    )

    print(
        f"PASS {row.subject} {row.label} | "
        f"{arr.shape} | {row.split}"
    )

if report["total_epochs"] != 1874:
    raise ValueError(f"Expected 1874 epochs, found {report['total_epochs']}")

expected_splits = {
    "train": 1118,
    "validation": 343,
    "test": 413,
}
if report["split_totals"] != expected_splits:
    raise ValueError(
        f"Unexpected split totals: {report['split_totals']}"
    )

out = Path("results/eeg/visual_motivation/step187_dataset_validation.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("\nSAVED DATASET VALIDATION: PASS")
print("Entries checked:", report["entries_checked"])
print("Total epochs:", report["total_epochs"])
print("Split totals:", report["split_totals"])
print("Class totals:", report["class_totals"])
print("Report saved:", out)
print("No source data or models modified.")
