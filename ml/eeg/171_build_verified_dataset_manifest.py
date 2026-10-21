from pathlib import Path
from scipy.io import loadmat
import numpy as np
import pandas as pd
import json

ROOTS = [
    Path("data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults 3"),
    Path("data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults-2 2"),
]

SUBJECTS = [
    "S01", "S03", "S05", "S07", "S08",
    "S09", "S10", "S13", "S14", "S16",
]

CLASSES = ["FF", "FR", "RF", "RR"]
EXPECTED_CHANNELS = 31
EXPECTED_TIMEPOINTS = 6500
SAMPLING_RATE_HZ = 500
TIME_START_MS = -1000
TIME_END_MS = 11998

def locate_file(subject):
    relative = (
        Path(subject) / "EEG" / "preprocessed"
        / f"{subject}_preprocessed.mat"
    )
    for root in ROOTS:
        path = root / relative
        if path.exists():
            return path
    return None

manifest = []
class_totals = {label: 0 for label in CLASSES}
subject_totals = {}
seen_subjects = set()

for subject in SUBJECTS:
    if subject in seen_subjects:
        raise ValueError(f"Duplicate subject configured: {subject}")
    seen_subjects.add(subject)

    path = locate_file(subject)
    if path is None:
        raise FileNotFoundError(
            f"Validated preprocessed file not found for {subject}"
        )

    mat = loadmat(path)
    subject_total = 0

    for label in CLASSES:
        if label not in mat:
            raise KeyError(f"{subject}: missing class {label}")

        data = np.asarray(mat[label])

        if data.shape[:2] != (
            EXPECTED_CHANNELS, EXPECTED_TIMEPOINTS
        ) or data.ndim != 3:
            raise ValueError(
                f"{subject}/{label}: unexpected shape {data.shape}"
            )

        if not np.isfinite(data).all():
            raise ValueError(
                f"{subject}/{label}: non-finite signal values"
            )

        epoch_count = int(data.shape[2])
        subject_total += epoch_count
        class_totals[label] += epoch_count

        manifest.append({
            "subject": subject,
            "label": label,
            "label_id": CLASSES.index(label),
            "epoch_count": epoch_count,
            "channels": EXPECTED_CHANNELS,
            "timepoints": EXPECTED_TIMEPOINTS,
            "sampling_rate_hz": SAMPLING_RATE_HZ,
            "time_start_ms": TIME_START_MS,
            "time_end_ms": TIME_END_MS,
            "source_file": str(path),
            "array_key": label,
            "dtype": str(data.dtype),
            "all_values_finite": True,
        })

    subject_totals[subject] = subject_total
    print(f"{subject}: {subject_total} verified epochs")

expected_subject_totals = {
    "S01": 122, "S03": 211, "S05": 208, "S07": 175,
    "S08": 187, "S09": 200, "S10": 194, "S13": 219,
    "S14": 202, "S16": 156,
}

if subject_totals != expected_subject_totals:
    raise ValueError(
        "Subject totals differ from validated Step 168 results.\n"
        f"Observed: {subject_totals}"
    )

out_dir = Path("results/eeg/visual_motivation")
out_dir.mkdir(parents=True, exist_ok=True)

csv_path = out_dir / "step171_verified_dataset_manifest.csv"
json_path = out_dir / "step171_verified_dataset_manifest.json"

pd.DataFrame(manifest).to_csv(csv_path, index=False)

report = {
    "step": 171,
    "dataset": "Visual Cognitive Motivation EEG",
    "subjects": SUBJECTS,
    "number_of_subjects": len(SUBJECTS),
    "number_of_class_arrays": len(manifest),
    "total_epochs": sum(subject_totals.values()),
    "class_totals": class_totals,
    "subject_totals": subject_totals,
    "channels": EXPECTED_CHANNELS,
    "timepoints_per_epoch": EXPECTED_TIMEPOINTS,
    "sampling_rate_hz": SAMPLING_RATE_HZ,
    "time_window_ms": [TIME_START_MS, TIME_END_MS],
    "classes": CLASSES,
    "manifest_csv": str(csv_path),
    "manifest_json": str(json_path),
    "source_data_modified": False,
    "existing_models_modified": False,
}

json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("\nCLASS TOTALS")
for label, count in class_totals.items():
    print(f"{label}: {count}")

print(f"\nTotal verified epochs: {report['total_epochs']}")
print(f"Manifest CSV: {csv_path}")
print(f"Manifest JSON: {json_path}")
print("Validation: PASS")
print("Original data and existing models were not modified.")
