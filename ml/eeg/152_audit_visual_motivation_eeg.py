from pathlib import Path
from scipy.io import loadmat
import pandas as pd
import numpy as np
import hashlib
import json

ROOTS = [
    Path("data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults 3"),
    Path("data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults-2 2"),
]
CLASSES = ("FF", "FR", "RF", "RR")
EXPECTED_CHANNELS = 31
EXPECTED_SAMPLES = 6500

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

report = {
    "purpose": "EEG-only preprocessed epoch audit",
    "expected_shape": [EXPECTED_CHANNELS, EXPECTED_SAMPLES, "epochs"],
    "datasets": [],
    "duplicate_files": [],
    "warnings": [],
}

hash_index = {}

for root in ROOTS:
    if not root.exists():
        report["warnings"].append(f"Dataset folder not found: {root}")
        continue

    dataset_report = {
        "folder": str(root),
        "subjects": [],
        "spreadsheet_label_counts": {},
    }

    for label_file in sorted(root.glob("S*/S*_type.xlsx")):
        try:
            df = pd.read_excel(label_file)
            if "Type" not in df.columns:
                dataset_report["spreadsheet_label_counts"][label_file.parent.name] = {
                    "error": "Missing Type column"
                }
                continue

            labels = df["Type"].astype(str).str.strip()
            counts = labels.value_counts().to_dict()
            unexpected = {
                label: count for label, count in counts.items()
                if label not in CLASSES
            }

            dataset_report["spreadsheet_label_counts"][label_file.parent.name] = {
                "rows": int(len(df)),
                "classes": {str(k): int(v) for k, v in counts.items()},
                "unexpected_labels": unexpected,
            }
        except Exception as exc:
            dataset_report["spreadsheet_label_counts"][label_file.parent.name] = {
                "error": str(exc)
            }

    for mat_path in sorted(root.glob("S*/EEG/preprocessed/*_preprocessed.mat")):
        subject = mat_path.parents[2].name
        subject_report = {
            "subject": subject,
            "mat_file": str(mat_path),
            "sha256": sha256(mat_path),
            "class_epochs": {},
            "class_shapes": {},
            "invalid_values": {},
            "total_epochs": 0,
            "valid_for_expected_shape": True,
        }

        digest = subject_report["sha256"]
        if digest in hash_index:
            previous = hash_index[digest]
            report["duplicate_files"].append({
                "first": previous,
                "duplicate": str(mat_path),
                "sha256": digest,
            })
        else:
            hash_index[digest] = str(mat_path)

        try:
            data = loadmat(mat_path)
            for class_name in CLASSES:
                if class_name not in data:
                    subject_report["class_epochs"][class_name] = 0
                    continue

                arr = np.asarray(data[class_name])
                subject_report["class_shapes"][class_name] = list(arr.shape)

                if arr.ndim != 3:
                    subject_report["valid_for_expected_shape"] = False
                    subject_report["invalid_values"][class_name] = {
                        "error": f"Expected 3D array, got {arr.ndim}D"
                    }
                    continue

                channels, samples, epochs = arr.shape
                subject_report["class_epochs"][class_name] = int(epochs)
                subject_report["total_epochs"] += int(epochs)

                bad_shape = (
                    channels != EXPECTED_CHANNELS
                    or samples != EXPECTED_SAMPLES
                )
                if bad_shape:
                    subject_report["valid_for_expected_shape"] = False

                finite = np.isfinite(arr)
                subject_report["invalid_values"][class_name] = {
                    "non_finite_values": int(arr.size - finite.sum()),
                    "expected_shape_ok": not bad_shape,
                }

        except Exception as exc:
            subject_report["error"] = str(exc)
            subject_report["valid_for_expected_shape"] = False

        dataset_report["subjects"].append(subject_report)

    report["datasets"].append(dataset_report)

out_dir = Path("results/eeg/dataset_inventory")
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / "step152_eeg_audit.json"
out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("\n=== STEP 152: EEG PREPROCESSED DATA AUDIT ===")
for dataset in report["datasets"]:
    print(f"\nDATASET: {Path(dataset['folder']).name}")
    print("Preprocessed subjects:", len(dataset["subjects"]))
    for s in dataset["subjects"]:
        print(
            f"  {s['subject']}: epochs={s['class_epochs']}, "
            f"total={s['total_epochs']}, "
            f"shape_valid={s['valid_for_expected_shape']}"
        )
    print("Spreadsheet subjects:", len(dataset["spreadsheet_label_counts"]))

print("\nExact duplicate MATLAB files:", len(report["duplicate_files"]))
for duplicate in report["duplicate_files"]:
    print("  FIRST:", duplicate["first"])
    print("  DUPLICATE:", duplicate["duplicate"])

print("\nWarnings:", report["warnings"] or "None")
print("Report saved to:", out_file)
print("No dataset files or existing models were modified.")
