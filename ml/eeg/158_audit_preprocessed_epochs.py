from pathlib import Path
from scipy.io import loadmat
import pandas as pd
import hashlib
import json

ROOTS = [
    Path("data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults 3"),
    Path("data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults-2 2"),
]

CLASSES = ["FF", "FR", "RF", "RR"]
report = {"subjects": {}, "files": [], "warnings": []}
seen = {}

for root in ROOTS:
    for path in sorted(root.glob("S*/EEG/preprocessed/*_preprocessed.mat")):
        subject = path.parts[-4]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()

        if subject in seen:
            if seen[subject]["hash"] == digest:
                report["files"].append({
                    "subject": subject,
                    "path": str(path),
                    "status": "duplicate_content",
                })
                continue

        try:
            mat = loadmat(path)
            counts = {}
            shapes = {}

            for cls in CLASSES:
                if cls not in mat:
                    counts[cls] = 0
                    continue

                arr = mat[cls]
                shapes[cls] = list(arr.shape)

                if arr.ndim != 3 or arr.shape[0] != 31 or arr.shape[1] != 6500:
                    report["warnings"].append(
                        f"{subject}: unexpected shape for {cls}: {arr.shape}"
                    )
                    counts[cls] = None
                else:
                    counts[cls] = int(arr.shape[2])

            label_counts = None
            label_path = None

            # Search both roots for this subject's label spreadsheet.
            for label_root in ROOTS:
                candidate = label_root / subject / f"{subject}_type.xlsx"
                if candidate.exists():
                    label_path = candidate
                    break

            if label_path:
                labels = pd.read_excel(label_path)
                if "Type" in labels.columns:
                    label_counts = {
                        str(k).strip().upper(): int(v)
                        for k, v in labels["Type"].astype(str).str.strip().str.upper()
                        .value_counts().items()
                    }
                    for cls in CLASSES:
                        label_counts.setdefault(cls, 0)
                else:
                    report["warnings"].append(
                        f"{subject}: Type column missing in {label_path}"
                    )
            else:
                report["warnings"].append(
                    f"{subject}: label spreadsheet not found in either root"
                )

            entry = {
                "subject": subject,
                "path": str(path),
                "sha256": digest,
                "epoch_counts": counts,
                "total_retained_epochs": (
                    sum(counts.values()) if all(v is not None for v in counts.values())
                    else None
                ),
                "array_shapes": shapes,
                "label_spreadsheet": str(label_path) if label_path else None,
                "spreadsheet_label_counts": label_counts,
            }

            report["files"].append(entry)
            seen[subject] = {"hash": digest, "path": str(path)}

            # Prefer a valid preprocessed file for each subject.
            report["subjects"][subject] = entry

        except Exception as exc:
            report["warnings"].append(f"{path}: {exc}")

print("\nPREPROCESSED EEG EPOCH AUDIT")
print("=" * 90)

for subject in sorted(report["subjects"]):
    row = report["subjects"][subject]
    print(f"\n{subject}: {row['path']}")
    print(f"  Epochs by class: {row['epoch_counts']}")
    print(f"  Total retained epochs: {row['total_retained_epochs']}")
    print(f"  Spreadsheet counts: {row['spreadsheet_label_counts']}")

    if row["spreadsheet_label_counts"] is None:
        print("  Comparison: UNAVAILABLE — no readable label counts")
    elif all(v is not None for v in row["epoch_counts"].values()):
        matches = row["epoch_counts"] == {
            cls: row["spreadsheet_label_counts"].get(cls, 0)
            for cls in CLASSES
        }
        print(f"  Class-count comparison: {'MATCH' if matches else 'DIFFERENT'}")

print("\nWARNINGS")
for warning in report["warnings"]:
    print(" -", warning)
if not report["warnings"]:
    print(" None")

out = Path("results/eeg/visual_motivation/step158_epoch_audit.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2), encoding="utf-8")

print(f"\nReport saved: {out}")
print("Original EEG files and label spreadsheets were not modified.")
