from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np
import json

ROOTS = [
    Path("data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults 3"),
    Path("data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults-2 2"),
]

report = {"subjects": [], "warnings": []}
subject_parts = {}

for root in ROOTS:
    if not root.exists():
        continue

    for annotation_file in sorted(root.glob("S*/EEG/raw/*_annotations.txt")):
        subject_dir = annotation_file.parents[2]
        subject = subject_dir.name
        name = annotation_file.name

        try:
            df = pd.read_csv(
                annotation_file,
                sep=r"\s+",
                engine="python",
                comment="#",
            )
            if "Onset" not in df.columns:
                report["warnings"].append(
                    f"Missing Onset column: {annotation_file}"
                )
                continue

            times = pd.to_numeric(df["Onset"], errors="coerce").dropna()
            original_count = int(len(times))
            # For diagnostics only: exact repeated timestamps are counted once.
            unique_times = np.sort(times.unique())
            unique_count = int(len(unique_times))
            deltas = np.diff(unique_times)

            part_report = {
                "subject": subject,
                "file": str(annotation_file),
                "original_event_count": original_count,
                "unique_timestamp_count": unique_count,
                "duplicate_timestamp_count": original_count - unique_count,
                "count_remainder_mod_3": unique_count % 3,
                "first_12_unique_onsets_seconds": [
                    round(float(x), 3) for x in unique_times[:12]
                ],
                "first_12_inter_event_gaps_seconds": [
                    round(float(x), 3) for x in deltas[:12]
                ],
                "tentative_complete_triplets": unique_count // 3,
            }

            subject_parts.setdefault(subject, []).append(part_report)
            print(
                f"{subject} | {name} | raw={original_count} | "
                f"unique={unique_count} | duplicates="
                f"{original_count - unique_count} | "
                f"remainder={unique_count % 3}"
            )
        except Exception as exc:
            report["warnings"].append(f"{annotation_file}: {exc}")

# Compare with label spreadsheets; this does not assume a trigger-to-label map.
for subject, parts in sorted(subject_parts.items()):
    label_files = []
    for root in ROOTS:
        p = root / subject / f"{subject}_type.xlsx"
        if p.exists():
            label_files.append(p)

    label_count = None
    label_classes = {}
    if label_files:
        try:
            labels = pd.read_excel(label_files[0])
            if "Type" in labels.columns:
                valid = labels["Type"].astype(str).str.strip()
                label_count = int(len(labels))
                label_classes = {
                    str(k): int(v)
                    for k, v in valid.value_counts().to_dict().items()
                }
        except Exception as exc:
            report["warnings"].append(f"Label read error for {subject}: {exc}")

    summary = {
        "subject": subject,
        "parts": parts,
        "label_file": str(label_files[0]) if label_files else None,
        "spreadsheet_rows": label_count,
        "spreadsheet_class_counts": label_classes,
        "sum_unique_events": sum(p["unique_timestamp_count"] for p in parts),
        "sum_tentative_triplets": sum(
            p["tentative_complete_triplets"] for p in parts
        ),
    }
    report["subjects"].append(summary)

    print(
        f"  SUBJECT SUMMARY {subject}: "
        f"unique events={summary['sum_unique_events']}, "
        f"tentative triplets={summary['sum_tentative_triplets']}, "
        f"label rows={label_count}"
    )

out = Path("results/eeg/visual_motivation/step156_trial_event_audit.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("\nWarnings:", report["warnings"] or "None")
print("Report saved:", out)
print("No EEG signals, annotations, or labels were modified.")
