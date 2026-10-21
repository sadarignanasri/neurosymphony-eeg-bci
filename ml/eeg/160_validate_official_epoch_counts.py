from pathlib import Path
from scipy.io import loadmat
import pandas as pd
import json

ROOTS = [
    Path("data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults 3"),
    Path("data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults-2 2"),
]

CLASSES = ["FF", "FR", "RF", "RR"]

summary_path = next(
    (root / "Total_epoch.xlsx" for root in ROOTS
     if (root / "Total_epoch.xlsx").exists()),
    None,
)

if summary_path is None:
    raise FileNotFoundError("Total_epoch.xlsx not found.")

summary = pd.read_excel(summary_path)
summary.columns = [str(c).strip() for c in summary.columns]
summary = summary.rename(columns={summary.columns[0]: "Subject"})
summary["Subject"] = summary["Subject"].astype(str).str.strip()

results = []
errors = []

for _, row in summary.iterrows():
    subject = row["Subject"]
    if subject.lower() == "total":
        continue

    expected = {cls: int(row[cls]) for cls in CLASSES}
    candidates = []

    for root in ROOTS:
        candidates.extend(
            root.glob(f"{subject}/EEG/preprocessed/{subject}_preprocessed.mat")
        )

    if not candidates:
        errors.append(f"{subject}: no preprocessed MATLAB file found")
        results.append({
            "subject": subject,
            "status": "MISSING_FILE",
            "expected": expected,
            "observed": None,
        })
        continue

    observed = None
    selected = None

    for path in candidates:
        mat = loadmat(path)
        counts = {
            cls: int(mat[cls].shape[2]) if cls in mat and mat[cls].ndim == 3 else None
            for cls in CLASSES
        }

        if all(v is not None for v in counts.values()):
            observed = counts
            selected = str(path)
            break

    if observed is None:
        errors.append(f"{subject}: no valid class arrays found")
        status = "INVALID_ARRAYS"
    else:
        status = "PASS" if observed == expected else "COUNT_MISMATCH"
        if status != "PASS":
            errors.append(
                f"{subject}: expected {expected}, observed {observed}"
            )

    results.append({
        "subject": subject,
        "status": status,
        "file": selected,
        "expected": expected,
        "observed": observed,
        "total_epochs": sum(observed.values()) if observed else None,
    })

print("\nSTEP 160 — OFFICIAL EPOCH COUNT VALIDATION")
print("=" * 90)

for item in results:
    print(
        f"{item['subject']}: {item['status']} | "
        f"expected={item['expected']} | observed={item['observed']}"
    )

print("\nSummary:")
print("Subjects checked:", len(results))
print("Passed:", sum(r["status"] == "PASS" for r in results))
print("Failed:", sum(r["status"] != "PASS" for r in results))
print("Total validated epochs:", sum(
    r["total_epochs"] or 0 for r in results
))

if errors:
    print("\nIssues:")
    for error in errors:
        print("-", error)
else:
    print("All subjects match the official epoch counts.")

out = Path("results/eeg/visual_motivation/step160_epoch_validation.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(results, indent=2), encoding="utf-8")
print("\nReport saved:", out)
print("No original data or models were modified.")
