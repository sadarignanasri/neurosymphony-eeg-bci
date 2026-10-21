from pathlib import Path
import pandas as pd
import json

ROOTS = [
    Path("data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults 3"),
    Path("data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults-2 2"),
]

MANIFEST = Path(
    "results/eeg/visual_motivation/step171_verified_dataset_manifest.csv"
)

if not MANIFEST.exists():
    raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

official_file = next(
    (root / "Total_epoch.xlsx"
     for root in ROOTS
     if (root / "Total_epoch.xlsx").exists()),
    None,
)

if official_file is None:
    raise FileNotFoundError("Official Total_epoch.xlsx not found.")

manifest = pd.read_csv(MANIFEST)
official = pd.read_excel(official_file)
official.columns = [str(c).strip() for c in official.columns]
official = official.rename(columns={official.columns[0]: "Subject"})

subjects = sorted(manifest["subject"].unique())
classes = ["FF", "FR", "RF", "RR"]

rows = []
for subject in subjects:
    official_rows = official[
        official["Subject"].astype(str).str.strip() == subject
    ]
    if official_rows.empty:
        raise ValueError(f"{subject} missing from official count sheet")

    official_row = official_rows.iloc[0]

    for label in classes:
        observed = int(
            manifest.loc[
                (manifest["subject"] == subject)
                & (manifest["label"] == label),
                "epoch_count",
            ].sum()
        )
        expected = int(official_row[label])
        rows.append({
            "subject": subject,
            "class": label,
            "official_count": expected,
            "manifest_count": observed,
            "difference": observed - expected,
            "match": observed == expected,
        })

comparison = pd.DataFrame(rows)
print("\nSUBJECT-CLASS COMPARISON")
print(comparison.to_string(index=False))

print("\nCLASS TOTALS FOR INCLUDED SUBJECTS")
print(
    comparison.groupby("class")[
        ["official_count", "manifest_count", "difference"]
    ].sum().to_string()
)

print("\nAll subject-class counts match:", bool(comparison["match"].all()))

out = Path(
    "results/eeg/visual_motivation/step172_count_crosscheck.csv"
)
out.parent.mkdir(parents=True, exist_ok=True)
comparison.to_csv(out, index=False)
print("Report saved:", out)
print("No source data or existing models modified.")
