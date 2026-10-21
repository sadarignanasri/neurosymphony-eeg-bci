from pathlib import Path
import pandas as pd
import json
import numpy as np

MANIFEST = Path(
    "results/eeg/visual_motivation/step171_verified_dataset_manifest.csv"
)

if not MANIFEST.exists():
    raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

df = pd.read_csv(MANIFEST)

EXPECTED_SUBJECTS = {
    "S01", "S03", "S05", "S07", "S08",
    "S09", "S10", "S13", "S14", "S16",
}
CLASSES = ["FF", "FR", "RF", "RR"]
SEED = 42

subjects = sorted(df["subject"].unique().tolist())

if set(subjects) != EXPECTED_SUBJECTS:
    raise ValueError(f"Unexpected subjects in manifest: {subjects}")

# Shuffle subjects only; epochs are never individually randomized across splits.
rng = np.random.default_rng(SEED)
shuffled = list(np.array(subjects)[rng.permutation(len(subjects))])

train_subjects = sorted(shuffled[:6])
validation_subjects = sorted(shuffled[6:8])
test_subjects = sorted(shuffled[8:10])

splits = {
    "train": train_subjects,
    "validation": validation_subjects,
    "test": test_subjects,
}

# Confirm that every subject belongs to exactly one split.
all_assigned = (
    train_subjects + validation_subjects + test_subjects
)
if len(all_assigned) != len(set(all_assigned)):
    raise ValueError("Subject leakage: a subject appears in multiple splits.")

if set(all_assigned) != EXPECTED_SUBJECTS:
    raise ValueError("Split does not include exactly the verified subjects.")

print("SUBJECT ASSIGNMENT")
for split_name, split_subjects in splits.items():
    split_df = df[df["subject"].isin(split_subjects)]
    counts = (
        split_df.groupby("label")["epoch_count"].sum()
        .reindex(CLASSES, fill_value=0)
        .astype(int)
        .to_dict()
    )
    print(f"\n{split_name.upper()}")
    print("Subjects:", split_subjects)
    print("Epochs:", int(split_df["epoch_count"].sum()))
    print("Class counts:", counts)
    print("Classes represented:", [
        label for label, count in counts.items() if count > 0
    ])

# Record the plan only; do not load or transform signal arrays here.
out_dir = Path("results/eeg/visual_motivation")
out_dir.mkdir(parents=True, exist_ok=True)

report = {
    "step": 173,
    "seed": SEED,
    "strategy": "subject-wise holdout",
    "train_subjects": train_subjects,
    "validation_subjects": validation_subjects,
    "test_subjects": test_subjects,
    "class_order": CLASSES,
    "subject_leakage": False,
    "data_transformed": False,
    "models_modified": False,
}

out = out_dir / "step173_subject_wise_split_plan.json"
out.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("\nSplit plan saved:", out)
print("No signal data or existing models modified.")
