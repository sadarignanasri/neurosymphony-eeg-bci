from pathlib import Path
import pandas as pd
import json

BASE = Path("results/eeg/visual_motivation")
MANIFEST = BASE / "step171_verified_dataset_manifest.csv"
SPLIT_PLAN = BASE / "step173_subject_wise_split_plan.json"

if not MANIFEST.exists():
    raise FileNotFoundError(MANIFEST)
if not SPLIT_PLAN.exists():
    raise FileNotFoundError(SPLIT_PLAN)

df = pd.read_csv(MANIFEST)
plan = json.loads(SPLIT_PLAN.read_text(encoding="utf-8"))

split_map = {}
for split_name in ["train", "validation", "test"]:
    for subject in plan[f"{split_name}_subjects"]:
        if subject in split_map:
            raise ValueError(f"Subject leakage detected: {subject}")
        split_map[subject] = split_name

if set(df["subject"]) != set(split_map):
    raise ValueError("Manifest subjects do not match split-plan subjects.")

df["split"] = df["subject"].map(split_map)

if df["split"].isna().any():
    raise ValueError("Some manifest rows have no split assignment.")

# Validate that every split contains all four classes.
classes = ["FF", "FR", "RF", "RR"]
for split_name in ["train", "validation", "test"]:
    subset = df[df["split"] == split_name]
    represented = set(subset.loc[subset["epoch_count"] > 0, "label"])
    if represented != set(classes):
        raise ValueError(
            f"{split_name} is missing classes: {set(classes) - represented}"
        )

out_csv = BASE / "step174_split_manifest.csv"
out_json = BASE / "step174_split_summary.json"

df.to_csv(out_csv, index=False)

summary = {
    "step": 174,
    "seed": plan["seed"],
    "split_subjects": {
        split_name: plan[f"{split_name}_subjects"]
        for split_name in ["train", "validation", "test"]
    },
    "split_epoch_counts": {
        split_name: int(df.loc[df["split"] == split_name, "epoch_count"].sum())
        for split_name in ["train", "validation", "test"]
    },
    "class_counts": {
        split_name: {
            label: int(
                df.loc[
                    (df["split"] == split_name) & (df["label"] == label),
                    "epoch_count",
                ].sum()
            )
            for label in classes
        }
        for split_name in ["train", "validation", "test"]
    },
    "subject_overlap": False,
    "all_classes_in_each_split": True,
    "signal_data_transformed": False,
    "models_modified": False,
}

out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("SPLIT MANIFEST VALIDATION: PASS")
print("\nSubject counts:")
for split_name, subjects in summary["split_subjects"].items():
    print(f"{split_name}: {len(subjects)} subjects -> {subjects}")

print("\nEpoch counts:")
for split_name, count in summary["split_epoch_counts"].items():
    print(f"{split_name}: {count}")

print("\nClass counts:")
for split_name, counts in summary["class_counts"].items():
    print(f"{split_name}: {counts}")

print("\nCSV:", out_csv)
print("JSON:", out_json)
print("No source data or existing models modified.")
