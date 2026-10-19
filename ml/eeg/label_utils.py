from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_LABELS = [
    "animal_tactile_imagery_task",
    "animal_visual_imagery_task",
    "animal_auditory_imagery_task",
    "animal_silent_naming_task",
    "tool_tactile_imagery_task",
    "tool_visual_imagery_task",
    "tool_auditory_imagery_task",
    "tool_silent_naming_task",
]

LABEL_TO_ID = {name: i for i, name in enumerate(EXPECTED_LABELS)}

def load_subject_labels(subject):
    path = (
        PROJECT_ROOT / "data" / "features"
        / f"temporal_eeg_{subject}"
        / f"{subject}_temporal_eeg_preprocessed_metadata.csv"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing trial metadata: {path}")

    df = pd.read_csv(path)
    required = {"trial_index", "label_id", "label_name"}
    if not required.issubset(df.columns):
        raise ValueError(f"{subject}: metadata columns missing: "
                         f"{sorted(required - set(df.columns))}")

    if len(df) != 720:
        raise ValueError(f"{subject}: expected 720 trials, found {len(df)}")

    if not np.array_equal(df["trial_index"].to_numpy(), np.arange(720)):
        raise ValueError(f"{subject}: trial indices are not in original order")

    expected_ids = df["label_name"].map(LABEL_TO_ID)
    if expected_ids.isna().any():
        raise ValueError(f"{subject}: unknown label name found")

    actual_ids = df["label_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(actual_ids, expected_ids.to_numpy(dtype=np.int64)):
        raise ValueError(f"{subject}: label IDs disagree with canonical labels")

    counts = df["label_name"].value_counts()
    if any(int(counts.get(name, 0)) != 90 for name in EXPECTED_LABELS):
        raise ValueError(f"{subject}: each class must contain exactly 90 trials")

    return actual_ids
