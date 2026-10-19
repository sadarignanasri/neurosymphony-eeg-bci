"""
Neuro-Symphony
============================================================
DS004514 EEG Feature Engineering
============================================================

Purpose:
    Create trial-level spatial/statistical EEG features from
    the approved DS004514 CSV.

IMPORTANT:
    Each row represents one subject/trial observation.
    A1-B32 are EEG channels, NOT temporal samples.

Therefore:
    We DO NOT calculate FFT, PSD, Delta, Theta, Alpha, Beta,
    Gamma, etc. from this CSV.

Feature groups:
    1. Raw EEG channel values
    2. Global trial statistics
    3. A-group statistics
    4. B-group statistics
    5. A-vs-B statistics
    6. Per-trial channel normalization

The original CSV is never modified.
"""

# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ds004514_eeg_approved.csv"
)

FEATURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "eeg"
)

FEATURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. COLUMN CONFIGURATION
# ============================================================

METADATA_COLUMNS = [
    "subject_id",
    "trial_id"
]

LABEL_NAME_COLUMN = "label_name"

TARGET_COLUMN = "label"

A_CHANNELS = [
    f"A{i}"
    for i in range(1, 33)
]

B_CHANNELS = [
    f"B{i}"
    for i in range(1, 33)
]

EEG_CHANNELS = (
    A_CHANNELS
    +
    B_CHANNELS
)


# ============================================================
# 3. HELPER FUNCTION
# ============================================================

def add_statistics(
    output_df,
    values,
    prefix
):
    """
    Add statistical features to output dataframe.
    """

    output_df[f"{prefix}_mean"] = values.mean(axis=1)

    output_df[f"{prefix}_std"] = values.std(
        axis=1
    )

    output_df[f"{prefix}_median"] = values.median(
        axis=1
    )

    output_df[f"{prefix}_min"] = values.min(
        axis=1
    )

    output_df[f"{prefix}_max"] = values.max(
        axis=1
    )

    output_df[f"{prefix}_range"] = (
        values.max(axis=1)
        -
        values.min(axis=1)
    )

    output_df[f"{prefix}_rms"] = np.sqrt(
        (values ** 2).mean(axis=1)
    )

    return output_df


# ============================================================
# 4. START
# ============================================================

print("=" * 75)
print("NEURO-SYMPHONY")
print("DS004514 EEG FEATURE ENGINEERING")
print("=" * 75)


# ============================================================
# 5. LOAD DATASET
# ============================================================

print("\nLoading approved EEG CSV...")

if not CSV_PATH.exists():

    raise FileNotFoundError(
        f"Dataset not found:\n{CSV_PATH}"
    )

df = pd.read_csv(
    CSV_PATH
)

print("✓ Dataset loaded successfully.")

print(
    f"Original shape: {df.shape}"
)


# ============================================================
# 6. VALIDATE DATA
# ============================================================

print("\n" + "=" * 75)
print("DATA VALIDATION")
print("=" * 75)

required_columns = (
    METADATA_COLUMNS
    +
    EEG_CHANNELS
    +
    [
        LABEL_NAME_COLUMN,
        TARGET_COLUMN
    ]
)

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        "Missing required columns:\n"
        +
        "\n".join(
            missing_columns
        )
    )

print("✓ Required columns found.")

print(
    f"Samples       : {len(df)}"
)

print(
    f"EEG channels  : {len(EEG_CHANNELS)}"
)

print(
    f"Subjects      : {df['subject_id'].nunique()}"
)

print(
    f"Classes       : {df[TARGET_COLUMN].nunique()}"
)


# ============================================================
# 7. CREATE OUTPUT DATAFRAME
# ============================================================

features = pd.DataFrame(
    index=df.index
)


# ============================================================
# 8. PRESERVE METADATA
# ============================================================

print("\n" + "=" * 75)
print("1. PRESERVING METADATA")
print("=" * 75)

features["subject_id"] = (
    df["subject_id"]
)

features["trial_id"] = (
    df["trial_id"]
)

features["label_name"] = (
    df[LABEL_NAME_COLUMN]
)

features["label"] = (
    df[TARGET_COLUMN]
)

print(
    "✓ subject_id preserved"
)

print(
    "✓ trial_id preserved"
)

print(
    "✓ label_name preserved"
)

print(
    "✓ label preserved"
)


# ============================================================
# 9. RAW EEG FEATURES
# ============================================================

print("\n" + "=" * 75)
print("2. RAW EEG CHANNEL FEATURES")
print("=" * 75)

for channel in EEG_CHANNELS:

    features[channel] = df[channel]

print(
    f"✓ Added {len(EEG_CHANNELS)} raw EEG features."
)


# ============================================================
# 10. GLOBAL TRIAL STATISTICS
# ============================================================

print("\n" + "=" * 75)
print("3. GLOBAL TRIAL STATISTICS")
print("=" * 75)

features = add_statistics(
    features,
    df[EEG_CHANNELS],
    "global"
)

print(
    "Added:"
)

print(
    "  global_mean"
)

print(
    "  global_std"
)

print(
    "  global_median"
)

print(
    "  global_min"
)

print(
    "  global_max"
)

print(
    "  global_range"
)

print(
    "  global_rms"
)


# ============================================================
# 11. A-GROUP STATISTICS
# ============================================================

print("\n" + "=" * 75)
print("4. A-GROUP STATISTICS")
print("=" * 75)

features = add_statistics(
    features,
    df[A_CHANNELS],
    "A_group"
)

print(
    "✓ Added A-group statistical features."
)


# ============================================================
# 12. B-GROUP STATISTICS
# ============================================================

print("\n" + "=" * 75)
print("5. B-GROUP STATISTICS")
print("=" * 75)

features = add_statistics(
    features,
    df[B_CHANNELS],
    "B_group"
)

print(
    "✓ Added B-group statistical features."
)


# ============================================================
# 13. A vs B FEATURES
# ============================================================

print("\n" + "=" * 75)
print("6. A-VS-B FEATURES")
print("=" * 75)

A_mean = df[A_CHANNELS].mean(axis=1)

B_mean = df[B_CHANNELS].mean(axis=1)

A_std = df[A_CHANNELS].std(axis=1)

B_std = df[B_CHANNELS].std(axis=1)


features["A_B_mean_difference"] = (
    A_mean - B_mean
)

features["A_B_absolute_difference"] = (
    (A_mean - B_mean).abs()
)

features["A_B_mean_ratio"] = (
    A_mean
    /
    (B_mean.abs() + 1e-8)
)

features["A_B_std_difference"] = (
    A_std - B_std
)

features["A_B_std_ratio"] = (
    A_std
    /
    (B_std + 1e-8)
)

print(
    "✓ Added A-vs-B comparison features."
)


# ============================================================
# 14. PER-TRIAL Z-SCORE FEATURES
# ============================================================

print("\n" + "=" * 75)
print("7. PER-TRIAL CHANNEL NORMALIZATION")
print("=" * 75)

eeg_values = (
    df[EEG_CHANNELS]
)

trial_mean = (
    eeg_values.mean(axis=1)
)

trial_std = (
    eeg_values.std(axis=1)
)

trial_std_safe = (
    trial_std
    .replace(
        0,
        1e-8
    )
)


normalized_eeg = (
    eeg_values
    .sub(
        trial_mean,
        axis=0
    )
    .div(
        trial_std_safe,
        axis=0
    )
)


for channel in EEG_CHANNELS:

    features[
        f"{channel}_z"
    ] = normalized_eeg[channel]


print(
    f"✓ Added {len(EEG_CHANNELS)} "
    "per-trial normalized EEG features."
)


# ============================================================
# 15. CHECK FEATURE MATRIX
# ============================================================

print("\n" + "=" * 75)
print("8. FEATURE MATRIX CHECK")
print("=" * 75)

print(
    f"Feature dataset shape: "
    f"{features.shape}"
)

print(
    f"Original dataset shape: "
    f"{df.shape}"
)

print(
    f"Number of generated columns: "
    f"{len(features.columns)}"
)


# ============================================================
# 16. MISSING VALUE CHECK
# ============================================================

print("\n" + "=" * 75)
print("9. MISSING VALUE CHECK")
print("=" * 75)

missing_count = (
    features.isnull().sum().sum()
)

print(
    f"Total missing values: "
    f"{missing_count}"
)

if missing_count == 0:

    print(
        "✓ No missing values introduced."
    )

else:

    print(
        "⚠ Missing values detected."
    )

    missing_columns = (
        features.isnull()
        .sum()
    )

    print(
        missing_columns[
            missing_columns > 0
        ]
    )


# ============================================================
# 17. INFINITE VALUE CHECK
# ============================================================

print("\n" + "=" * 75)
print("10. INFINITE VALUE CHECK")
print("=" * 75)

numeric_columns = (
    features
    .select_dtypes(
        include=np.number
    )
    .columns
)

infinite_count = np.isinf(
    features[
        numeric_columns
    ].to_numpy()
).sum()

print(
    f"Total infinite values: "
    f"{infinite_count}"
)

if infinite_count == 0:

    print(
        "✓ No infinite values introduced."
    )

else:

    print(
        "⚠ Infinite values detected."
    )


# ============================================================
# 18. DUPLICATE CHECK
# ============================================================

print("\n" + "=" * 75)
print("11. DUPLICATE CHECK")
print("=" * 75)

duplicate_count = (
    features.duplicated().sum()
)

print(
    f"Duplicate feature rows: "
    f"{duplicate_count}"
)


# ============================================================
# 19. FEATURE STATISTICS
# ============================================================

print("\n" + "=" * 75)
print("12. FEATURE STATISTICS")
print("=" * 75)

# Exclude metadata and labels
feature_columns = [
    column
    for column in features.columns
    if column not in (
        METADATA_COLUMNS
        +
        [
            LABEL_NAME_COLUMN,
            TARGET_COLUMN
        ]
    )
]

feature_statistics = (
    features[
        feature_columns
    ]
    .describe()
    .T
)

print(
    feature_statistics.head(
        20
    ).to_string()
)

print(
    f"\nTotal ML feature columns: "
    f"{len(feature_columns)}"
)


# ============================================================
# 20. SAVE FEATURE STATISTICS
# ============================================================

statistics_path = (
    RESULTS_DIR
    /
    "engineered_feature_statistics.csv"
)

feature_statistics.to_csv(
    statistics_path
)

print(
    f"\n✓ Saved feature statistics:"
    f"\n{statistics_path}"
)


# ============================================================
# 21. SAVE ENGINEERED DATASET
# ============================================================

output_path = (
    FEATURE_DIR
    /
    "ds004514_eeg_engineered.csv"
)

features.to_csv(
    output_path,
    index=False
)

print(
    "\n" + "=" * 75
)

print(
    "13. SAVING ENGINEERED DATASET"
)

print(
    "=" * 75
)

print(
    f"✓ Saved engineered dataset:"
)

print(
    output_path
)

print(
    f"\nFinal shape: "
    f"{features.shape}"
)


# ============================================================
# 22. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("FEATURE ENGINEERING SUMMARY")
print("=" * 75)

print(
    f"Original samples       : "
    f"{len(df)}"
)

print(
    f"Original EEG channels  : "
    f"{len(EEG_CHANNELS)}"
)

print(
    f"Generated ML features  : "
    f"{len(feature_columns)}"
)

print(
    f"Final dataset columns  : "
    f"{len(features.columns)}"
)

print(
    f"Subjects               : "
    f"{df['subject_id'].nunique()}"
)

print(
    f"Classes                : "
    f"{df[TARGET_COLUMN].nunique()}"
)

print(
    f"Missing values         : "
    f"{missing_count}"
)

print(
    f"Infinite values        : "
    f"{infinite_count}"
)

print(
    f"Duplicate rows         : "
    f"{duplicate_count}"
)


# ============================================================
# 23. FEATURE GROUP SUMMARY
# ============================================================

print("\nFeature groups generated:")

print(
    "  1. 64 raw EEG channel features"
)

print(
    "  2. 7 global statistical features"
)

print(
    "  3. 7 A-group statistical features"
)

print(
    "  4. 7 B-group statistical features"
)

print(
    "  5. 5 A-vs-B comparison features"
)

print(
    "  6. 64 per-trial normalized EEG features"
)


# ============================================================
# 24. COMPLETION
# ============================================================

print("\n" + "=" * 75)
print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
print("=" * 75)

print(
    "\nOriginal approved CSV was not modified."
)

print(
    "\nEngineered dataset is ready for "
    "the next ML experiment."
)

print(
    "\nOutput:"
)

print(
    "data/features/ds004514_eeg_engineered.csv"
)