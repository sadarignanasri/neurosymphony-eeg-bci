"""
Neuro-Symphony
============================================================
DS004514 EEG Dataset Analysis
============================================================

Purpose:
    Perform the first systematic analysis of the guide-approved
    DS004514 EEG CSV before machine-learning training.

Dataset:
    data/processed/ds004514_eeg_approved.csv

Expected structure:
    subject_id
    trial_id
    A1 ... A32
    B1 ... B32
    label_name
    label

Important:
    - The original approved CSV is NOT modified.
    - label_name = human-readable class name
    - label      = numerical ML target
    - A1-B32     = 64 EEG input features
"""

# ============================================================
# IMPORTS
# ============================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

# Project root:
# ml/eeg/01_dataset_analysis.py
#       ↑
# parents[0] = eeg
# parents[1] = ml
# parents[2] = neuro-symphony-bci

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ds004514_eeg_approved.csv"
)


# ============================================================
# 2. EXPECTED COLUMNS
# ============================================================

METADATA_COLUMNS = [
    "subject_id",
    "trial_id",
]

LABEL_NAME_COLUMN = "label_name"

TARGET_COLUMN = "label"

EXPECTED_EEG_CHANNELS = [
    f"A{i}" for i in range(1, 33)
] + [
    f"B{i}" for i in range(1, 33)
]


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def print_section(title):
    """Print a formatted section heading."""
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def check_required_columns(df):
    """Check that all required columns exist."""

    required_columns = (
        METADATA_COLUMNS
        + EXPECTED_EEG_CHANNELS
        + [
            LABEL_NAME_COLUMN,
            TARGET_COLUMN,
        ]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        print("\nERROR: Required columns are missing:")
        for column in missing_columns:
            print(f"  - {column}")

        print("\nDataset columns found:")
        for column in df.columns:
            print(f"  - {column}")

        sys.exit(1)

    print("✓ All required columns are present.")


# ============================================================
# 4. START ANALYSIS
# ============================================================

print("=" * 75)
print("NEURO-SYMPHONY")
print("DS004514 EEG DATASET ANALYSIS")
print("=" * 75)

print("\nProject root:")
print(PROJECT_ROOT)

print("\nDataset path:")
print(CSV_PATH)


# ============================================================
# 5. CHECK FILE
# ============================================================

print_section("1. DATASET FILE CHECK")

if not CSV_PATH.exists():
    print("ERROR: Dataset file was not found.")
    print(f"\nExpected location:\n{CSV_PATH}")
    sys.exit(1)

print("✓ Dataset file found.")
print(f"File size: {CSV_PATH.stat().st_size / (1024 * 1024):.2f} MB")


# ============================================================
# 6. LOAD DATASET
# ============================================================

print_section("2. LOADING DATASET")

try:
    df = pd.read_csv(CSV_PATH)
except Exception as error:
    print("ERROR: Could not load the CSV file.")
    print(f"Reason: {error}")
    sys.exit(1)

print("✓ Dataset loaded successfully.")


# ============================================================
# 7. BASIC DATASET INFORMATION
# ============================================================

print_section("3. BASIC DATASET INFORMATION")

num_rows = df.shape[0]
num_columns = df.shape[1]

print(f"Number of data rows : {num_rows}")
print(f"Number of columns   : {num_columns}")
print(f"Dataset shape       : {df.shape}")


# ============================================================
# 8. VERIFY REQUIRED COLUMNS
# ============================================================

print_section("4. REQUIRED COLUMN VERIFICATION")

check_required_columns(df)


# ============================================================
# 9. COLUMN LIST
# ============================================================

print_section("5. COMPLETE COLUMN LIST")

for index, column in enumerate(df.columns, start=1):
    print(f"{index:02d}. {column}")


# ============================================================
# 10. IDENTIFY EEG CHANNELS
# ============================================================

print_section("6. EEG CHANNEL IDENTIFICATION")

eeg_channels = [
    column
    for column in df.columns
    if column in EXPECTED_EEG_CHANNELS
]

print(f"Number of EEG channels: {len(eeg_channels)}")

if len(eeg_channels) != 64:
    print(
        "\nERROR: Expected 64 EEG channels "
        f"but found {len(eeg_channels)}."
    )
    sys.exit(1)

print("✓ Confirmed: 64 EEG channels.")

print("\nEEG channels:")

for index, channel in enumerate(eeg_channels, start=1):
    print(f"{index:02d}. {channel}")


# ============================================================
# 11. COLUMN CATEGORIES
# ============================================================

print_section("7. COLUMN CATEGORIES")

print("Metadata columns:")
for column in METADATA_COLUMNS:
    print(f"  - {column}")

print("\nEEG feature columns:")
print("  - A1 through A32")
print("  - B1 through B32")
print(f"  - Total EEG features: {len(eeg_channels)}")

print("\nLabel columns:")
print(f"  - {LABEL_NAME_COLUMN} → human-readable class")
print(f"  - {TARGET_COLUMN} → numerical ML target")


# ============================================================
# 12. DATA TYPES
# ============================================================

print_section("8. DATA TYPES")

print(df.dtypes.to_string())


# ============================================================
# 13. EEG DATA TYPE CHECK
# ============================================================

print_section("9. EEG DATA TYPE VALIDATION")

non_numeric_eeg_columns = [
    column
    for column in eeg_channels
    if not pd.api.types.is_numeric_dtype(df[column])
]

if non_numeric_eeg_columns:
    print("WARNING: The following EEG columns are not numeric:")

    for column in non_numeric_eeg_columns:
        print(f"  - {column}")

else:
    print("✓ All 64 EEG channels are numeric.")


# ============================================================
# 14. LABEL DATA TYPE CHECK
# ============================================================

print_section("10. LABEL DATA TYPE VALIDATION")

print(
    f"Target column '{TARGET_COLUMN}' "
    f"dtype: {df[TARGET_COLUMN].dtype}"
)

if pd.api.types.is_numeric_dtype(df[TARGET_COLUMN]):
    print("✓ Numerical label column is numeric.")
else:
    print("WARNING: Numerical label column is not numeric.")


# ============================================================
# 15. MISSING VALUE ANALYSIS
# ============================================================

print_section("11. MISSING VALUE ANALYSIS")

missing_values = df.isnull().sum()

total_missing = int(missing_values.sum())

print(f"Total missing values: {total_missing}")

if total_missing == 0:

    print("✓ No missing values found.")

else:

    print("\nColumns containing missing values:")

    for column, count in missing_values[missing_values > 0].items():
        print(f"  - {column}: {count}")


# ============================================================
# 16. INFINITE VALUE CHECK
# ============================================================

print_section("12. INFINITE VALUE ANALYSIS")

numeric_columns = df.select_dtypes(
    include=np.number
).columns

infinite_values = np.isinf(
    df[numeric_columns].to_numpy()
).sum()

print(f"Total infinite numeric values: {infinite_values}")

if infinite_values == 0:
    print("✓ No infinite values found.")
else:
    print("WARNING: Infinite values were found.")


# ============================================================
# 17. DUPLICATE ANALYSIS
# ============================================================

print_section("13. DUPLICATE ROW ANALYSIS")

duplicate_count = int(df.duplicated().sum())

print(f"Duplicate rows: {duplicate_count}")

if duplicate_count == 0:
    print("✓ No duplicate rows found.")
else:
    print("WARNING: Duplicate rows found.")


# ============================================================
# 18. SUBJECT ANALYSIS
# ============================================================

print_section("14. SUBJECT DISTRIBUTION")

number_of_subjects = df["subject_id"].nunique()

print(f"Number of unique subjects: {number_of_subjects}")

subject_counts = (
    df["subject_id"]
    .value_counts()
    .sort_index()
)

print("\nSamples per subject:")

print(subject_counts.to_string())


# ============================================================
# 19. SUBJECT BALANCE CHECK
# ============================================================

print_section("15. SUBJECT BALANCE CHECK")

minimum_subject_samples = subject_counts.min()
maximum_subject_samples = subject_counts.max()

print(
    f"Minimum samples per subject: "
    f"{minimum_subject_samples}"
)

print(
    f"Maximum samples per subject: "
    f"{maximum_subject_samples}"
)

if minimum_subject_samples == maximum_subject_samples:
    print("✓ All subjects contain the same number of samples.")
else:
    print("⚠ Subjects have different numbers of samples.")


# ============================================================
# 20. NUMERICAL LABEL DISTRIBUTION
# ============================================================

print_section("16. NUMERICAL LABEL DISTRIBUTION")

label_counts = (
    df[TARGET_COLUMN]
    .value_counts()
    .sort_index()
)

print(label_counts.to_string())

print(
    f"\nNumber of unique numerical labels: "
    f"{df[TARGET_COLUMN].nunique()}"
)


# ============================================================
# 21. LABEL NAME DISTRIBUTION
# ============================================================

print_section("17. LABEL NAME DISTRIBUTION")

label_name_counts = (
    df[LABEL_NAME_COLUMN]
    .value_counts()
)

print(label_name_counts.to_string())

print(
    f"\nNumber of unique label names: "
    f"{df[LABEL_NAME_COLUMN].nunique()}"
)


# ============================================================
# 22. LABEL ↔ LABEL_NAME MAPPING
# ============================================================

print_section("18. LABEL ↔ LABEL_NAME MAPPING")

label_mapping = (
    df[
        [
            TARGET_COLUMN,
            LABEL_NAME_COLUMN
        ]
    ]
    .drop_duplicates()
    .sort_values(TARGET_COLUMN)
)

print(
    label_mapping.to_string(index=False)
)


# ============================================================
# 23. LABEL CONSISTENCY CHECK
# ============================================================

print_section("19. LABEL CONSISTENCY CHECK")

mapping_counts = (
    df.groupby(TARGET_COLUMN)[LABEL_NAME_COLUMN]
    .nunique()
)

if mapping_counts.max() == 1:

    print(
        "✓ Every numerical label maps "
        "to exactly one label_name."
    )

else:

    print(
        "WARNING: At least one numerical label "
        "maps to multiple label names."
    )

    inconsistent_labels = mapping_counts[
        mapping_counts > 1
    ]

    print("\nInconsistent labels:")
    print(inconsistent_labels)


# ============================================================
# 24. EXPECTED CLASS CHECK
# ============================================================

print_section("20. EXPECTED CLASS CHECK")

expected_labels = list(range(1, 9))

actual_labels = sorted(
    df[TARGET_COLUMN]
    .dropna()
    .unique()
    .tolist()
)

print(f"Expected numerical labels: {expected_labels}")
print(f"Actual numerical labels  : {actual_labels}")

if actual_labels == expected_labels:
    print("✓ All 8 expected numerical labels are present.")
else:
    print("⚠ Numerical labels do not exactly match 1–8.")


# ============================================================
# 25. CLASS BALANCE
# ============================================================

print_section("21. CLASS BALANCE ANALYSIS")

class_counts = (
    df.groupby(
        [
            TARGET_COLUMN,
            LABEL_NAME_COLUMN
        ]
    )
    .size()
    .reset_index(name="sample_count")
    .sort_values(TARGET_COLUMN)
)

print(
    class_counts.to_string(index=False)
)

minimum_class_samples = class_counts["sample_count"].min()
maximum_class_samples = class_counts["sample_count"].max()

print(
    f"\nMinimum samples per class: "
    f"{minimum_class_samples}"
)

print(
    f"Maximum samples per class: "
    f"{maximum_class_samples}"
)

if minimum_class_samples == maximum_class_samples:
    print("✓ All classes are perfectly balanced.")
else:
    print("⚠ Classes are not perfectly balanced.")


# ============================================================
# 26. TRIAL ANALYSIS
# ============================================================

print_section("22. TRIAL ANALYSIS")

number_of_trials = df["trial_id"].nunique()

print(f"Unique trial IDs: {number_of_trials}")

trial_counts = (
    df["trial_id"]
    .value_counts()
    .sort_index()
)

print("\nSamples per trial ID:")

print(trial_counts.to_string())


# ============================================================
# 27. EEG DESCRIPTIVE STATISTICS
# ============================================================

print_section("23. EEG DESCRIPTIVE STATISTICS")

eeg_statistics = (
    df[eeg_channels]
    .describe()
    .T
)

print(
    eeg_statistics.to_string()
)


# ============================================================
# 28. EEG RANGE CHECK
# ============================================================

print_section("24. EEG VALUE RANGE")

eeg_minimum = df[eeg_channels].min().min()
eeg_maximum = df[eeg_channels].max().max()

print(f"Minimum EEG value: {eeg_minimum:.6f}")
print(f"Maximum EEG value: {eeg_maximum:.6f}")


# ============================================================
# 29. CHANNEL VARIANCE CHECK
# ============================================================

print_section("25. EEG CHANNEL VARIANCE CHECK")

channel_variances = (
    df[eeg_channels]
    .var()
    .sort_values()
)

print("Lowest-variance channels:")

print(
    channel_variances.head(10).to_string()
)

print("\nHighest-variance channels:")

print(
    channel_variances.tail(10).sort_values(
        ascending=False
    ).to_string()
)


# ============================================================
# 30. CONSTANT CHANNEL CHECK
# ============================================================

print_section("26. CONSTANT EEG CHANNEL CHECK")

constant_channels = [
    channel
    for channel in eeg_channels
    if df[channel].nunique() <= 1
]

if constant_channels:

    print("WARNING: Constant EEG channels found:")

    for channel in constant_channels:
        print(f"  - {channel}")

else:

    print("✓ No constant EEG channels found.")


# ============================================================
# 31. UNIQUE VALUES PER EEG CHANNEL
# ============================================================

print_section("27. EEG CHANNEL UNIQUENESS")

channel_unique_counts = (
    df[eeg_channels]
    .nunique()
    .sort_values()
)

print(
    channel_unique_counts.to_string()
)


# ============================================================
# 32. ML FEATURE MATRIX CHECK
# ============================================================

print_section("28. ML FEATURE MATRIX CHECK")

X = df[eeg_channels]

y = df[TARGET_COLUMN]

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

print(
    f"\nExpected X shape: "
    f"({len(df)}, 64)"
)

if X.shape == (len(df), 64):
    print("✓ EEG feature matrix is correctly shaped.")
else:
    print("⚠ Unexpected EEG feature matrix shape.")


# ============================================================
# 33. TARGET VECTOR CHECK
# ============================================================

print_section("29. ML TARGET VECTOR CHECK")

print(f"Target shape: {y.shape}")

print(
    f"Target classes: "
    f"{sorted(y.unique().tolist())}"
)

if y.nunique() == 8:
    print("✓ Target contains 8 classes.")
else:
    print("⚠ Target does not contain exactly 8 classes.")


# ============================================================
# 34. FINAL DATASET STRUCTURE
# ============================================================

print_section("30. FINAL DATASET STRUCTURE")

print("Metadata:")
print("  1. subject_id")
print("  2. trial_id")

print("\nEEG input features:")
print("  A1 - A32")
print("  B1 - B32")
print("  Total = 64 EEG channels")

print("\nTarget information:")
print("  label_name → human-readable class")
print("  label      → numerical ML target")

print("\nTotal columns:")
print(f"  {len(df.columns)}")

print("\nTotal samples:")
print(f"  {len(df)}")


# ============================================================
# 35. ML DATASET SUMMARY
# ============================================================

print_section("31. ML DATASET SUMMARY")

print(f"Samples                 : {len(df)}")
print(f"Subjects                : {number_of_subjects}")
print(f"EEG channels            : {len(eeg_channels)}")
print(f"Classes                 : {df[TARGET_COLUMN].nunique()}")
print(f"Total CSV columns       : {len(df.columns)}")
print(f"ML input features       : {len(eeg_channels)}")
print(f"Missing values          : {total_missing}")
print(f"Duplicate rows          : {duplicate_count}")
print(f"Infinite numeric values : {infinite_values}")


# ============================================================
# 36. FINAL STATUS
# ============================================================

print_section("32. ANALYSIS STATUS")

analysis_successful = True

if len(df) != 8640:
    print("⚠ Unexpected number of samples.")
    analysis_successful = False

if len(df.columns) != 68:
    print("⚠ Unexpected number of columns.")
    analysis_successful = False

if len(eeg_channels) != 64:
    print("⚠ Expected 64 EEG channels.")
    analysis_successful = False

if df[TARGET_COLUMN].nunique() != 8:
    print("⚠ Expected 8 classes.")
    analysis_successful = False

if total_missing != 0:
    print("⚠ Missing values detected.")
    analysis_successful = False

if infinite_values != 0:
    print("⚠ Infinite values detected.")
    analysis_successful = False

if duplicate_count != 0:
    print("⚠ Duplicate rows detected.")
    analysis_successful = False

if analysis_successful:

    print("✓ DATASET PASSED INITIAL ANALYSIS CHECKS")
    print("\nThe dataset is ready for the EDA stage.")

else:

    print("⚠ DATASET REQUIRES FURTHER INVESTIGATION")
    print("\nDo NOT start model training yet.")


# ============================================================
# END
# ============================================================

print("\n" + "=" * 75)
print("END OF DS004514 EEG DATASET ANALYSIS")
print("=" * 75)