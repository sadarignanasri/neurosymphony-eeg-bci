import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/features/ds004514_fnirs.csv"
OUTPUT_DIR = "results/fnirs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# HELPER FUNCTION
# ============================================================

def print_section(title):
    print("\n")
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("NEURO-SYMPHONY")
print("fNIRS DATASET VALIDATION AND EDA")
print("=" * 70)

print("\nLoading dataset:")
print(DATA_PATH)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_PATH}\n\n"
        "Make sure the fNIRS CSV is placed inside:\n"
        "data/features/"
    )

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully.")
print("Shape:", df.shape)


# ============================================================
# DISPLAY COLUMNS
# ============================================================

print_section("DATASET COLUMNS")

print("Total columns:", len(df.columns))

for i, column in enumerate(df.columns, start=1):
    print(f"{i:3d}. {column}")


# ============================================================
# STANDARDIZE LABEL COLUMN
# ============================================================

print_section("LABEL COLUMN CHECK")

if "Label" in df.columns and "label" not in df.columns:

    print("Found column: Label")
    print("Renaming 'Label' → 'label'")

    df = df.rename(columns={"Label": "label"})

elif "label" in df.columns:

    print("Found standardized column: label")

else:

    raise ValueError(
        "ERROR: Neither 'Label' nor 'label' exists in the dataset."
    )


# ============================================================
# REQUIRED COLUMN CHECK
# ============================================================

required_columns = [
    "subject_id",
    "trial_id",
    "label_name",
    "label"
]

print_section("REQUIRED COLUMN VALIDATION")

missing_required = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_required:

    raise ValueError(
        f"Missing required columns: {missing_required}"
    )

print("All required columns are present.")

print("\nRequired columns:")
for column in required_columns:
    print("  ✓", column)


# ============================================================
# IDENTIFY fNIRS FEATURE COLUMNS
# ============================================================

metadata_columns = [
    "subject_id",
    "trial_id",
    "label_name",
    "label"
]

feature_columns = [
    column
    for column in df.columns
    if column not in metadata_columns
]

print_section("fNIRS FEATURE IDENTIFICATION")

print("Number of fNIRS features:", len(feature_columns))

print("\nfNIRS feature columns:")

for i, column in enumerate(feature_columns, start=1):
    print(f"{i:3d}. {column}")


# ============================================================
# FEATURE DATA TYPE CHECK
# ============================================================

print_section("FEATURE DATA TYPE CHECK")

non_numeric_features = [
    column
    for column in feature_columns
    if not pd.api.types.is_numeric_dtype(df[column])
]

if non_numeric_features:

    print("WARNING: Non-numeric fNIRS features found:")

    for column in non_numeric_features:
        print("  ✗", column)

else:

    print("All fNIRS feature columns are numeric.")


# ============================================================
# SUBJECT VALIDATION
# ============================================================

print_section("SUBJECT VALIDATION")

subjects = sorted(
    df["subject_id"].dropna().unique()
)

print("Number of subjects:", len(subjects))

print("\nSubjects:")

for subject in subjects:
    print(" ", subject)

samples_per_subject = (
    df["subject_id"]
    .value_counts()
    .sort_index()
)

print("\nSamples per subject:")
print(samples_per_subject.to_string())


# ============================================================
# CHECK EXPECTED SUBJECT SAMPLE COUNTS
# ============================================================

print("\nExpected samples per subject: 720")

unexpected_subject_counts = (
    samples_per_subject[
        samples_per_subject != 720
    ]
)

if len(unexpected_subject_counts) > 0:

    print(
        "\nWARNING: Some subjects do not contain exactly 720 samples:"
    )

    print(unexpected_subject_counts)

else:

    print("✓ Every subject contains exactly 720 samples.")


# ============================================================
# LABEL VALIDATION
# ============================================================

print_section("LABEL VALIDATION")

num_classes = df["label"].nunique()

print("Number of classes:", num_classes)

print("\nNumeric label values:")

print(
    sorted(
        df["label"]
        .dropna()
        .unique()
    )
)

print("\nClass distribution:")

class_distribution = (
    df.groupby(
        ["label", "label_name"]
    )
    .size()
    .reset_index(name="sample_count")
    .sort_values("label")
)

print(
    class_distribution.to_string(
        index=False
    )
)


# ============================================================
# LABEL NAME VALIDATION
# ============================================================

print_section("LABEL NAME VALIDATION")

label_name_count = df["label_name"].nunique()

print(
    "Number of unique label names:",
    label_name_count
)

print("\nLabel names:")

for label_name in sorted(
    df["label_name"].dropna().unique()
):

    print(" ", label_name)


# ============================================================
# LABEL ↔ LABEL NAME MAPPING
# ============================================================

print_section("LABEL ↔ LABEL NAME MAPPING")

label_mapping = (
    df[
        ["label", "label_name"]
    ]
    .drop_duplicates()
    .sort_values("label")
)

print(
    label_mapping.to_string(
        index=False
    )
)

if label_mapping["label"].nunique() != len(label_mapping):

    raise ValueError(
        "ERROR: One numeric label maps to multiple label names."
    )

if label_mapping["label_name"].nunique() != len(label_mapping):

    raise ValueError(
        "ERROR: One label name maps to multiple numeric labels."
    )

print("\n✓ Every label has exactly one label name.")


# ============================================================
# SUBJECT × CLASS DISTRIBUTION
# ============================================================

print_section("SUBJECT × CLASS DISTRIBUTION")

subject_class = pd.crosstab(
    df["subject_id"],
    df["label_name"]
)

print(
    subject_class.to_string()
)

print(
    "\nExpected samples per subject/class: 90"
)

unexpected_subject_class = (
    subject_class[
        subject_class != 90
    ]
    .stack()
    .dropna()
)

if len(unexpected_subject_class) > 0:

    print(
        "\nWARNING: Some subject/class combinations "
        "do not contain exactly 90 samples."
    )

else:

    print(
        "✓ Every subject contains exactly "
        "90 samples for every class."
    )


# ============================================================
# DUPLICATE SUBJECT/TRIAL CHECK
# ============================================================

print_section("DUPLICATE TRIAL CHECK")

duplicate_trials = df.duplicated(
    subset=[
        "subject_id",
        "trial_id"
    ]
).sum()

print(
    "Duplicate subject/trial pairs:",
    duplicate_trials
)

if duplicate_trials == 0:

    print(
        "✓ Every subject/trial combination is unique."
    )

else:

    print(
        "WARNING: Duplicate subject/trial combinations found."
    )


# ============================================================
# DUPLICATE COMPLETE ROW CHECK
# ============================================================

duplicate_rows = df.duplicated().sum()

print(
    "\nDuplicate complete rows:",
    duplicate_rows
)

if duplicate_rows == 0:

    print("✓ No duplicate complete rows.")

else:

    print(
        "WARNING: Duplicate complete rows found."
    )


# ============================================================
# MISSING VALUE ANALYSIS
# ============================================================

print_section("MISSING VALUE ANALYSIS")

missing_by_feature = (
    df[feature_columns]
    .isna()
    .sum()
    .sort_values(
        ascending=False
    )
)

missing_percentage = (
    missing_by_feature
    / len(df)
    * 100
)

missing_table = pd.DataFrame({

    "missing_count":
        missing_by_feature,

    "missing_percentage":
        missing_percentage

})

print(
    "\nMissing values by fNIRS feature:"
)

print(
    missing_table.to_string()
)


total_missing = int(
    df[feature_columns]
    .isna()
    .sum()
    .sum()
)

print(
    "\nTotal missing fNIRS values:",
    total_missing
)

print(
    "Total possible fNIRS values:",
    len(df) * len(feature_columns)
)

missing_percentage_total = (
    total_missing
    / (len(df) * len(feature_columns))
    * 100
)

print(
    "Overall missing percentage:",
    f"{missing_percentage_total:.2f}%"
)


# ============================================================
# COMPLETELY MISSING FEATURES
# ============================================================

completely_missing = [
    column
    for column in feature_columns
    if df[column].isna().all()
]

print_section("COMPLETELY MISSING FEATURES")

if completely_missing:

    print(
        "Features that are completely missing:"
    )

    for column in completely_missing:
        print("  ✗", column)

else:

    print(
        "✓ No fNIRS feature is completely missing."
    )


# ============================================================
# CHECK CONSTANT FEATURES
# ============================================================

print_section("CONSTANT FEATURE CHECK")

constant_features = []

for column in feature_columns:

    non_missing = df[column].dropna()

    if len(non_missing) == 0:
        continue

    if non_missing.nunique() <= 1:
        constant_features.append(column)


if constant_features:

    print(
        "Constant fNIRS features:"
    )

    for column in constant_features:
        print("  ✗", column)

else:

    print(
        "✓ No constant fNIRS features found."
    )


# ============================================================
# INFINITE VALUE CHECK
# ============================================================

print_section("INFINITE VALUE CHECK")

numeric_features = df[
    feature_columns
].select_dtypes(
    include=np.number
)

infinite_count = int(
    np.isinf(
        numeric_features.to_numpy()
    ).sum()
)

print(
    "Infinite values:",
    infinite_count
)

if infinite_count == 0:

    print(
        "✓ No infinite values found."
    )

else:

    print(
        "WARNING: Infinite values detected."
    )


# ============================================================
# FEATURE STATISTICS
# ============================================================

print_section("fNIRS FEATURE STATISTICS")

stats = (
    df[feature_columns]
    .describe()
    .T
)

stats["missing_count"] = (
    df[feature_columns]
    .isna()
    .sum()
)

stats["missing_percentage"] = (
    stats["missing_count"]
    / len(df)
    * 100
)

statistics_path = os.path.join(
    OUTPUT_DIR,
    "fnirs_feature_statistics.csv"
)

stats.to_csv(
    statistics_path
)

print(
    "Feature statistics saved to:"
)

print(statistics_path)


# ============================================================
# SAVE MISSING VALUE REPORT
# ============================================================

missing_report_path = os.path.join(
    OUTPUT_DIR,
    "fnirs_missing_value_report.csv"
)

missing_table.to_csv(
    missing_report_path
)

print(
    "\nMissing-value report saved to:"
)

print(missing_report_path)


# ============================================================
# SAVE LABEL DISTRIBUTION
# ============================================================

class_distribution_path = os.path.join(
    OUTPUT_DIR,
    "fnirs_class_distribution.csv"
)

class_distribution.to_csv(
    class_distribution_path,
    index=False
)

print(
    "\nClass distribution saved to:"
)

print(class_distribution_path)


# ============================================================
# SAVE SUBJECT × CLASS TABLE
# ============================================================

subject_class_path = os.path.join(
    OUTPUT_DIR,
    "fnirs_subject_class_distribution.csv"
)

subject_class.to_csv(
    subject_class_path
)

print(
    "\nSubject/class distribution saved to:"
)

print(subject_class_path)


# ============================================================
# CLASS DISTRIBUTION PLOT
# ============================================================

plt.figure(
    figsize=(12, 6)
)

(
    df["label_name"]
    .value_counts()
    .sort_index()
    .plot(
        kind="bar"
    )
)

plt.title(
    "fNIRS Class Distribution"
)

plt.xlabel(
    "Task"
)

plt.ylabel(
    "Number of Samples"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

class_plot_path = os.path.join(
    OUTPUT_DIR,
    "01_class_distribution.png"
)

plt.savefig(
    class_plot_path,
    dpi=300
)

plt.close()


# ============================================================
# SUBJECT DISTRIBUTION PLOT
# ============================================================

plt.figure(
    figsize=(10, 6)
)

(
    df["subject_id"]
    .value_counts()
    .sort_index()
    .plot(
        kind="bar"
    )
)

plt.title(
    "fNIRS Subject Distribution"
)

plt.xlabel(
    "Subject"
)

plt.ylabel(
    "Number of Samples"
)

plt.tight_layout()

subject_plot_path = os.path.join(
    OUTPUT_DIR,
    "02_subject_distribution.png"
)

plt.savefig(
    subject_plot_path,
    dpi=300
)

plt.close()


# ============================================================
# MISSING VALUE PLOT
# ============================================================

plt.figure(
    figsize=(14, 7)
)

(
    missing_percentage
    .sort_values(
        ascending=False
    )
    .plot(
        kind="bar"
    )
)

plt.title(
    "fNIRS Feature Missing Percentage"
)

plt.xlabel(
    "fNIRS Feature"
)

plt.ylabel(
    "Missing Percentage (%)"
)

plt.xticks(
    rotation=90
)

plt.tight_layout()

missing_plot_path = os.path.join(
    OUTPUT_DIR,
    "03_missing_values.png"
)

plt.savefig(
    missing_plot_path,
    dpi=300
)

plt.close()


# ============================================================
# SUBJECT × CLASS HEATMAP
# ============================================================

plt.figure(
    figsize=(14, 7)
)

plt.imshow(
    subject_class.values,
    aspect="auto"
)

plt.colorbar(
    label="Number of Samples"
)

plt.xticks(
    range(len(subject_class.columns)),
    subject_class.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(subject_class.index)),
    subject_class.index
)

plt.xlabel(
    "Task"
)

plt.ylabel(
    "Subject"
)

plt.title(
    "fNIRS Subject × Class Distribution"
)

plt.tight_layout()

heatmap_path = os.path.join(
    OUTPUT_DIR,
    "04_subject_class_heatmap.png"
)

plt.savefig(
    heatmap_path,
    dpi=300
)

plt.close()


# ============================================================
# SAVE STANDARDIZED DATASET
# ============================================================

df.to_csv(
    DATA_PATH,
    index=False
)

print_section("STANDARDIZED DATASET SAVED")

print(
    "Standardized dataset saved to:"
)

print(DATA_PATH)

print(
    "\nLabel column is now:",
    "label"
)


# ============================================================
# FINAL VALIDATION SUMMARY
# ============================================================

print_section("FINAL VALIDATION SUMMARY")

print(
    "Rows:",
    len(df)
)

print(
    "Columns:",
    len(df.columns)
)

print(
    "Subjects:",
    df["subject_id"].nunique()
)

print(
    "Classes:",
    df["label"].nunique()
)

print(
    "fNIRS features:",
    len(feature_columns)
)

print(
    "Duplicate subject/trial pairs:",
    duplicate_trials
)

print(
    "Duplicate complete rows:",
    duplicate_rows
)

print(
    "Missing fNIRS values:",
    total_missing
)

print(
    "Infinite values:",
    infinite_count
)

print(
    "Constant features:",
    len(constant_features)
)

print(
    "Completely missing features:",
    len(completely_missing)
)


# ============================================================
# OUTPUT FILES
# ============================================================

print_section("OUTPUT FILES")

print(
    "1.",
    statistics_path
)

print(
    "2.",
    missing_report_path
)

print(
    "3.",
    class_distribution_path
)

print(
    "4.",
    subject_class_path
)

print(
    "5.",
    class_plot_path
)

print(
    "6.",
    subject_plot_path
)

print(
    "7.",
    missing_plot_path
)

print(
    "8.",
    heatmap_path
)


# ============================================================
# COMPLETED
# ============================================================

print("\n")
print("=" * 70)
print("fNIRS VALIDATION AND EDA COMPLETED SUCCESSFULLY")
print("=" * 70)