"""
Neuro-Symphony
============================================================
DS004514 EEG Exploratory Data Analysis (EDA)
============================================================

Purpose:
    Perform visual and statistical exploratory analysis of the
    guide-approved DS004514 EEG CSV.

Input:
    data/processed/ds004514_eeg_approved.csv

Output:
    results/eeg/

Analysed:
    1. Class distribution
    2. Subject distribution
    3. Samples per subject/class
    4. EEG channel variance
    5. EEG channel mean
    6. EEG feature distributions
    7. EEG correlation matrix
    8. Class-wise EEG statistics
    9. Subject-wise EEG statistics

IMPORTANT:
    The original CSV is never modified.
"""

# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "eeg"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. DATASET CONFIGURATION
# ============================================================

METADATA_COLUMNS = [
    "subject_id",
    "trial_id"
]

LABEL_NAME_COLUMN = "label_name"

TARGET_COLUMN = "label"

EEG_CHANNELS = (
    [f"A{i}" for i in range(1, 33)]
    +
    [f"B{i}" for i in range(1, 33)]
)


# ============================================================
# 3. HELPER FUNCTION
# ============================================================

def save_figure(filename):
    """
    Save the current matplotlib figure into results/eeg/.
    """

    output_path = RESULTS_DIR / filename

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"✓ Saved: {output_path}")

    plt.close()


# ============================================================
# 4. START
# ============================================================

print("=" * 75)
print("NEURO-SYMPHONY")
print("DS004514 EEG EXPLORATORY DATA ANALYSIS")
print("=" * 75)


# ============================================================
# 5. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

if not CSV_PATH.exists():

    raise FileNotFoundError(
        f"Dataset not found:\n{CSV_PATH}"
    )

df = pd.read_csv(CSV_PATH)

print("✓ Dataset loaded successfully.")

print(f"Dataset shape: {df.shape}")


# ============================================================
# 6. BASIC VALIDATION
# ============================================================

print("\n" + "=" * 75)
print("DATASET VALIDATION")
print("=" * 75)

print(f"Samples       : {len(df)}")
print(f"Columns       : {len(df.columns)}")
print(f"Subjects      : {df['subject_id'].nunique()}")
print(f"Classes       : {df[TARGET_COLUMN].nunique()}")
print(f"EEG channels  : {len(EEG_CHANNELS)}")

missing = df.isnull().sum().sum()

print(f"Missing values: {missing}")


# ============================================================
# 7. CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 75)
print("1. CLASS DISTRIBUTION")
print("=" * 75)

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


# ------------------------------------------------------------
# Plot class distribution
# ------------------------------------------------------------

plt.figure(figsize=(14, 7))

sns.barplot(
    data=class_counts,
    x=LABEL_NAME_COLUMN,
    y="sample_count"
)

plt.title(
    "DS004514 EEG Class Distribution",
    fontsize=16
)

plt.xlabel(
    "Class",
    fontsize=12
)

plt.ylabel(
    "Number of Samples",
    fontsize=12
)

plt.xticks(
    rotation=45,
    ha="right"
)

save_figure(
    "01_class_distribution.png"
)


# ============================================================
# 8. SUBJECT DISTRIBUTION
# ============================================================

print("\n" + "=" * 75)
print("2. SUBJECT DISTRIBUTION")
print("=" * 75)

subject_counts = (
    df["subject_id"]
    .value_counts()
    .sort_index()
)

print(
    subject_counts.to_string()
)


# ------------------------------------------------------------
# Plot subject distribution
# ------------------------------------------------------------

plt.figure(figsize=(12, 6))

subject_counts.plot(
    kind="bar"
)

plt.title(
    "DS004514 EEG Samples per Subject",
    fontsize=16
)

plt.xlabel(
    "Subject",
    fontsize=12
)

plt.ylabel(
    "Number of Samples",
    fontsize=12
)

plt.xticks(
    rotation=45
)

save_figure(
    "02_subject_distribution.png"
)


# ============================================================
# 9. SUBJECT × CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 75)
print("3. SUBJECT × CLASS DISTRIBUTION")
print("=" * 75)

subject_class_table = pd.crosstab(
    df["subject_id"],
    df[LABEL_NAME_COLUMN]
)

print(
    subject_class_table.to_string()
)


# ------------------------------------------------------------
# Heatmap
# ------------------------------------------------------------

plt.figure(
    figsize=(15, 8)
)

sns.heatmap(
    subject_class_table,
    annot=True,
    fmt="d",
    cmap="viridis"
)

plt.title(
    "Samples per Subject and Class",
    fontsize=16
)

plt.xlabel(
    "Class"
)

plt.ylabel(
    "Subject"
)

plt.xticks(
    rotation=45,
    ha="right"
)

save_figure(
    "03_subject_class_heatmap.png"
)


# ============================================================
# 10. EEG CHANNEL MEAN
# ============================================================

print("\n" + "=" * 75)
print("4. EEG CHANNEL MEAN")
print("=" * 75)

channel_means = (
    df[EEG_CHANNELS]
    .mean()
    .sort_values()
)

print(
    channel_means.to_string()
)


# ------------------------------------------------------------
# Plot channel means
# ------------------------------------------------------------

plt.figure(
    figsize=(16, 7)
)

channel_means.plot(
    kind="bar"
)

plt.title(
    "Mean Value of Each EEG Channel",
    fontsize=16
)

plt.xlabel(
    "EEG Channel"
)

plt.ylabel(
    "Mean EEG Value"
)

plt.xticks(
    rotation=90
)

save_figure(
    "04_channel_means.png"
)


# ============================================================
# 11. EEG CHANNEL VARIANCE
# ============================================================

print("\n" + "=" * 75)
print("5. EEG CHANNEL VARIANCE")
print("=" * 75)

channel_variances = (
    df[EEG_CHANNELS]
    .var()
    .sort_values()
)

print(
    channel_variances.to_string()
)


# ------------------------------------------------------------
# Plot channel variance
# ------------------------------------------------------------

plt.figure(
    figsize=(16, 7)
)

channel_variances.plot(
    kind="bar"
)

plt.title(
    "Variance of Each EEG Channel",
    fontsize=16
)

plt.xlabel(
    "EEG Channel"
)

plt.ylabel(
    "Variance"
)

plt.xticks(
    rotation=90
)

save_figure(
    "05_channel_variance.png"
)


# ============================================================
# 12. TOP VARIABLE CHANNELS
# ============================================================

print("\n" + "=" * 75)
print("6. TOP 10 HIGHEST-VARIANCE EEG CHANNELS")
print("=" * 75)

top_variance_channels = (
    channel_variances
    .sort_values(
        ascending=False
    )
    .head(10)
)

print(
    top_variance_channels.to_string()
)


# ============================================================
# 13. EEG FEATURE DISTRIBUTION
# ============================================================

print("\n" + "=" * 75)
print("7. EEG FEATURE DISTRIBUTIONS")
print("=" * 75)

# Select representative EEG channels for visualization.
# We don't create 64 separate plots at this stage.

selected_channels = [
    "A1",
    "A8",
    "A16",
    "A24",
    "A32",
    "B1",
    "B8",
    "B16",
    "B24",
    "B32"
]


plt.figure(
    figsize=(16, 10)
)

for channel in selected_channels:

    sns.kdeplot(
        data=df,
        x=channel,
        label=channel,
        fill=False
    )

plt.title(
    "Distribution of Representative EEG Channels",
    fontsize=16
)

plt.xlabel(
    "EEG Value"
)

plt.ylabel(
    "Density"
)

plt.legend(
    ncol=2
)

save_figure(
    "06_eeg_feature_distributions.png"
)


# ============================================================
# 14. EEG CORRELATION MATRIX
# ============================================================

print("\n" + "=" * 75)
print("8. EEG CHANNEL CORRELATION")
print("=" * 75)

correlation_matrix = (
    df[EEG_CHANNELS]
    .corr()
)

print(
    "\nAverage absolute channel correlation:"
)

mean_absolute_correlation = (
    correlation_matrix
    .where(
        ~np.eye(
            len(EEG_CHANNELS),
            dtype=bool
        )
    )
    .abs()
    .stack()
    .mean()
)

print(
    f"{mean_absolute_correlation:.4f}"
)


# ------------------------------------------------------------
# Correlation heatmap
# ------------------------------------------------------------

plt.figure(
    figsize=(16, 14)
)

sns.heatmap(
    correlation_matrix,
    cmap="coolwarm",
    center=0,
    square=True
)

plt.title(
    "EEG Channel Correlation Matrix",
    fontsize=16
)

save_figure(
    "07_eeg_correlation_matrix.png"
)


# ============================================================
# 15. CLASS-WISE EEG MEAN
# ============================================================

print("\n" + "=" * 75)
print("9. CLASS-WISE EEG MEAN")
print("=" * 75)

class_means = (
    df.groupby(
        TARGET_COLUMN
    )[EEG_CHANNELS]
    .mean()
)

print(
    class_means
)


# ============================================================
# 16. CLASS-WISE CHANNEL VARIANCE
# ============================================================

print("\n" + "=" * 75)
print("10. CLASS-WISE EEG VARIANCE")
print("=" * 75)

class_variances = (
    df.groupby(
        TARGET_COLUMN
    )[EEG_CHANNELS]
    .var()
)

print(
    class_variances
)


# ============================================================
# 17. SUBJECT-WISE EEG MEAN
# ============================================================

print("\n" + "=" * 75)
print("11. SUBJECT-WISE EEG MEAN")
print("=" * 75)

subject_means = (
    df.groupby(
        "subject_id"
    )[EEG_CHANNELS]
    .mean()
)

print(
    subject_means
)


# ============================================================
# 18. OUTLIER ANALYSIS
# ============================================================

print("\n" + "=" * 75)
print("12. EEG OUTLIER ANALYSIS")
print("=" * 75)

outlier_summary = []

for channel in EEG_CHANNELS:

    values = df[channel]

    q1 = values.quantile(0.25)

    q3 = values.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr

    upper_bound = q3 + 1.5 * iqr

    outliers = (
        (values < lower_bound)
        |
        (values > upper_bound)
    )

    outlier_count = int(
        outliers.sum()
    )

    outlier_percentage = (
        outlier_count
        /
        len(values)
        *
        100
    )

    outlier_summary.append(
        {
            "channel": channel,
            "outlier_count": outlier_count,
            "outlier_percentage": outlier_percentage
        }
    )


outlier_df = pd.DataFrame(
    outlier_summary
)

outlier_df = (
    outlier_df
    .sort_values(
        "outlier_percentage",
        ascending=False
    )
)

print(
    outlier_df.to_string(index=False)
)


# ------------------------------------------------------------
# Save outlier report
# ------------------------------------------------------------

outlier_path = (
    RESULTS_DIR
    / "eeg_outlier_report.csv"
)

outlier_df.to_csv(
    outlier_path,
    index=False
)

print(
    f"\n✓ Saved: {outlier_path}"
)


# ============================================================
# 19. SAVE CHANNEL STATISTICS
# ============================================================

print("\n" + "=" * 75)
print("13. SAVING CHANNEL STATISTICS")
print("=" * 75)

channel_statistics = pd.DataFrame(
    {
        "mean": df[EEG_CHANNELS].mean(),
        "std": df[EEG_CHANNELS].std(),
        "variance": df[EEG_CHANNELS].var(),
        "minimum": df[EEG_CHANNELS].min(),
        "maximum": df[EEG_CHANNELS].max(),
        "median": df[EEG_CHANNELS].median()
    }
)

channel_statistics_path = (
    RESULTS_DIR
    / "eeg_channel_statistics.csv"
)

channel_statistics.to_csv(
    channel_statistics_path
)

print(
    f"✓ Saved: {channel_statistics_path}"
)


# ============================================================
# 20. SAVE CLASS STATISTICS
# ============================================================

class_statistics = (
    df.groupby(
        [
            TARGET_COLUMN,
            LABEL_NAME_COLUMN
        ]
    )[EEG_CHANNELS]
    .mean()
)

class_statistics_path = (
    RESULTS_DIR
    / "eeg_class_statistics.csv"
)

class_statistics.to_csv(
    class_statistics_path
)

print(
    f"✓ Saved: {class_statistics_path}"
)


# ============================================================
# 21. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("EDA SUMMARY")
print("=" * 75)

print(f"Samples analyzed       : {len(df)}")
print(f"Subjects analyzed      : {df['subject_id'].nunique()}")
print(f"EEG channels analyzed  : {len(EEG_CHANNELS)}")
print(f"Classes analyzed       : {df[TARGET_COLUMN].nunique()}")
print(f"Missing values         : {missing}")
print(
    f"Average |correlation|  : "
    f"{mean_absolute_correlation:.4f}"
)

print("\nGenerated visualizations:")

eda_files = [
    "01_class_distribution.png",
    "02_subject_distribution.png",
    "03_subject_class_heatmap.png",
    "04_channel_means.png",
    "05_channel_variance.png",
    "06_eeg_feature_distributions.png",
    "07_eeg_correlation_matrix.png",
]

for filename in eda_files:

    print(
        f"  ✓ results/eeg/{filename}"
    )

print("\n" + "=" * 75)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 75)

print(
    "\nNext stage:"
    "\nBaseline ML model training"
)