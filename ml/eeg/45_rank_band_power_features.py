from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_classif


# ============================================================
# STEP 45
# RANK THE 320 BAND-POWER FEATURES
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "eeg"
    / "band_analysis"
    / "step44"
    / "step44_sub01_band_power_only.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "eeg"
    / "band_analysis"
    / "step45"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.15

VALIDATION_SIZE = 0.15

TOP_N = 50


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STEP 45 — BAND-POWER FEATURE RANKING")
print("=" * 70)

print("\nInput:")
print(INPUT_PATH)


# ============================================================
# CHECK INPUT
# ============================================================

if not INPUT_PATH.exists():

    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_PATH}"
    )


# ============================================================
# LOAD FEATURE MATRIX
# ============================================================

df = pd.read_csv(
    INPUT_PATH
)

print(
    f"\nDataset shape: "
    f"{df.shape}"
)


# ============================================================
# METADATA COLUMNS
# ============================================================

METADATA_COLUMNS = [
    "subject",
    "trial",
    "label",
    "label_name",
    "event_row",
    "start_sample",
    "duration_seconds",
]


# ============================================================
# IDENTIFY POWER FEATURES
# ============================================================

feature_columns = [
    column
    for column in df.columns
    if column.endswith("_power")
    and not column.endswith("_logpower")
]


print(
    f"\nNumber of power features: "
    f"{len(feature_columns)}"
)


if len(feature_columns) != 320:

    raise ValueError(
        f"Expected 320 power features, "
        f"found {len(feature_columns)}."
    )


# ============================================================
# INPUT / TARGET
# ============================================================

X = df[
    feature_columns
].to_numpy(
    dtype=np.float64
)

y = df[
    "label"
].to_numpy()


# ============================================================
# CHECK DATA
# ============================================================

if np.isnan(X).any():

    raise ValueError(
        "NaN values found in feature matrix."
    )


if np.isinf(X).any():

    raise ValueError(
        "Infinite values found in feature matrix."
    )


print(
    "\n✓ No NaN values"
)

print(
    "✓ No infinite values"
)


# ============================================================
# STRATIFIED TRAIN / VALIDATION / TEST SPLIT
# ============================================================

# First:
# 85% temporary training data
# 15% test data

X_temp, X_test, y_temp, y_test, idx_temp, idx_test = (
    train_test_split(
        X,
        y,
        np.arange(len(df)),
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )
)


# ------------------------------------------------------------
# Split temporary set into:
#
# 70% final training
# 15% validation
# ------------------------------------------------------------

validation_fraction = (
    VALIDATION_SIZE
    / (1.0 - TEST_SIZE)
)


X_train, X_val, y_train, y_val, idx_train, idx_val = (
    train_test_split(
        X_temp,
        y_temp,
        idx_temp,
        test_size=validation_fraction,
        stratify=y_temp,
        random_state=RANDOM_STATE
    )
)


# ============================================================
# VERIFY SPLIT
# ============================================================

print("\n" + "=" * 70)
print("DATA SPLIT")
print("=" * 70)

print(
    f"\nTraining   : {len(X_train)}"
)

print(
    f"Validation : {len(X_val)}"
)

print(
    f"Test       : {len(X_test)}"
)

print(
    f"Total      : "
    f"{len(X_train) + len(X_val) + len(X_test)}"
)


# ============================================================
# FEATURE RANKING
# ============================================================

print("\n" + "=" * 70)
print("ANOVA FEATURE RANKING")
print("=" * 70)

print(
    "\nIMPORTANT:"
)

print(
    "Feature ranking is calculated using "
    "TRAINING DATA ONLY."
)


# ------------------------------------------------------------
# Calculate ANOVA F-score and p-value
# ------------------------------------------------------------

f_scores, p_values = f_classif(
    X_train,
    y_train
)


# ============================================================
# CREATE RANKING TABLE
# ============================================================

ranking_df = pd.DataFrame({

    "feature":
        feature_columns,

    "f_score":
        f_scores,

    "p_value":
        p_values,
})


# ------------------------------------------------------------
# Add channel and frequency band
# ------------------------------------------------------------

ranking_df["channel"] = (
    ranking_df["feature"]
    .str.replace(
        r"_(delta|theta|alpha|beta|gamma)_power$",
        "",
        regex=True
    )
)

ranking_df["band"] = (
    ranking_df["feature"]
    .str.extract(
        r"_(delta|theta|alpha|beta|gamma)_power$"
    )[0]
)


# ============================================================
# SORT FEATURES
# ============================================================

ranking_df = ranking_df.sort_values(
    by="f_score",
    ascending=False
).reset_index(
    drop=True
)


ranking_df["rank"] = (
    np.arange(len(ranking_df))
    + 1
)


# ============================================================
# REORDER COLUMNS
# ============================================================

ranking_df = ranking_df[
    [
        "rank",
        "feature",
        "channel",
        "band",
        "f_score",
        "p_value",
    ]
]


# ============================================================
# SAVE COMPLETE RANKING
# ============================================================

complete_path = (
    OUTPUT_DIR
    / "step45_all_320_feature_ranking.csv"
)

ranking_df.to_csv(
    complete_path,
    index=False
)


# ============================================================
# SAVE TOP 50
# ============================================================

top50_df = ranking_df.head(
    TOP_N
)

top50_path = (
    OUTPUT_DIR
    / "step45_top50_features.csv"
)

top50_df.to_csv(
    top50_path,
    index=False
)


# ============================================================
# SAVE TOP 20
# ============================================================

top20_df = ranking_df.head(
    20
)

top20_path = (
    OUTPUT_DIR
    / "step45_top20_features.csv"
)

top20_df.to_csv(
    top20_path,
    index=False
)


# ============================================================
# BAND SUMMARY
# ============================================================

band_summary = (
    ranking_df
    .groupby("band")
    .agg(
        mean_f_score=(
            "f_score",
            "mean"
        ),
        max_f_score=(
            "f_score",
            "max"
        ),
        median_f_score=(
            "f_score",
            "median"
        ),
        significant_features=(
            "p_value",
            lambda x: (x < 0.05).sum()
        )
    )
    .sort_values(
        "mean_f_score",
        ascending=False
    )
)


band_summary_path = (
    OUTPUT_DIR
    / "step45_band_summary.csv"
)

band_summary.to_csv(
    band_summary_path
)


# ============================================================
# CHANNEL SUMMARY
# ============================================================

channel_summary = (
    ranking_df
    .groupby("channel")
    .agg(
        mean_f_score=(
            "f_score",
            "mean"
        ),
        max_f_score=(
            "f_score",
            "max"
        ),
        significant_features=(
            "p_value",
            lambda x: (x < 0.05).sum()
        )
    )
    .sort_values(
        "mean_f_score",
        ascending=False
    )
)


channel_summary_path = (
    OUTPUT_DIR
    / "step45_channel_summary.csv"
)

channel_summary.to_csv(
    channel_summary_path
)


# ============================================================
# DISPLAY TOP FEATURES
# ============================================================

pd.set_option(
    "display.max_rows",
    50
)

pd.set_option(
    "display.float_format",
    lambda x: f"{x:.6e}"
)


print("\n" + "=" * 70)
print("TOP 20 FEATURES")
print("=" * 70)

print(
    top20_df.to_string(
        index=False
    )
)


# ============================================================
# DISPLAY BAND SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("BAND SUMMARY")
print("=" * 70)

print(
    band_summary
)


# ============================================================
# DISPLAY TOP FEATURE PER BAND
# ============================================================

print("\n" + "=" * 70)
print("BEST FEATURE IN EACH BAND")
print("=" * 70)

best_per_band = (
    ranking_df
    .sort_values(
        "f_score",
        ascending=False
    )
    .groupby(
        "band",
        as_index=False
    )
    .first()
)

print(
    best_per_band[
        [
            "rank",
            "feature",
            "channel",
            "band",
            "f_score",
            "p_value",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# SAVE SPLIT INDICES
# ============================================================

split_df = pd.DataFrame({

    "dataset_index":
        np.concatenate(
            [
                idx_train,
                idx_val,
                idx_test
            ]
        ),

    "split":
        (
            ["train"] * len(idx_train)
            +
            ["validation"] * len(idx_val)
            +
            ["test"] * len(idx_test)
        )
})


split_df = split_df.sort_values(
    "dataset_index"
)


split_path = (
    OUTPUT_DIR
    / "step45_split_indices.csv"
)

split_df.to_csv(
    split_path,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 45 COMPLETE")
print("=" * 70)

print(
    "\nFeatures analyzed : "
    f"{len(feature_columns)}"
)

print(
    f"Training trials   : {len(X_train)}"
)

print(
    f"Validation trials : {len(X_val)}"
)

print(
    f"Test trials       : {len(X_test)}"
)

print(
    "\nFeature ranking was "
    "calculated using TRAINING DATA ONLY."
)

print("\nSaved files:")

print(complete_path)

print(top50_path)

print(top20_path)

print(band_summary_path)

print(channel_summary_path)

print(split_path)