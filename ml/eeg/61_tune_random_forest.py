
"""
===============================================================================
NeuroSymphony — Step 61: Random Forest Hyperparameter Tuning
===============================================================================

Subject: 01
Features: Original Step 35 band-power features (720 x 320)
Splits: Exact saved Step 35 train / validation / test indices

Protocol:
    - Fit each candidate using training data only.
    - Select the configuration using validation macro F1.
    - Do not evaluate the test set in this step.
    - Save validation results and the selected trained pipeline.
    - Preserve previous experiment outputs.
===============================================================================
"""

from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

FEATURE_DIR = ROOT / "data" / "features" / "temporal_spectral_sub01"

FEATURE_FILE = FEATURE_DIR / "sub-01_band_power.npy"
METADATA_FILE = FEATURE_DIR / "sub-01_spectral_metadata.csv"

SPLIT_DIR = ROOT / "data" / "splits" / "sub01"

TRAIN_FILE = SPLIT_DIR / "train_indices.npy"
VAL_FILE = SPLIT_DIR / "val_indices.npy"
TEST_FILE = SPLIT_DIR / "test_indices.npy"

OUTPUT_DIR = (
    ROOT
    / "results"
    / "eeg"
    / "model_comparison"
    / "sub01"
    / "step61_random_forest_tuning"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

EXPECTED_TRIALS = 720
EXPECTED_FEATURES = 320
EXPECTED_CLASSES = 8


# =============================================================================
# LOAD FEATURES AND LABELS
# =============================================================================

print("=" * 78)
print("NEUROSYMPHONY — STEP 61: RANDOM FOREST HYPERPARAMETER TUNING")
print("=" * 78)

required_files = [
    FEATURE_FILE,
    METADATA_FILE,
    TRAIN_FILE,
    VAL_FILE,
    TEST_FILE,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

X = np.load(FEATURE_FILE)

if X.shape != (720, 64, 5):
    raise ValueError(
        f"Unexpected original feature shape: {X.shape}. "
        "Expected (720, 64, 5)."
    )

# Flatten channels and frequency bands:
# (720 trials, 64 channels, 5 bands) -> (720 trials, 320 features)
X = X.reshape(EXPECTED_TRIALS, EXPECTED_FEATURES)

if not np.isfinite(X).all():
    raise ValueError("Features contain NaN or infinite values.")

metadata = pd.read_csv(METADATA_FILE)

if len(metadata) != EXPECTED_TRIALS:
    raise ValueError(
        f"Expected 720 metadata rows, found {len(metadata)}."
    )

if "label" not in metadata.columns:
    raise ValueError("Metadata is missing the 'label' column.")

y = metadata["label"].to_numpy()

if pd.isna(y).any():
    raise ValueError("Missing labels detected.")

if len(np.unique(y)) != EXPECTED_CLASSES:
    raise ValueError(
        f"Expected 8 classes, found {len(np.unique(y))}."
    )

print(f"Feature shape: {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")


# =============================================================================
# LOAD AND VERIFY STEP 35 SPLITS
# =============================================================================

train_idx = np.load(TRAIN_FILE).astype(int)
val_idx = np.load(VAL_FILE).astype(int)
test_idx = np.load(TEST_FILE).astype(int)

if (
    len(train_idx) != 504
    or len(val_idx) != 108
    or len(test_idx) != 108
):
    raise ValueError(
        "Unexpected split sizes. Expected 504 train, 108 validation, "
        "and 108 test trials."
    )

for name, indices in [
    ("training", train_idx),
    ("validation", val_idx),
    ("test", test_idx),
]:
    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"Duplicate indices found in {name} split.")

    if np.any(indices < 0) or np.any(indices >= EXPECTED_TRIALS):
        raise ValueError(f"Out-of-range index found in {name} split.")

if (
    set(train_idx) & set(val_idx)
    or set(train_idx) & set(test_idx)
    or set(val_idx) & set(test_idx)
):
    raise ValueError("Overlap detected between the saved splits.")

all_indices = np.concatenate([train_idx, val_idx, test_idx])

if not np.array_equal(
    np.sort(all_indices),
    np.arange(EXPECTED_TRIALS),
):
    raise ValueError("The splits do not partition all 720 trials.")

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]

print("\nStep 35 split verification successful.")
print(f"Training trials:   {len(train_idx)}")
print(f"Validation trials: {len(val_idx)}")
print(f"Test trials:       {len(test_idx)}")
print("Test indices are used only to verify split integrity.")
print("Test-set performance will NOT be evaluated in Step 61.")


# =============================================================================
# DEFINE HYPERPARAMETER GRID
# =============================================================================

n_estimators_values = [100, 300, 500]
max_depth_values = [5, 10, None]
min_samples_leaf_values = [1, 3]
class_weight_values = [None, "balanced"]

configurations = []

for n_estimators in n_estimators_values:
    for max_depth in max_depth_values:
        for min_samples_leaf in min_samples_leaf_values:
            for class_weight in class_weight_values:
                configurations.append(
                    {
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "min_samples_leaf": min_samples_leaf,
                        "class_weight": class_weight,
                    }
                )

print(f"\nConfigurations to evaluate: {len(configurations)}")


# =============================================================================
# TRAIN AND EVALUATE USING TRAINING AND VALIDATION DATA
# =============================================================================

results = []
pipelines = []

for config_index, params in enumerate(configurations):

    print(
        f"\n[{config_index + 1}/{len(configurations)}] "
        f"trees={params['n_estimators']}, "
        f"depth={params['max_depth']}, "
        f"min_leaf={params['min_samples_leaf']}, "
        f"class_weight={params['class_weight']}"
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    **params,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    start_time = time.perf_counter()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline.fit(X_train, y_train)

    elapsed_seconds = time.perf_counter() - start_time

    train_predictions = pipeline.predict(X_train)
    val_predictions = pipeline.predict(X_val)

    row = {
        # Preserves the association with the trained pipeline.
        "configuration_index": config_index,
        **params,
        "training_accuracy": accuracy_score(
            y_train, train_predictions
        ),
        "validation_accuracy": accuracy_score(
            y_val, val_predictions
        ),
        "validation_balanced_accuracy": balanced_accuracy_score(
            y_val, val_predictions
        ),
        "validation_macro_f1": f1_score(
            y_val,
            val_predictions,
            average="macro",
            zero_division=0,
        ),
        "training_seconds": elapsed_seconds,
        "warning_count": len(caught),
    }

    results.append(row)
    pipelines.append(pipeline)

    print(
        f"Validation accuracy: "
        f"{row['validation_accuracy']:.4f} | "
        f"Balanced accuracy: "
        f"{row['validation_balanced_accuracy']:.4f} | "
        f"Macro F1: {row['validation_macro_f1']:.4f}"
    )


# =============================================================================
# RANK CONFIGURATIONS USING VALIDATION MACRO F1
# =============================================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=[
        "validation_macro_f1",
        "validation_balanced_accuracy",
        "validation_accuracy",
    ],
    ascending=False,
).reset_index(drop=True)

results_df.insert(
    0,
    "validation_rank",
    np.arange(1, len(results_df) + 1),
)

ranking_file = OUTPUT_DIR / "step61_validation_results.csv"
results_df.to_csv(ranking_file, index=False)

print("\n" + "=" * 78)
print("STEP 61 — VALIDATION RANKING")
print("Primary selection metric: Validation macro F1")
print("=" * 78)

print(
    results_df.head(10).to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)


# =============================================================================
# SELECT AND SAVE THE BEST TRAINED MODEL
# =============================================================================

best_row = results_df.iloc[0]

# Use the original configuration index, not the sorted row position.
# This avoids matching None against pandas NaN.
best_position = int(best_row["configuration_index"])

if not 0 <= best_position < len(pipelines):
    raise ValueError(
        f"Invalid winning configuration index: {best_position}"
    )

best_pipeline = pipelines[best_position]

# Recover original Python values, preserving None rather than NaN.
best_params = configurations[best_position]

pipeline_file = OUTPUT_DIR / "step61_selected_pipeline.joblib"
joblib.dump(best_pipeline, pipeline_file)


# =============================================================================
# SAVE SUMMARY
# =============================================================================

summary_file = OUTPUT_DIR / "step61_summary.txt"

parameter_names = [
    "n_estimators",
    "max_depth",
    "min_samples_leaf",
    "class_weight",
]

metric_names = [
    "training_accuracy",
    "validation_accuracy",
    "validation_balanced_accuracy",
    "validation_macro_f1",
    "training_seconds",
    "warning_count",
]

with open(summary_file, "w", encoding="utf-8") as file:
    file.write("NEUROSYMPHONY — STEP 61 SUMMARY\n")
    file.write("=" * 60 + "\n")
    file.write("Subject: 01\n")
    file.write("Features: Original Step 35 band-power features\n")
    file.write("Feature count: 320\n")
    file.write("Split: Original Step 35 saved indices\n")
    file.write("Configurations evaluated: 36\n")
    file.write("Selection metric: Validation macro F1\n")
    file.write("Test evaluation: NOT performed\n\n")

    file.write("Selected hyperparameters:\n")
    for name in parameter_names:
        file.write(f"{name}: {best_params[name]}\n")

    file.write("\nSelected validation metrics:\n")
    for name in metric_names:
        file.write(f"{name}: {best_row[name]}\n")

    file.write("\nTop 10 validation configurations:\n")
    file.write(results_df.head(10).to_string(index=False))
    file.write("\n")


# =============================================================================
# FINAL REPORT
# =============================================================================

print("\n" + "=" * 78)
print("STEP 61 COMPLETED SUCCESSFULLY")
print("=" * 78)

print("\nSelected hyperparameters:")
for name in parameter_names:
    print(f"{name}: {best_params[name]}")

print("\nSelected validation metrics:")
for name in metric_names:
    print(f"{name}: {best_row[name]}")

print("\nSaved files:")
print(f"Validation ranking: {ranking_file}")
print(f"Selected pipeline:  {pipeline_file}")
print(f"Summary:            {summary_file}")

print("\nNo test-set performance was evaluated.")
