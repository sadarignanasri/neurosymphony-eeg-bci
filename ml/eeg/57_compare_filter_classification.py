"""
NeuroSymphony — Step 57
Compare 22 EEG filtering conditions using Logistic Regression.

Protocol:
- Use the existing Step 45 train/validation/test split.
- Fit StandardScaler on training data only.
- Rank filters using validation metrics only.
- Evaluate the selected filter on the test set only.
"""

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# 1. PATHS AND CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_DIR = (
    PROJECT_ROOT
    / "results"
    / "eeg"
    / "filter_analysis"
    / "sub01"
    / "step55"
)

SPLIT_FILE = (
    PROJECT_ROOT
    / "results"
    / "eeg"
    / "band_analysis"
    / "step45"
    / "step45_split_indices.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "eeg"
    / "filter_analysis"
    / "sub01"
    / "step57"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_SUFFIXES = (
    "_delta_power",
    "_theta_power",
    "_alpha_power",
    "_beta_power",
    "_gamma_power",
)

EXPECTED_CONDITIONS = 22
EXPECTED_TRIALS = 720
EXPECTED_FEATURES = 320
EXPECTED_CLASSES = 8


# ---------------------------------------------------------
# 2. METRIC HELPER
# ---------------------------------------------------------

def calculate_metrics(y_true, y_pred):
    """Calculate classification metrics."""

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
    }


# ---------------------------------------------------------
# 3. LOAD AND VALIDATE THE SAVED SPLIT
# ---------------------------------------------------------

print("=" * 70)
print("NEUROSYMPHONY — STEP 57: FILTER CLASSIFICATION")
print("=" * 70)

if not SPLIT_FILE.exists():
    raise FileNotFoundError(
        f"Saved split file not found: {SPLIT_FILE}"
    )

if not FEATURE_DIR.exists():
    raise FileNotFoundError(
        f"Feature directory not found: {FEATURE_DIR}"
    )

split_df = pd.read_csv(SPLIT_FILE)

required_split_columns = {"dataset_index", "split"}

if not required_split_columns.issubset(split_df.columns):
    raise ValueError(
        "Split file must contain dataset_index and split."
    )

if split_df["dataset_index"].duplicated().any():
    raise ValueError(
        "Duplicate dataset_index values in the split file."
    )

if len(split_df) != EXPECTED_TRIALS:
    raise ValueError(
        f"Expected {EXPECTED_TRIALS} split entries, "
        f"found {len(split_df)}."
    )

allowed_splits = {"train", "validation", "test"}
actual_splits = set(split_df["split"].unique())

if actual_splits != allowed_splits:
    raise ValueError(
        f"Unexpected split names: {actual_splits}"
    )

split_counts = split_df["split"].value_counts()

expected_counts = {
    "train": 504,
    "validation": 108,
    "test": 108,
}

for name, expected_count in expected_counts.items():
    actual_count = int(split_counts.get(name, 0))

    if actual_count != expected_count:
        raise ValueError(
            f"{name}: expected {expected_count}, "
            f"found {actual_count}."
        )

print("\nSaved split verified:")
print(split_counts.to_string())


# ---------------------------------------------------------
# 4. SELECT ONLY THE 22 FEATURE DATASETS
# ---------------------------------------------------------

feature_files = sorted(
    path
    for path in FEATURE_DIR.glob("step55_*.csv")
    if not path.name.endswith("_timing.csv")
    and path.name != "step55_filter_conditions.csv"
)

if len(feature_files) != EXPECTED_CONDITIONS:
    raise ValueError(
        f"Expected {EXPECTED_CONDITIONS} feature datasets, "
        f"found {len(feature_files)}."
    )

print(f"\nFeature datasets found: {len(feature_files)}")


# ---------------------------------------------------------
# 5. VALIDATE FEATURE STRUCTURE AND LABEL ALIGNMENT
# ---------------------------------------------------------

reference_df = pd.read_csv(feature_files[0])

feature_columns = [
    column
    for column in reference_df.columns
    if column.endswith(FEATURE_SUFFIXES)
]

if len(feature_columns) != EXPECTED_FEATURES:
    raise ValueError(
        f"Expected {EXPECTED_FEATURES} features, "
        f"found {len(feature_columns)}."
    )

if len(reference_df) != EXPECTED_TRIALS:
    raise ValueError("Reference dataset has an unexpected row count.")

if reference_df["trial_index"].duplicated().any():
    raise ValueError("Duplicate trial_index in reference dataset.")

if reference_df["label"].nunique() != EXPECTED_CLASSES:
    raise ValueError("Reference dataset does not contain 8 classes.")

expected_trial_indices = set(
    reference_df["trial_index"].astype(int)
)

saved_dataset_indices = set(
    split_df["dataset_index"].astype(int)
)

if expected_trial_indices != saved_dataset_indices:
    raise ValueError(
        "The trial_index values do not match the saved "
        "dataset_index values. Stopping to avoid a wrong split."
    )

reference_labels = (
    reference_df
    .sort_values("trial_index")["label"]
    .to_numpy()
)

reference_feature_names = feature_columns.copy()

for path in feature_files:
    check_df = pd.read_csv(path)

    check_features = [
        column
        for column in check_df.columns
        if column.endswith(FEATURE_SUFFIXES)
    ]

    if len(check_df) != EXPECTED_TRIALS:
        raise ValueError(f"Unexpected row count: {path.name}")

    if check_features != reference_feature_names:
        raise ValueError(f"Feature mismatch: {path.name}")

    if check_df["trial_index"].duplicated().any():
        raise ValueError(f"Duplicate trial indices: {path.name}")

    check_df = check_df.sort_values("trial_index")

    if not np.array_equal(
        check_df["trial_index"].to_numpy(),
        np.sort(list(expected_trial_indices)),
    ):
        raise ValueError(f"Trial alignment error: {path.name}")

    if not np.array_equal(
        check_df["label"].to_numpy(),
        reference_labels,
    ):
        raise ValueError(f"Label alignment error: {path.name}")

print(f"Feature count: {len(feature_columns)}")
print("Trial and label alignment: passed")


# ---------------------------------------------------------
# 6. TRAIN ON TRAINING DATA; RANK ON VALIDATION DATA
# ---------------------------------------------------------

validation_results = []
fitted_models = {}

for file_number, path in enumerate(feature_files, start=1):

    condition = path.stem.replace("step55_", "")

    print(
        f"\n[{file_number}/{len(feature_files)}] "
        f"Evaluating: {condition}"
    )

    df = pd.read_csv(path)

    # Join using the saved split indices, not row assumptions.
    df["trial_index"] = df["trial_index"].astype(int)

    split_mapping = split_df.rename(
        columns={"dataset_index": "trial_index"}
    )

    df = df.merge(
        split_mapping,
        on="trial_index",
        how="left",
        validate="one_to_one",
        indicator=True,
        sort=False,
    )

    if not (df["_merge"] == "both").all():
        raise ValueError(
            f"Some trials did not match the split: {condition}"
        )

    if df["split"].isna().any():
        raise ValueError(
            f"Missing split assignments: {condition}"
        )

    X = df[feature_columns].to_numpy(dtype=np.float64)
    y = df["label"].to_numpy(dtype=int)
    split_values = df["split"].to_numpy()

    if not np.isfinite(X).all():
        raise ValueError(
            f"Non-finite features found in {condition}"
        )

    train_mask = split_values == "train"
    val_mask = split_values == "validation"

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_val = X[val_mask]
    y_val = y[val_mask]

    if len(X_train) != 504 or len(X_val) != 108:
        raise ValueError(
            f"Unexpected split sizes for {condition}"
        )

    # Scaler learns from training data only.
    # Logistic Regression settings are held constant.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                max_iter=5000,
                random_state=42,
                solver="lbfgs",
            ),
        ),
    ])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline.fit(X_train, y_train)

    convergence_warning = any(
        issubclass(w.category, ConvergenceWarning)
        for w in caught
    )

    # Validation only — no test-set predictions here.
    y_val_pred = pipeline.predict(X_val)
    metrics = calculate_metrics(y_val, y_val_pred)

    result = {
        "condition": condition,
        "feature_file": path.name,
        "train_trials": len(X_train),
        "validation_trials": len(X_val),
        "validation_accuracy": metrics["accuracy"],
        "validation_balanced_accuracy": metrics[
            "balanced_accuracy"
        ],
        "validation_macro_f1": metrics["macro_f1"],
        "convergence_warning": convergence_warning,
    }

    validation_results.append(result)
    fitted_models[condition] = pipeline

    print(
        f"  Validation accuracy: "
        f"{metrics['accuracy'] * 100:.2f}%"
    )
    print(
        f"  Balanced accuracy: "
        f"{metrics['balanced_accuracy'] * 100:.2f}%"
    )
    print(
        f"  Macro F1: {metrics['macro_f1']:.4f}"
    )

    if convergence_warning:
        print("  WARNING: Logistic Regression convergence warning.")


# ---------------------------------------------------------
# 7. RANK FILTERS USING VALIDATION METRICS ONLY
# ---------------------------------------------------------

results_df = pd.DataFrame(validation_results)

results_df = results_df.sort_values(
    by=[
        "validation_accuracy",
        "validation_balanced_accuracy",
        "validation_macro_f1",
    ],
    ascending=[False, False, False],
).reset_index(drop=True)

results_df.insert(
    0,
    "validation_rank",
    np.arange(1, len(results_df) + 1),
)

validation_output = (
    OUTPUT_DIR / "step57_validation_results.csv"
)

results_df.to_csv(validation_output, index=False)

print("\n" + "=" * 70)
print("FILTER RANKING — VALIDATION SET ONLY")
print("=" * 70)

print(
    results_df[
        [
            "validation_rank",
            "condition",
            "validation_accuracy",
            "validation_balanced_accuracy",
            "validation_macro_f1",
            "convergence_warning",
        ]
    ].head(22).to_string(index=False)
)

winner = results_df.iloc[0]
winner_condition = winner["condition"]
winner_model = fitted_models[winner_condition]

print("\nSelected filter based on validation metrics:")
print(winner_condition)
print(
    f"Validation accuracy: "
    f"{winner['validation_accuracy'] * 100:.2f}%"
)


# ---------------------------------------------------------
# 8. EVALUATE ONLY THE SELECTED FILTER ON THE TEST SET
# ---------------------------------------------------------

winner_path = FEATURE_DIR / winner["feature_file"]
test_df = pd.read_csv(winner_path)

test_df["trial_index"] = test_df["trial_index"].astype(int)

split_mapping = split_df.rename(
    columns={"dataset_index": "trial_index"}
)

test_df = test_df.merge(
    split_mapping,
    on="trial_index",
    how="left",
    validate="one_to_one",
)

test_df = test_df.loc[test_df["split"] == "test"].copy()

if len(test_df) != 108:
    raise ValueError(
        f"Expected 108 test trials, found {len(test_df)}."
    )

X_test = test_df[feature_columns].to_numpy(dtype=np.float64)
y_test = test_df["label"].to_numpy(dtype=int)

# No fitting or model selection using test data.
y_test_pred = winner_model.predict(X_test)
test_metrics = calculate_metrics(y_test, y_test_pred)

test_result = {
    "selected_condition": winner_condition,
    "selection_basis": "validation accuracy, balanced accuracy, macro F1",
    "test_trials": len(X_test),
    "test_accuracy": test_metrics["accuracy"],
    "test_balanced_accuracy": test_metrics["balanced_accuracy"],
    "test_macro_f1": test_metrics["macro_f1"],
}

test_output = (
    OUTPUT_DIR / "step57_selected_filter_test_result.csv"
)

pd.DataFrame([test_result]).to_csv(test_output, index=False)

# Save the selected training-fitted pipeline for reproducibility.
model_output = OUTPUT_DIR / "step57_selected_pipeline.joblib"
joblib.dump(winner_model, model_output)


# ---------------------------------------------------------
# 9. SAVE A SUMMARY
# ---------------------------------------------------------

summary_output = OUTPUT_DIR / "step57_summary.txt"

summary_lines = [
    "NEUROSYMPHONY — STEP 57 SUMMARY",
    "=" * 50,
    f"Feature datasets evaluated: {len(feature_files)}",
    f"Features per trial: {len(feature_columns)}",
    f"Training trials: {expected_counts['train']}",
    f"Validation trials: {expected_counts['validation']}",
    f"Test trials: {expected_counts['test']}",
    "",
    "Selected filter (validation ranking):",
    winner_condition,
    f"Validation accuracy: {winner['validation_accuracy']:.6f}",
    (
        "Validation balanced accuracy: "
        f"{winner['validation_balanced_accuracy']:.6f}"
    ),
    f"Validation macro F1: {winner['validation_macro_f1']:.6f}",
    "",
    "Held-out test metrics for selected filter only:",
    f"Test accuracy: {test_metrics['accuracy']:.6f}",
    f"Test balanced accuracy: {test_metrics['balanced_accuracy']:.6f}",
    f"Test macro F1: {test_metrics['macro_f1']:.6f}",
    "",
    "Test data was not used to rank or select filters.",
    "Results apply to subject 01 and this saved split.",
]

summary_output.write_text(
    "\n".join(summary_lines) + "\n",
    encoding="utf-8",
)

print("\n" + "=" * 70)
print("HELD-OUT TEST RESULTS — SELECTED FILTER ONLY")
print("=" * 70)
print(f"Selected filter: {winner_condition}")
print(f"Test accuracy: {test_metrics['accuracy'] * 100:.2f}%")
print(
    "Test balanced accuracy: "
    f"{test_metrics['balanced_accuracy'] * 100:.2f}%"
)
print(f"Test macro F1: {test_metrics['macro_f1']:.4f}")

print("\nSaved outputs:")
print(f"Validation ranking: {validation_output}")
print(f"Selected-filter test result: {test_output}")
print(f"Selected pipeline: {model_output}")
print(f"Summary: {summary_output}")

print("\nSTEP 57 COMPLETED SUCCESSFULLY.")