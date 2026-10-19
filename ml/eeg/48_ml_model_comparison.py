"""
STEP 48 — ML MODEL COMPARISON
==============================

Purpose
-------
Compare different classical ML classifiers using the SAME
320 absolute EEG band-power features.

Feature representation
----------------------
64 EEG channels × 5 frequency bands = 320 features

Bands:
    Delta : 1–4 Hz
    Theta : 4–8 Hz
    Alpha : 8–13 Hz
    Beta  : 13–30 Hz
    Gamma : 30–40 Hz

Models:
    1. Logistic Regression
    2. Linear SVM
    3. RBF SVM
    4. Random Forest
    5. Gradient Boosting

Experimental rules
------------------
- Same 320 features for every model.
- Same Step 45 train/validation/test split.
- StandardScaler fitted ONLY on training data.
- Model selection based ONLY on validation performance.
- Test set is NOT used in this step.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    PROJECT_ROOT
    / "results"
    / "eeg"
    / "band_analysis"
    / "step44"
    / "step44_sub01_band_power_only.csv"
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
    / "band_analysis"
    / "step48"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STEP 48 — ML MODEL COMPARISON")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # 1. LOAD FEATURE MATRIX
    # ========================================================

    print("\nLoading feature matrix...")

    df = pd.read_csv(
        FEATURE_FILE
    )

    print(
        f"Dataset shape: {df.shape}"
    )

    # ========================================================
    # 2. IDENTIFY METADATA AND FEATURES
    # ========================================================

    # These are the ACTUAL metadata columns present in Step 44.
    metadata_columns = [
        "subject",
        "trial",
        "label",
        "label_name",
        "event_row",
        "start_sample",
        "duration_seconds",
    ]

    # Verify all expected metadata columns exist.
    missing_metadata = [
        column
        for column in metadata_columns
        if column not in df.columns
    ]

    if missing_metadata:

        raise ValueError(
            "Missing expected metadata columns:\n"
            + "\n".join(missing_metadata)
        )

    # Everything except metadata is an EEG band-power feature.
    feature_columns = [
        column
        for column in df.columns
        if column not in metadata_columns
    ]

    print(
        f"Total features: {len(feature_columns)}"
    )

    # ========================================================
    # 3. VERIFY EXACTLY 320 BAND-POWER FEATURES
    # ========================================================

    if len(feature_columns) != 320:

        raise ValueError(
            f"Expected 320 absolute band-power features, "
            f"but found {len(feature_columns)}."
        )

    # Verify that every feature follows the expected naming
    # structure.
    expected_bands = {
        "delta",
        "theta",
        "alpha",
        "beta",
        "gamma",
    }

    invalid_features = []

    for feature in feature_columns:

        if not feature.endswith("_power"):
            invalid_features.append(feature)
            continue

        parts = feature.split("_")

        if len(parts) != 3:
            invalid_features.append(feature)
            continue

        channel = parts[0]
        band = parts[1]

        if band not in expected_bands:
            invalid_features.append(feature)

        if not (
            channel.startswith("A")
            or channel.startswith("B")
        ):
            invalid_features.append(feature)

    if invalid_features:

        raise ValueError(
            "Unexpected feature names detected:\n"
            + "\n".join(invalid_features[:20])
        )

    print(
        "Verified: 320 absolute band-power features."
    )

    # ========================================================
    # 4. CREATE FEATURE MATRIX
    # ========================================================

    X = df[feature_columns].values

    # Labels are strings in the Step 44 dataset.
    y = df["label"].values

    print(
        f"Feature matrix shape: {X.shape}"
    )

    print(
        f"Number of classes: "
        f"{len(np.unique(y))}"
    )

    # ========================================================
    # 5. VERIFY CLASS DISTRIBUTION
    # ========================================================

    print("\nOverall class distribution:")

    print(
        pd.Series(y)
        .value_counts()
        .sort_index()
    )

    # ========================================================
    # 6. LOAD EXACT STEP 45 SPLIT
    # ========================================================

    print("\nLoading Step 45 split...")

    split_df = pd.read_csv(
        SPLIT_FILE
    )

    required_split_columns = {
        "dataset_index",
        "split",
    }

    missing_split_columns = (
        required_split_columns
        - set(split_df.columns)
    )

    if missing_split_columns:

        raise ValueError(
            "Missing split columns:\n"
            + "\n".join(missing_split_columns)
        )

    train_idx = (
        split_df.loc[
            split_df["split"] == "train",
            "dataset_index"
        ]
        .astype(int)
        .values
    )

    val_idx = (
        split_df.loc[
            split_df["split"] == "validation",
            "dataset_index"
        ]
        .astype(int)
        .values
    )

    test_idx = (
        split_df.loc[
            split_df["split"] == "test",
            "dataset_index"
        ]
        .astype(int)
        .values
    )

    print(
        f"Train      : {len(train_idx)}"
    )

    print(
        f"Validation : {len(val_idx)}"
    )

    print(
        f"Test       : {len(test_idx)}"
    )

    # ========================================================
    # 7. SAFETY CHECK SPLIT SIZES
    # ========================================================

    if len(train_idx) != 504:
        raise ValueError(
            f"Expected 504 training samples, "
            f"found {len(train_idx)}."
        )

    if len(val_idx) != 108:
        raise ValueError(
            f"Expected 108 validation samples, "
            f"found {len(val_idx)}."
        )

    if len(test_idx) != 108:
        raise ValueError(
            f"Expected 108 test samples, "
            f"found {len(test_idx)}."
        )

    # ========================================================
    # 8. CHECK SPLIT OVERLAP
    # ========================================================

    train_set = set(train_idx)
    val_set = set(val_idx)
    test_set = set(test_idx)

    if train_set & val_set:
        raise ValueError(
            "Train and validation sets overlap."
        )

    if train_set & test_set:
        raise ValueError(
            "Train and test sets overlap."
        )

    if val_set & test_set:
        raise ValueError(
            "Validation and test sets overlap."
        )

    print(
        "Split overlap check: PASSED"
    )

    # ========================================================
    # 9. CREATE TRAIN / VALIDATION DATA
    # ========================================================
    #
    # Test data is deliberately NOT used here.
    # ========================================================

    X_train = X[train_idx]
    y_train = y[train_idx]

    X_val = X[val_idx]
    y_val = y[val_idx]

    # ========================================================
    # 10. STANDARDIZATION
    # ========================================================
    #
    # IMPORTANT:
    # Fit scaler ONLY on training data.
    # ========================================================

    print(
        "\nStandardizing features..."
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_val_scaled = scaler.transform(
        X_val
    )

    print(
        "Scaler fitted on training data only."
    )

    # ========================================================
    # 11. DEFINE ML MODELS
    # ========================================================

    models = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
                random_state=RANDOM_STATE,
            ),

        "Linear SVM":
            SVC(
                kernel="linear",
                C=1.0,
            ),

        "RBF SVM":
            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=300,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),

        "Gradient Boosting":
            GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=RANDOM_STATE,
            ),
    }

    # ========================================================
    # 12. TRAIN AND EVALUATE EACH MODEL
    # ========================================================

    results = []

    for model_name, model in models.items():

        print("\n")
        print("-" * 70)
        print(
            f"MODEL: {model_name}"
        )
        print("-" * 70)

        print(
            "Training..."
        )

        model.fit(
            X_train_scaled,
            y_train
        )

        # ----------------------------------------------------
        # TRAIN PREDICTIONS
        # ----------------------------------------------------

        y_train_pred = model.predict(
            X_train_scaled
        )

        # ----------------------------------------------------
        # VALIDATION PREDICTIONS
        # ----------------------------------------------------

        y_val_pred = model.predict(
            X_val_scaled
        )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        train_accuracy = accuracy_score(
            y_train,
            y_train_pred
        )

        validation_accuracy = accuracy_score(
            y_val,
            y_val_pred
        )

        validation_balanced_accuracy = (
            balanced_accuracy_score(
                y_val,
                y_val_pred
            )
        )

        validation_macro_f1 = f1_score(
            y_val,
            y_val_pred,
            average="macro"
        )

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        print(
            f"Train accuracy       : "
            f"{train_accuracy:.4f} "
            f"({train_accuracy * 100:.2f}%)"
        )

        print(
            f"Validation accuracy  : "
            f"{validation_accuracy:.4f} "
            f"({validation_accuracy * 100:.2f}%)"
        )

        print(
            f"Validation balanced  : "
            f"{validation_balanced_accuracy:.4f} "
            f"({validation_balanced_accuracy * 100:.2f}%)"
        )

        print(
            f"Validation macro F1  : "
            f"{validation_macro_f1:.4f}"
        )

        # ----------------------------------------------------
        # SAVE RESULT
        # ----------------------------------------------------

        results.append({
            "model": model_name,
            "train_accuracy": train_accuracy,
            "validation_accuracy":
                validation_accuracy,
            "validation_balanced_accuracy":
                validation_balanced_accuracy,
            "validation_macro_f1":
                validation_macro_f1,
        })

    # ========================================================
    # 13. CREATE RESULTS TABLE
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    # Rank models using validation accuracy.
    results_df = results_df.sort_values(
        by=[
            "validation_accuracy",
            "validation_balanced_accuracy",
            "validation_macro_f1",
        ],
        ascending=False,
    ).reset_index(
        drop=True
    )

    results_df.insert(
        0,
        "rank",
        np.arange(
            1,
            len(results_df) + 1
        )
    )

    # ========================================================
    # 14. DISPLAY FINAL COMPARISON
    # ========================================================

    print("\n")
    print("=" * 70)
    print("ML MODEL COMPARISON")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # ========================================================
    # 15. BEST MODEL
    # ========================================================

    best_model = results_df.iloc[0]

    print("\n")
    print("=" * 70)
    print("BEST MODEL BASED ON VALIDATION")
    print("=" * 70)

    print(
        f"Model                 : "
        f"{best_model['model']}"
    )

    print(
        f"Validation accuracy   : "
        f"{best_model['validation_accuracy']:.4f} "
        f"({best_model['validation_accuracy'] * 100:.2f}%)"
    )

    print(
        f"Balanced accuracy     : "
        f"{best_model['validation_balanced_accuracy']:.4f} "
        f"({best_model['validation_balanced_accuracy'] * 100:.2f}%)"
    )

    print(
        f"Macro F1              : "
        f"{best_model['validation_macro_f1']:.4f}"
    )

    # ========================================================
    # 16. SAVE RESULTS
    # ========================================================

    results_path = (
        OUTPUT_DIR
        / "step48_ml_model_comparison.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    # ========================================================
    # 17. SAVE EXPERIMENT INFORMATION
    # ========================================================

    experiment_info = pd.DataFrame([
        {
            "feature_type":
                "Absolute band power",
            "number_of_features":
                320,
            "channels":
                64,
            "bands":
                "Delta, Theta, Alpha, Beta, Gamma",
            "train_samples":
                len(train_idx),
            "validation_samples":
                len(val_idx),
            "test_samples":
                len(test_idx),
            "random_state":
                RANDOM_STATE,
            "test_used_for_model_selection":
                False,
        }
    ])

    experiment_info.to_csv(
        OUTPUT_DIR
        / "step48_experiment_info.csv",
        index=False
    )

    # ========================================================
    # 18. METHODOLOGICAL NOTE
    # ========================================================

    print("\n")
    print("=" * 70)
    print("METHODOLOGICAL NOTE")
    print("=" * 70)

    print(
        "\nAll models used the same 320 absolute "
        "band-power features."
    )

    print(
        "All models used the same Step 45 "
        "train/validation/test split."
    )

    print(
        "StandardScaler was fitted only on "
        "the training data."
    )

    print(
        "The test set was NOT used for model selection."
    )

    print(
        "The best model was selected using "
        "validation performance."
    )

    # ========================================================
    # 19. COMPLETE
    # ========================================================

    print("\n")
    print("=" * 70)
    print("STEP 48 COMPLETE")
    print("=" * 70)

    print(
        f"\nResults saved to:\n"
        f"{OUTPUT_DIR}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()