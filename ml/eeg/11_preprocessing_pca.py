"""
Neuro-Symphony
EEG Preprocessing Experiment 11: StandardScaler + PCA

Purpose:
    Evaluate whether PCA-based dimensionality reduction improves
    subject-independent EEG classification.

Pipeline:
    Raw EEG CSV
        -> StandardScaler
        -> PCA
        -> Classifier

Validation:
    Strict Leave-One-Subject-Out (LOSO)

Important:
    StandardScaler and PCA are fitted ONLY on the training subjects
    inside each LOSO fold to prevent data leakage.

PCA settings:
    - 90% explained variance
    - 95% explained variance
    - 99% explained variance

Models:
    1. Logistic Regression
    2. Linear SVM
    3. LDA
    4. KNN
    5. Random Forest
    6. Extra Trees

Outputs:
    results/eeg/preprocessing/pca_results.csv
    results/eeg/preprocessing/pca_predictions.csv
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ds004514_eeg_approved.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "eeg"
    / "preprocessing"
)

RESULTS_FILE = OUTPUT_DIR / "pca_results.csv"
PREDICTIONS_FILE = OUTPUT_DIR / "pca_predictions.csv"


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

PCA_VARIANCES = [0.90, 0.95, 0.99]

TARGET_COLUMNS = [
    "subject_id",
    "trial_id",
    "label_name",
    "label",
]

EXPECTED_EEG_CHANNELS = [
    f"A{i}" for i in range(1, 33)
] + [
    f"B{i}" for i in range(1, 33)
]


# ============================================================
# MODELS
# ============================================================

MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    ),

    "Linear SVM": LinearSVC(
        max_iter=5000,
        random_state=RANDOM_STATE,
    ),

    "LDA": LinearDiscriminantAnalysis(
        solver="lsqr",
        shrinkage="auto",
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5,
        n_jobs=-1,
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
}


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_dataset(df):
    """Validate the EEG CSV before running experiments."""

    print("\n" + "=" * 80)
    print("DATASET VALIDATION")
    print("=" * 80)

    print(f"Dataset shape: {df.shape}")

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = TARGET_COLUMNS + EXPECTED_EEG_CHANNELS

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print("✓ Required columns present")

    # --------------------------------------------------------
    # EEG feature columns
    # --------------------------------------------------------

    eeg_columns = [
        col for col in EXPECTED_EEG_CHANNELS
        if col in df.columns
    ]

    if len(eeg_columns) != 64:
        raise ValueError(
            f"Expected 64 EEG channels, found {len(eeg_columns)}"
        )

    print(f"✓ EEG channels: {len(eeg_columns)}")

    # --------------------------------------------------------
    # Numeric validation
    # --------------------------------------------------------

    non_numeric = [
        col
        for col in eeg_columns
        if not pd.api.types.is_numeric_dtype(df[col])
    ]

    if non_numeric:
        raise ValueError(
            f"Non-numeric EEG columns found: {non_numeric}"
        )

    print("✓ All EEG channels are numeric")

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing_values = df[eeg_columns].isna().sum().sum()

    if missing_values != 0:
        raise ValueError(
            f"Found {missing_values} missing EEG values"
        )

    print("✓ No missing EEG values")

    # --------------------------------------------------------
    # Infinite values
    # --------------------------------------------------------

    infinite_values = np.isinf(
        df[eeg_columns].to_numpy()
    ).sum()

    if infinite_values != 0:
        raise ValueError(
            f"Found {infinite_values} infinite EEG values"
        )

    print("✓ No infinite EEG values")

    # --------------------------------------------------------
    # Subjects
    # --------------------------------------------------------

    subjects = sorted(df["subject_id"].unique())

    print(f"✓ Subjects: {len(subjects)}")

    if len(subjects) != 12:
        print(
            f"WARNING: Expected 12 subjects, found {len(subjects)}"
        )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    labels = sorted(df["label"].unique())

    print(f"✓ Number of classes: {len(labels)}")
    print(f"✓ Labels: {labels}")

    if len(labels) != 8:
        print(
            f"WARNING: Expected 8 classes, found {len(labels)}"
        )

    # --------------------------------------------------------
    # Subject distribution
    # --------------------------------------------------------

    subject_counts = df.groupby("subject_id").size()

    print("\nSamples per subject:")
    print(subject_counts.to_string())

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    class_counts = df["label"].value_counts().sort_index()

    print("\nSamples per class:")
    print(class_counts.to_string())

    # --------------------------------------------------------
    # Duplicate trial keys
    # --------------------------------------------------------

    duplicate_keys = df.duplicated(
        subset=["subject_id", "trial_id"]
    ).sum()

    if duplicate_keys != 0:
        raise ValueError(
            f"Found {duplicate_keys} duplicate "
            f"(subject_id, trial_id) keys"
        )

    print("✓ No duplicate subject/trial keys")

    print("\n✓ DATASET VALIDATION PASSED")

    return eeg_columns, subjects


# ============================================================
# CREATE PIPELINE
# ============================================================

def create_pipeline(pca_variance, model):
    """
    Create a leakage-safe preprocessing + classification pipeline.

    StandardScaler and PCA are fitted only on training data
    within each LOSO fold.
    """

    pipeline = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler()
            ),
            (
                "pca",
                PCA(
                    n_components=pca_variance,
                    svd_solver="full",
                    random_state=RANDOM_STATE,
                ),
            ),
            (
                "classifier",
                clone(model)
            ),
        ]
    )

    return pipeline


# ============================================================
# RUN ONE PCA EXPERIMENT
# ============================================================

def run_pca_experiment(
    X,
    y,
    subjects,
    eeg_columns,
    pca_variance,
    model_name,
    model,
):
    """
    Run one PCA + classifier experiment using strict LOSO.
    """

    all_predictions = []

    fold_accuracies = []
    fold_f1_scores = []
    fold_balanced_accuracies = []

    fold_pca_components = []

    print(
        f"\n  PCA={int(pca_variance * 100)}% | "
        f"{model_name}"
    )

    start_time = time.time()

    # --------------------------------------------------------
    # LOSO
    # --------------------------------------------------------

    for fold_number, test_subject in enumerate(subjects, start=1):

        train_mask = subjects_series != test_subject
        test_mask = subjects_series == test_subject

        X_train = X.loc[train_mask, eeg_columns]
        X_test = X.loc[test_mask, eeg_columns]

        y_train = y.loc[train_mask]
        y_test = y.loc[test_mask]

        # ----------------------------------------------------
        # Pipeline
        # ----------------------------------------------------

        pipeline = create_pipeline(
            pca_variance=pca_variance,
            model=model,
        )

        pipeline.fit(
            X_train,
            y_train,
        )

        y_pred = pipeline.predict(X_test)

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            y_pred,
        )

        macro_f1 = f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )

        balanced_accuracy = balanced_accuracy_score(
            y_test,
            y_pred,
        )

        # ----------------------------------------------------
        # Number of PCA components
        # ----------------------------------------------------

        pca_step = pipeline.named_steps["pca"]

        n_components = pca_step.n_components_

        fold_pca_components.append(
            n_components
        )

        fold_accuracies.append(accuracy)
        fold_f1_scores.append(macro_f1)
        fold_balanced_accuracies.append(
            balanced_accuracy
        )

        # ----------------------------------------------------
        # Store predictions
        # ----------------------------------------------------

        test_indices = X_test.index

        for index, true_label, predicted_label in zip(
            test_indices,
            y_test,
            y_pred,
        ):

            all_predictions.append(
                {
                    "pca_variance": pca_variance,
                    "pca_variance_percent": int(
                        pca_variance * 100
                    ),
                    "model": model_name,
                    "subject_id": df.loc[
                        index,
                        "subject_id"
                    ],
                    "trial_id": df.loc[
                        index,
                        "trial_id"
                    ],
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                }
            )

        print(
            f"    Fold {fold_number:02d}/12 | "
            f"Test={test_subject} | "
            f"Acc={accuracy * 100:.3f}% | "
            f"F1={macro_f1 * 100:.3f}% | "
            f"PCA components={n_components}"
        )

    # --------------------------------------------------------
    # Aggregate metrics
    # --------------------------------------------------------

    elapsed = time.time() - start_time

    mean_accuracy = np.mean(
        fold_accuracies
    )

    std_accuracy = np.std(
        fold_accuracies
    )

    mean_f1 = np.mean(
        fold_f1_scores
    )

    std_f1 = np.std(
        fold_f1_scores
    )

    mean_balanced_accuracy = np.mean(
        fold_balanced_accuracies
    )

    std_balanced_accuracy = np.std(
        fold_balanced_accuracies
    )

    mean_components = np.mean(
        fold_pca_components
    )

    min_components = np.min(
        fold_pca_components
    )

    max_components = np.max(
        fold_pca_components
    )

    print(
        f"\n    RESULT | "
        f"PCA {int(pca_variance * 100)}% | "
        f"{model_name}"
    )

    print(
        f"    Accuracy: "
        f"{mean_accuracy * 100:.3f}% "
        f"± {std_accuracy * 100:.3f}%"
    )

    print(
        f"    Macro F1: "
        f"{mean_f1 * 100:.3f}% "
        f"± {std_f1 * 100:.3f}%"
    )

    print(
        f"    Balanced Accuracy: "
        f"{mean_balanced_accuracy * 100:.3f}% "
        f"± {std_balanced_accuracy * 100:.3f}%"
    )

    print(
        f"    PCA components: "
        f"{mean_components:.1f} "
        f"(range {min_components}-{max_components})"
    )

    print(
        f"    Time: {elapsed:.1f} seconds"
    )

    result = {
        "pca_variance": pca_variance,
        "pca_variance_percent": int(
            pca_variance * 100
        ),
        "model": model_name,
        "accuracy": mean_accuracy,
        "accuracy_percent": mean_accuracy * 100,
        "accuracy_std": std_accuracy,
        "accuracy_std_percent": std_accuracy * 100,
        "macro_f1": mean_f1,
        "macro_f1_percent": mean_f1 * 100,
        "macro_f1_std": std_f1,
        "macro_f1_std_percent": std_f1 * 100,
        "balanced_accuracy": mean_balanced_accuracy,
        "balanced_accuracy_percent": (
            mean_balanced_accuracy * 100
        ),
        "balanced_accuracy_std": std_balanced_accuracy,
        "balanced_accuracy_std_percent": (
            std_balanced_accuracy * 100
        ),
        "mean_pca_components": mean_components,
        "min_pca_components": min_components,
        "max_pca_components": max_components,
        "runtime_seconds": elapsed,
    }

    return result, all_predictions


# ============================================================
# MAIN
# ============================================================

def main():

    global df
    global subjects_series

    print("=" * 80)
    print("NEURO-SYMPHONY — EEG PCA PREPROCESSING EXPERIMENT")
    print("=" * 80)

    print(f"\nInput file:")
    print(INPUT_FILE)

    print(f"\nOutput directory:")
    print(OUTPUT_DIR)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    eeg_columns, subjects = validate_dataset(
        df
    )

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    X = df[eeg_columns].copy()

    y = df["label"].copy()

    subjects_series = df["subject_id"].copy()

    print("\n" + "=" * 80)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 80)

    print(f"Samples: {len(df)}")
    print(f"EEG features: {len(eeg_columns)}")
    print(f"Subjects: {len(subjects)}")
    print(f"Classes: {df['label'].nunique()}")

    print("\nPCA variance levels:")

    for variance in PCA_VARIANCES:
        print(
            f"  - {int(variance * 100)}%"
        )

    print("\nModels:")

    for model_name in MODELS:
        print(
            f"  - {model_name}"
        )

    print(
        "\nValidation: "
        "12-fold Leave-One-Subject-Out"
    )

    print(
        "\nPipeline per fold:"
    )

    print(
        "  Training subjects"
        " → StandardScaler"
        " → PCA"
        " → Classifier"
    )

    print(
        "\nIMPORTANT: "
        "Scaler and PCA are fitted only on training subjects."
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []
    predictions = []

    total_start_time = time.time()

    # --------------------------------------------------------
    # Run all experiments
    # --------------------------------------------------------

    for pca_variance in PCA_VARIANCES:

        for model_name, model in MODELS.items():

            result, experiment_predictions = (
                run_pca_experiment(
                    X=X,
                    y=y,
                    subjects=subjects,
                    eeg_columns=eeg_columns,
                    pca_variance=pca_variance,
                    model_name=model_name,
                    model=model,
                )
            )

            results.append(
                result
            )

            predictions.extend(
                experiment_predictions
            )

    total_elapsed = (
        time.time() - total_start_time
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    predictions_df = pd.DataFrame(
        predictions
    )

    # Sort best accuracy first
    results_df = results_df.sort_values(
        by="accuracy",
        ascending=False,
    ).reset_index(
        drop=True
    )

    results_df.to_csv(
        RESULTS_FILE,
        index=False,
    )

    predictions_df.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n\n" + "=" * 80)
    print("PCA EXPERIMENT COMPLETE")
    print("=" * 80)

    print(
        f"\nTotal runtime: "
        f"{total_elapsed / 60:.2f} minutes"
    )

    print(
        "\nResults saved to:"
    )

    print(
        RESULTS_FILE
    )

    print(
        PREDICTIONS_FILE
    )

    # --------------------------------------------------------
    # Display summary
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("RESULT SUMMARY")
    print("=" * 80)

    display_columns = [
        "pca_variance_percent",
        "model",
        "accuracy_percent",
        "accuracy_std_percent",
        "macro_f1_percent",
        "balanced_accuracy_percent",
        "mean_pca_components",
    ]

    print(
        results_df[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}"
        )
    )

    # --------------------------------------------------------
    # Best result
    # --------------------------------------------------------

    best_result = results_df.iloc[0]

    print("\n" + "=" * 80)
    print("BEST PCA RESULT")
    print("=" * 80)

    print(
        f"PCA variance: "
        f"{best_result['pca_variance_percent']:.0f}%"
    )

    print(
        f"Model: "
        f"{best_result['model']}"
    )

    print(
        f"Accuracy: "
        f"{best_result['accuracy_percent']:.3f}% "
        f"± "
        f"{best_result['accuracy_std_percent']:.3f}%"
    )

    print(
        f"Macro F1: "
        f"{best_result['macro_f1_percent']:.3f}%"
    )

    print(
        f"Balanced Accuracy: "
        f"{best_result['balanced_accuracy_percent']:.3f}%"
    )

    print(
        f"Mean PCA components: "
        f"{best_result['mean_pca_components']:.1f}"
    )

    # --------------------------------------------------------
    # Compare against current baselines
    # --------------------------------------------------------

    RAW_BASELINE = 12.905
    STANDARD_SCALER = 12.905
    ROBUST_SCALER = 13.044
    TRIAL_NORMALIZATION = 12.824
    CHANCE = 12.50

    best_accuracy = (
        best_result["accuracy_percent"]
    )

    print("\n" + "=" * 80)
    print("COMPARISON WITH PREVIOUS EXPERIMENTS")
    print("=" * 80)

    print(
        f"8-class chance:        {CHANCE:.3f}%"
    )

    print(
        f"Raw EEG baseline:      {RAW_BASELINE:.3f}%"
    )

    print(
        f"StandardScaler:        {STANDARD_SCALER:.3f}%"
    )

    print(
        f"RobustScaler:          {ROBUST_SCALER:.3f}%"
    )

    print(
        f"Trial normalization:   {TRIAL_NORMALIZATION:.3f}%"
    )

    print(
        f"Best PCA result:       {best_accuracy:.3f}%"
    )

    print(
        f"\nPCA vs RobustScaler:   "
        f"{best_accuracy - ROBUST_SCALER:+.3f} pp"
    )

    print(
        f"PCA vs chance:         "
        f"{best_accuracy - CHANCE:+.3f} pp"
    )

    # --------------------------------------------------------
    # Target status
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("25–30% TARGET STATUS")
    print("=" * 80)

    if best_accuracy >= 25:
        print(
            "✓ The 25% minimum target has been reached."
        )

    else:
        print(
            "✗ The 25% target has not yet been reached."
        )

        print(
            "\nThis is an experimental result, not a failure."
        )

        print(
            "If PCA remains near chance, the next legitimate "
            "step is feature selection / other leakage-safe "
            "preprocessing, followed by returning to the "
            "original BDF for temporal/frequency EEG preprocessing."
        )

    print("\nDone.")


if __name__ == "__main__":
    main()