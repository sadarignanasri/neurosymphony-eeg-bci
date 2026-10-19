
"""
NeuroSymphony — Step 63
Test-set evaluation of five classifiers for all 12 EEG subjects.

Outputs:
  - Test accuracy, balanced accuracy and macro-F1 for 60 models
  - Best classifier for each subject
  - Per-subject confusion matrices and classification reports
  - Aggregate CSV reports

IMPORTANT:
Models are fitted on training data only.
Validation and test data are not used for fitting.
"""

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

FEATURE_ROOT = ROOT / "data" / "features"
SPLIT_ROOT = ROOT / "data" / "splits"
RESULT_ROOT = ROOT / "results" / "eeg" / "model_comparison"

AGGREGATE_DIR = (
    RESULT_ROOT / "all_subjects" / "step63_test_summary"
)

RANDOM_STATE = 42
EXPECTED_SUBJECTS = 12
EXPECTED_TRIALS = 720
EXPECTED_CLASSES = 8

warnings.filterwarnings(
    "default", category=InconsistentVersionWarning
)


# ============================================================
# PATHS
# ============================================================

def get_subject_paths(number):
    sid = f"sub-{number:02d}"

    if number == 1:
        feature_dir = FEATURE_ROOT / "temporal_spectral_sub01"
        split_dir = SPLIT_ROOT / "sub01"
        metadata_file = (
            feature_dir / f"{sid}_spectral_metadata.csv"
        )
    else:
        feature_dir = FEATURE_ROOT / f"temporal_spectral_{sid}"
        split_dir = SPLIT_ROOT / sid
        metadata_file = (
            FEATURE_ROOT
            / f"temporal_eeg_{sid}"
            / f"{sid}_temporal_eeg_preprocessed_metadata.csv"
        )

    # Separate output folder to preserve previous Step 63 results.
    output_dir = (
        RESULT_ROOT
        / ("sub01" if number == 1 else sid)
        / "step63_test_comparison_all_subjects"
    )

    return {
        "sid": sid,
        "feature": feature_dir / f"{sid}_band_power.npy",
        "metadata": metadata_file,
        "split": split_dir,
        "output": output_dir,
    }


# ============================================================
# MODELS
# ============================================================

def create_models():
    return {
        "logistic_regression": LogisticRegression(
            solver="lbfgs",
            max_iter=5000,
            random_state=RANDOM_STATE,
        ),

        "linear_svm": LinearSVC(
            C=1.0,
            max_iter=10000,
            random_state=RANDOM_STATE,
        ),

        "rbf_svm": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            class_weight=None,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "knn": KNeighborsClassifier(
            n_neighbors=5,
            weights="uniform",
            metric="minkowski",
            p=2,
        ),
    }


# ============================================================
# LOAD DATA AND VALIDATE SPLITS
# ============================================================

def load_subject_data(number):
    paths = get_subject_paths(number)
    sid = paths["sid"]

    for key in ("feature", "metadata"):
        if not paths[key].is_file():
            raise FileNotFoundError(
                f"{sid}: missing {key} file: {paths[key]}"
            )

    split_dir = paths["split"]
    split_files = {
        "train": split_dir / "train_indices.npy",
        "validation": split_dir / "val_indices.npy",
        "test": split_dir / "test_indices.npy",
    }

    for name, file_path in split_files.items():
        if not file_path.is_file():
            raise FileNotFoundError(
                f"{sid}: missing {name} split: {file_path}"
            )

    features = np.load(paths["feature"])
    metadata = pd.read_csv(paths["metadata"])

    if features.ndim == 3:
        X = features.reshape(features.shape[0], -1)
    elif features.ndim == 2:
        X = features
    else:
        raise ValueError(
            f"{sid}: unexpected feature shape {features.shape}"
        )

    if X.shape[0] != EXPECTED_TRIALS:
        raise ValueError(
            f"{sid}: expected {EXPECTED_TRIALS} trials, "
            f"found {X.shape[0]}"
        )

    if not np.isfinite(X).all():
        raise ValueError(
            f"{sid}: features contain NaN or infinity"
        )

    label_column = "label" if number == 1 else "label_id"

    if label_column not in metadata.columns:
        raise ValueError(
            f"{sid}: missing '{label_column}' in metadata. "
            f"Available columns: {metadata.columns.tolist()}"
        )

    if len(metadata) != len(X):
        raise ValueError(
            f"{sid}: feature/metadata row counts do not match"
        )

    y = pd.to_numeric(
        metadata[label_column], errors="raise"
    ).to_numpy()

    if pd.isna(y).any():
        raise ValueError(f"{sid}: labels contain missing values")

    classes = np.sort(np.unique(y))

    if len(classes) != EXPECTED_CLASSES:
        raise ValueError(
            f"{sid}: expected {EXPECTED_CLASSES} classes, "
            f"found {len(classes)}: {classes.tolist()}"
        )

    train_idx = np.load(split_files["train"]).astype(int)
    val_idx = np.load(split_files["validation"]).astype(int)
    test_idx = np.load(split_files["test"]).astype(int)

    # Validate the existing split files.
    if (
        len(train_idx) != 504
        or len(val_idx) != 108
        or len(test_idx) != 108
    ):
        raise ValueError(
            f"{sid}: unexpected split sizes: "
            f"train={len(train_idx)}, "
            f"validation={len(val_idx)}, "
            f"test={len(test_idx)}"
        )

    all_idx = np.concatenate([train_idx, val_idx, test_idx])

    if np.any(all_idx < 0) or np.any(all_idx >= len(X)):
        raise ValueError(f"{sid}: split contains invalid indices")

    if len(np.unique(all_idx)) != len(all_idx):
        raise ValueError(
            f"{sid}: split indices overlap or contain duplicates"
        )

    if not np.array_equal(
        np.sort(all_idx), np.arange(len(X))
    ):
        raise ValueError(
            f"{sid}: splits do not cover all trials exactly once"
        )

    # Return only the data required for this test evaluation.
    return paths, X[train_idx], y[train_idx], X[test_idx], y[test_idx], classes


# ============================================================
# EVALUATE ONE SUBJECT
# ============================================================

def evaluate_subject(number):
    (
        paths,
        X_train,
        y_train,
        X_test,
        y_test,
        classes,
    ) = load_subject_data(number)

    sid = paths["sid"]
    output_dir = paths["output"]
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 76)
    print(f"STEP 63 — TEST EVALUATION: {sid}")
    print("=" * 76)
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples:     {len(X_test)}")
    print(f"Features:         {X_train.shape[1]}")
    print(f"Classes:          {len(classes)}")

    subject_results = []
    subject_reports = []

    for model_name, classifier in create_models().items():
        print(f"\nTraining {model_name}...")

        # Fit the scaler and classifier using training data only.
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pipeline.fit(X_train, y_train)

        for warning in caught:
            print(f"Warning ({model_name}): {warning.message}")

        # The test set is used only for final prediction/evaluation.
        y_pred = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(
            y_test,
            y_pred,
            labels=classes,
            average="macro",
            zero_division=0,
        )

        result = {
            "subject": sid,
            "classifier": model_name,
            "test_samples": len(y_test),
            "test_accuracy": accuracy,
            "test_accuracy_percent": accuracy * 100,
            "test_balanced_accuracy": balanced_acc,
            "test_balanced_accuracy_percent": balanced_acc * 100,
            "test_macro_f1": macro_f1,
        }

        subject_results.append(result)

        print(
            f"{model_name}: "
            f"Test accuracy = {accuracy * 100:.2f}% | "
            f"Balanced accuracy = {balanced_acc * 100:.2f}% | "
            f"Macro-F1 = {macro_f1:.4f}"
        )

        # Save the per-class report.
        report = classification_report(
            y_test,
            y_pred,
            labels=classes,
            output_dict=True,
            zero_division=0,
        )

        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(
            output_dir / f"{model_name}_classification_report.csv"
        )

        subject_reports.append({
            "subject": sid,
            "classifier": model_name,
            "report": report_df,
        })

        # Save the confusion matrix.
        cm = confusion_matrix(
            y_test, y_pred, labels=classes
        )

        pd.DataFrame(
            cm,
            index=[f"actual_{c}" for c in classes],
            columns=[f"predicted_{c}" for c in classes],
        ).to_csv(output_dir / f"{model_name}_confusion_matrix.csv")

        fig, ax = plt.subplots(figsize=(8, 7))
        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=classes,
        )
        display.plot(
            ax=ax, cmap="Blues", colorbar=False, values_format="d"
        )
        ax.set_title(f"{sid} — {model_name} — Test Set")
        fig.tight_layout()
        fig.savefig(
            output_dir / f"{model_name}_confusion_matrix.png",
            dpi=150,
        )
        plt.close(fig)

    # Save this subject's five classifier results.
    subject_df = pd.DataFrame(subject_results).sort_values(
        "test_accuracy", ascending=False
    )

    subject_df.to_csv(
        output_dir / "step63_test_accuracy.csv", index=False
    )

    best = subject_df.iloc[0]

    print(f"\nBest classifier for {sid}: {best['classifier']}")
    print(f"Best test accuracy: {best['test_accuracy_percent']:.2f}%")
    print(f"Subject results saved to: {output_dir}")

    return subject_results


# ============================================================
# MAIN — ALL 12 SUBJECTS
# ============================================================

def main():
    print("=" * 76)
    print("NEUROSYMPHONY — STEP 63: ALL-SUBJECT TEST ACCURACY")
    print("=" * 76)
    print("Five classifiers × 12 subjects = 60 evaluations")
    print("Training data is used for fitting; test data for evaluation.")

    AGGREGATE_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    failures = []

    for number in range(1, EXPECTED_SUBJECTS + 1):
        try:
            results = evaluate_subject(number)
            all_results.extend(results)
        except Exception as exc:
            sid = f"sub-{number:02d}"
            failures.append({
                "subject": sid,
                "error": str(exc),
            })
            print(f"\nERROR — {sid}: {exc}")

    if all_results:
        results_df = pd.DataFrame(all_results)

        results_path = (
            AGGREGATE_DIR / "step63_all_subjects_test_accuracy.csv"
        )

        results_df.to_csv(results_path, index=False)

        # Subject × classifier accuracy table.
        accuracy_table = results_df.pivot(
            index="subject",
            columns="classifier",
            values="test_accuracy_percent",
        )

        accuracy_table.to_csv(
            AGGREGATE_DIR / "step63_test_accuracy_by_subject.csv"
        )

        # Best-performing classifier per subject.
        best_indices = results_df.groupby(
            "subject"
        )["test_accuracy"].idxmax()

        best_df = results_df.loc[best_indices].sort_values("subject")

        best_df.to_csv(
            AGGREGATE_DIR / "step63_best_classifier_per_subject.csv",
            index=False,
        )

        # Mean accuracy across the 12 subjects for each classifier.
        average_df = (
            results_df.groupby("classifier", as_index=False)
            .agg(
                mean_test_accuracy_percent=(
                    "test_accuracy_percent", "mean"
                ),
                mean_balanced_accuracy_percent=(
                    "test_balanced_accuracy_percent", "mean"
                ),
                mean_macro_f1=("test_macro_f1", "mean"),
                subjects_evaluated=("subject", "nunique"),
            )
            .sort_values(
                "mean_test_accuracy_percent", ascending=False
            )
        )

        average_df.to_csv(
            AGGREGATE_DIR / "step63_classifier_average_results.csv",
            index=False,
        )

        print("\n" + "=" * 76)
        print("TEST ACCURACY BY SUBJECT (%)")
        print("=" * 76)
        print(accuracy_table.round(2).to_string())

        print("\nBEST CLASSIFIER PER SUBJECT")
        print(
            best_df[
                ["subject", "classifier", "test_accuracy_percent"]
            ].to_string(index=False)
        )

        print("\nMEAN RESULTS ACROSS SUBJECTS")
        print(average_df.round(3).to_string(index=False))

        print(f"\nAll-classifier results: {results_path}")
        print(f"Aggregate reports: {AGGREGATE_DIR}")

    if failures:
        failure_path = AGGREGATE_DIR / "step63_failures.csv"
        pd.DataFrame(failures).to_csv(failure_path, index=False)

        print("\nSome subjects failed. Check:")
        print(failure_path)

        for failure in failures:
            print(f"{failure['subject']}: {failure['error']}")

        raise RuntimeError(
            f"Step 63 completed with {len(failures)} subject failure(s)."
        )

    print("\n" + "=" * 76)
    print("STEP 63 COMPLETED FOR ALL 12 SUBJECTS")
    print("All 60 classifier evaluations completed successfully.")
    print("=" * 76)


if __name__ == "__main__":
    main()
