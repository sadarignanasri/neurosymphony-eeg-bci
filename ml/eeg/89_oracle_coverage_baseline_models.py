
from pathlib import Path
from itertools import combinations
import warnings

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score


# ============================================================
# NEUROSYMPHONY STEP 89
# SAMPLE-WISE CORRECTNESS AND ORACLE COVERAGE
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

FEATURE_ROOT = ROOT / "data" / "features"
SPLIT_ROOT = ROOT / "data" / "splits"

OUTPUT_DIR = (
    ROOT / "results" / "eeg" / "oracle_coverage" / "step89"
)

RANDOM_STATE = 42
EXPECTED_SUBJECTS = 12
EXPECTED_TRIALS = 720
EXPECTED_TEST_SAMPLES_PER_SUBJECT = 108
EXPECTED_CLASSES = 8


def get_paths(subject_number):
    sid = f"sub-{subject_number:02d}"

    if subject_number == 1:
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

    return (
        sid,
        feature_dir / f"{sid}_band_power.npy",
        metadata_file,
        split_dir / "train_indices.npy",
        split_dir / "test_indices.npy",
    )


def create_models():
    # All models use the same subject-specific train/test split.
    # Scalers are fitted on training samples only.
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
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "knn": KNeighborsClassifier(
            n_neighbors=5,
            weights="uniform",
        ),
        "lda": LinearDiscriminantAnalysis(),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=100,
            random_state=RANDOM_STATE,
        ),
    }


def load_subject(subject_number):
    sid, feature_file, metadata_file, train_file, test_file = (
        get_paths(subject_number)
    )

    required_files = [
        feature_file,
        metadata_file,
        train_file,
        test_file,
    ]

    for file_path in required_files:
        if not file_path.is_file():
            raise FileNotFoundError(
                f"{sid}: required file not found: {file_path}"
            )

    features = np.load(feature_file)
    metadata = pd.read_csv(metadata_file)

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
        raise ValueError(f"{sid}: features contain NaN or infinity")

    label_column = "label" if subject_number == 1 else "label_id"

    if label_column not in metadata.columns:
        raise ValueError(
            f"{sid}: missing label column {label_column}"
        )

    if len(metadata) != len(X):
        raise ValueError(
            f"{sid}: feature and metadata row counts differ"
        )

    y = pd.to_numeric(
        metadata[label_column], errors="raise"
    ).to_numpy()

    if pd.isna(y).any():
        raise ValueError(f"{sid}: missing labels detected")

    classes = np.sort(np.unique(y))

    if len(classes) != EXPECTED_CLASSES:
        raise ValueError(
            f"{sid}: expected {EXPECTED_CLASSES} classes, "
            f"found {len(classes)}"
        )

    train_idx = np.load(train_file).astype(int)
    test_idx = np.load(test_file).astype(int)

    if len(train_idx) != 504:
        raise ValueError(f"{sid}: expected 504 training samples")

    if len(test_idx) != EXPECTED_TEST_SAMPLES_PER_SUBJECT:
        raise ValueError(f"{sid}: expected 108 test samples")

    if len(np.unique(train_idx)) != len(train_idx):
        raise ValueError(f"{sid}: duplicate training indices")

    if len(np.unique(test_idx)) != len(test_idx):
        raise ValueError(f"{sid}: duplicate test indices")

    if np.intersect1d(train_idx, test_idx).size:
        raise ValueError(f"{sid}: train/test split leakage detected")

    if (
        np.any(train_idx < 0)
        or np.any(train_idx >= len(X))
        or np.any(test_idx < 0)
        or np.any(test_idx >= len(X))
    ):
        raise ValueError(f"{sid}: invalid split indices")

    return sid, X, y, train_idx, test_idx, classes


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prediction_rows = []
    correctness_rows = []

    model_names = list(create_models().keys())

    print("=" * 72)
    print("NEUROSYMPHONY — STEP 89")
    print("SAMPLE-WISE CORRECTNESS AND ORACLE COVERAGE")
    print("=" * 72)

    for subject_number in range(1, EXPECTED_SUBJECTS + 1):
        sid, X, y, train_idx, test_idx, classes = load_subject(
            subject_number
        )

        X_train = X[train_idx]
        y_train = y[train_idx]

        X_test = X[test_idx]
        y_test = y[test_idx]

        print(
            f"\n{sid}: training={len(train_idx)}, "
            f"test={len(test_idx)}, features={X.shape[1]}"
        )

        subject_predictions = {
            "sample_id": [
                f"{sid}_trial_{index + 1:04d}"
                for index in test_idx
            ],
            "subject": sid,
            "trial_index": test_idx + 1,
            "true_label": y_test,
        }

        subject_correctness = {
            "sample_id": subject_predictions["sample_id"],
            "subject": sid,
            "trial_index": test_idx + 1,
            "true_label": y_test,
        }

        for model_name, classifier in create_models().items():
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", classifier),
            ])

            print(f"  Training {model_name}...")

            with warnings.catch_warnings():
                warnings.simplefilter("default")
                pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            correct = (y_pred == y_test).astype(int)

            subject_predictions[f"{model_name}_prediction"] = y_pred
            subject_correctness[model_name] = correct

            prediction_rows.append(
                pd.DataFrame({
                    "sample_id": subject_predictions["sample_id"],
                    "subject": sid,
                    "trial_index": test_idx + 1,
                    "true_label": y_test,
                    "model": model_name,
                    "predicted_label": y_pred,
                    "correct": correct,
                })
            )

            accuracy = accuracy_score(y_test, y_pred) * 100

            print(
                f"    Test accuracy: {accuracy:.2f}% "
                f"({correct.sum()}/{len(correct)})"
            )

        correctness_df = pd.DataFrame(subject_correctness)

        model_correctness_columns = model_names

        # A sample contributes only once, even if several
        # classifiers predict it correctly.
        correctness_df["models_correct_count"] = (
            correctness_df[model_correctness_columns].sum(axis=1)
        )

        correctness_df["any_model_correct"] = (
            correctness_df["models_correct_count"] > 0
        ).astype(int)

        correctness_rows.append(correctness_df)

    # --------------------------------------------------------
    # 1. SAVE SAMPLE-WISE PREDICTIONS
    # --------------------------------------------------------

    predictions = pd.concat(prediction_rows, ignore_index=True)

    predictions.to_csv(
        OUTPUT_DIR / "sample_wise_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 2. SAVE THE SAMPLE-BY-MODEL CORRECTNESS MATRIX
    # --------------------------------------------------------

    correctness = pd.concat(correctness_rows, ignore_index=True)

    correctness.to_csv(
        OUTPUT_DIR / "correctness_matrix.csv",
        index=False,
    )

    model_columns = model_names

    # This is the unique-sample count. Never sum the individual
    # model correct counts to calculate oracle coverage.
    total_samples = len(correctness)
    unique_covered = int(correctness["any_model_correct"].sum())
    missed_by_all = total_samples - unique_covered

    print("\n" + "=" * 72)
    print("OVERALL SAMPLE-WISE RESULTS")
    print("=" * 72)
    print(f"Total held-out test samples: {total_samples}")
    print(f"Correct for at least one model: {unique_covered}")
    print(f"Missed by every model: {missed_by_all}")
    print(
        f"All-model oracle coverage: "
        f"{100 * unique_covered / total_samples:.2f}%"
    )

    # --------------------------------------------------------
    # 3. INDIVIDUAL MODEL SUMMARY
    # --------------------------------------------------------

    individual_rows = []

    for model_name in model_columns:
        n_correct = int(correctness[model_name].sum())

        individual_rows.append({
            "model": model_name,
            "test_samples": total_samples,
            "correct_samples": n_correct,
            "incorrect_samples": total_samples - n_correct,
            "accuracy_percent": 100 * n_correct / total_samples,
        })

    individual_df = pd.DataFrame(individual_rows).sort_values(
        "accuracy_percent", ascending=False
    )

    individual_df.to_csv(
        OUTPUT_DIR / "individual_model_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 4. SELECTED PAIRS, TRIPLES, AND LARGER COMBINATIONS
    # --------------------------------------------------------

    requested_combinations = [
        ("logistic_regression", "random_forest"),
        ("logistic_regression", "linear_svm"),
        ("logistic_regression", "extra_trees"),
        ("random_forest", "extra_trees"),
        (
            "logistic_regression",
            "random_forest",
            "extra_trees",
        ),
        (
            "logistic_regression",
            "linear_svm",
            "random_forest",
        ),
        (
            "logistic_regression",
            "random_forest",
            "hist_gradient_boosting",
        ),
        tuple(model_columns),
    ]

    coverage_rows = []

    for selected_models in requested_combinations:
        correct_any = (
            correctness[list(selected_models)].max(axis=1).astype(bool)
        )

        covered_count = int(correct_any.sum())

        lr_correct = correctness["logistic_regression"].astype(bool)

        additional_beyond_lr = int(
            (correct_any & ~lr_correct).sum()
        )

        coverage_rows.append({
            "models": " + ".join(selected_models),
            "number_of_models": len(selected_models),
            "test_samples": total_samples,
            "unique_samples_correct": covered_count,
            "samples_missed_by_all_selected": (
                total_samples - covered_count
            ),
            "oracle_coverage_percent": (
                100 * covered_count / total_samples
            ),
            "additional_samples_beyond_lr": additional_beyond_lr,
        })

    coverage_df = pd.DataFrame(coverage_rows).sort_values(
        "oracle_coverage_percent", ascending=False
    )

    coverage_df.to_csv(
        OUTPUT_DIR / "oracle_coverage_combinations.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 5. MODEL-PAIR OVERLAP
    # Number of samples both models classify correctly.
    # --------------------------------------------------------

    overlap = pd.DataFrame(
        index=model_columns,
        columns=model_columns,
        dtype=int,
    )

    for model_a in model_columns:
        for model_b in model_columns:
            overlap.loc[model_a, model_b] = int(
                (
                    correctness[model_a].astype(bool)
                    & correctness[model_b].astype(bool)
                ).sum()
            )

    overlap = overlap.astype(int)

    overlap.to_csv(
        OUTPUT_DIR / "model_pair_overlap.csv"
    )

    # --------------------------------------------------------
    # 6. SAMPLES MISSED BY EVERY MODEL
    # --------------------------------------------------------

    missed = correctness[
        correctness["any_model_correct"] == 0
    ].copy()

    missed.to_csv(
        OUTPUT_DIR / "samples_missed_by_all.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 7. PER-CLASS ORACLE COVERAGE
    # --------------------------------------------------------

    per_class_rows = []

    for class_label, group in correctness.groupby("true_label"):
        covered = int(group["any_model_correct"].sum())

        per_class_rows.append({
            "true_label": class_label,
            "test_samples": len(group),
            "samples_correct_for_at_least_one_model": covered,
            "samples_missed_by_all": len(group) - covered,
            "oracle_coverage_percent": 100 * covered / len(group),
        })

    pd.DataFrame(per_class_rows).to_csv(
        OUTPUT_DIR / "per_class_oracle_coverage.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 8. FINAL REPORT
    # --------------------------------------------------------

    print("\nINDIVIDUAL MODEL ACCURACY")
    print(individual_df.round(2).to_string(index=False))

    print("\nORACLE COVERAGE BY SELECTED COMBINATION")
    print(
        coverage_df[
            [
                "models",
                "unique_samples_correct",
                "samples_missed_by_all_selected",
                "oracle_coverage_percent",
                "additional_samples_beyond_lr",
            ]
        ].round(2).to_string(index=False)
    )

    print("\nFiles saved to:")
    print(OUTPUT_DIR.resolve())

    print("\nSTEP 89 COMPLETED.")


if __name__ == "__main__":
    main()
