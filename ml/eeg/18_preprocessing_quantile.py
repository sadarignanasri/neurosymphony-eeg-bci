
"""
NEURO-SYMPHONY
Experiment 18 — Quantile Transformation

Current representation:
    1 row = 1 trial
    64 EEG channel values

Important:
    The 64 values are channel measurements, not time samples.
    Therefore this experiment does not perform FFT/Wavelet.

Pipeline:

    EEG CSV
        ↓
    Strict LOSO
        ↓
    QuantileTransformer
        ↓
    StandardScaler
        ↓
    Classical classifiers
        ↓
    8-class evaluation
"""

import os
import time
import random
import warnings

import numpy as np
import pandas as pd

from sklearn.preprocessing import (
    QuantileTransformer,
    StandardScaler
)

from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis
)

from sklearn.svm import LinearSVC

from sklearn.neighbors import (
    KNeighborsClassifier
)

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

CSV_PATH = (
    "data/processed/"
    "ds004514_eeg_approved.csv"
)

RESULTS_DIR = (
    "results/eeg/preprocessing"
)

RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    "quantile_transform_results.csv"
)

PREDICTIONS_FILE = os.path.join(
    RESULTS_DIR,
    "quantile_transform_predictions.csv"
)

CHANCE_LEVEL = 12.5

N_QUANTILES = 1000


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)

np.random.seed(SEED)


# ============================================================
# MODELS
# ============================================================

def create_models():

    models = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=3000,
                C=1.0
            ),

        "Linear SVM":
            LinearSVC(
                C=1.0,
                max_iter=5000
            ),

        "LDA":
            LinearDiscriminantAnalysis(
                solver="lsqr",
                shrinkage="auto"
            ),

        "KNN":
            KNeighborsClassifier(
                n_neighbors=5
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=300,
                random_state=SEED,
                n_jobs=-1
            ),

        "Extra Trees":
            ExtraTreesClassifier(
                n_estimators=300,
                random_state=SEED,
                n_jobs=-1
            )
    }

    return models


# ============================================================
# DATASET VALIDATION
# ============================================================

def validate_dataset(df):

    required_columns = [
        "subject_id",
        "trial_id",
        "label"
    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Missing required column: {column}"
            )

    eeg_columns = [
        column
        for column in df.columns
        if column.startswith("A")
        or column.startswith("B")
    ]

    if len(eeg_columns) != 64:

        raise ValueError(
            f"Expected 64 EEG channels, "
            f"found {len(eeg_columns)}"
        )

    if df[eeg_columns].isna().any().any():

        raise ValueError(
            "Missing EEG values detected."
        )

    if not np.isfinite(
        df[eeg_columns].values
    ).all():

        raise ValueError(
            "Infinite EEG values detected."
        )

    duplicate_count = df.duplicated(
        subset=[
            "subject_id",
            "trial_id"
        ]
    ).sum()

    if duplicate_count > 0:

        raise ValueError(
            f"Duplicate subject/trial rows: "
            f"{duplicate_count}"
        )

    return eeg_columns


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)

    print(
        "NEURO-SYMPHONY — "
        "QUANTILE TRANSFORMATION"
    )

    print("=" * 80)

    start_time = time.time()

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = pd.read_csv(
        CSV_PATH
    )

    print(
        f"Dataset shape: {df.shape}"
    )

    eeg_columns = validate_dataset(
        df
    )

    subjects = sorted(
        df["subject_id"].unique()
    )

    labels = sorted(
        df["label"].unique()
    )

    print(
        f"EEG channels: "
        f"{len(eeg_columns)}"
    )

    print(
        f"Subjects: "
        f"{len(subjects)}"
    )

    print(
        f"Classes: "
        f"{len(labels)}"
    )

    print(
        f"Quantiles: "
        f"{N_QUANTILES}"
    )

    print(
        "Output distribution: normal"
    )

    print(
        "✓ DATASET VALIDATION PASSED"
    )

    # --------------------------------------------------------
    # LABEL ENCODING
    # --------------------------------------------------------

    label_map = {
        label: index
        for index, label
        in enumerate(labels)
    }

    df["_label_index"] = (
        df["label"]
        .map(label_map)
        .astype(int)
    )

    all_results = []

    all_predictions = []

    # ========================================================
    # LOSO
    # ========================================================

    for fold, test_subject in enumerate(
        subjects,
        start=1
    ):

        print()

        print("=" * 80)

        print(
            f"FOLD {fold}/{len(subjects)}"
        )

        print(
            f"Test subject: "
            f"{test_subject}"
        )

        print("=" * 80)

        # ----------------------------------------------------
        # SUBJECT SPLIT
        # ----------------------------------------------------

        remaining_subjects = [
            subject
            for subject in subjects
            if subject != test_subject
        ]

        # ----------------------------------------------------
        # INNER VALIDATION SUBJECT
        # ----------------------------------------------------

        validation_subject = (
            remaining_subjects[-1]
        )

        train_subjects = [
            subject
            for subject in remaining_subjects
            if subject != validation_subject
        ]

        train_df = df[
            df["subject_id"].isin(
                train_subjects
            )
        ]

        validation_df = df[
            df["subject_id"]
            == validation_subject
        ]

        test_df = df[
            df["subject_id"]
            == test_subject
        ]

        print(
            f"Training samples: "
            f"{len(train_df)}"
        )

        print(
            f"Validation samples: "
            f"{len(validation_df)}"
        )

        print(
            f"Test samples: "
            f"{len(test_df)}"
        )

        print(
            f"Validation subject: "
            f"{validation_subject}"
        )

        # ----------------------------------------------------
        # EXTRACT FEATURES
        # ----------------------------------------------------

        X_train = (
            train_df[eeg_columns]
            .values
            .astype(np.float64)
        )

        X_validation = (
            validation_df[eeg_columns]
            .values
            .astype(np.float64)
        )

        X_test = (
            test_df[eeg_columns]
            .values
            .astype(np.float64)
        )

        y_train = (
            train_df["_label_index"]
            .values
        )

        y_validation = (
            validation_df["_label_index"]
            .values
        )

        y_test = (
            test_df["_label_index"]
            .values
        )

        # ----------------------------------------------------
        # QUANTILE TRANSFORMATION
        #
        # FIT ONLY ON TRAINING SUBJECTS
        # ----------------------------------------------------

        quantile_transformer = (
            QuantileTransformer(
                n_quantiles=min(
                    N_QUANTILES,
                    len(X_train)
                ),
                output_distribution="normal",
                random_state=SEED
            )
        )

        X_train = (
            quantile_transformer
            .fit_transform(X_train)
        )

        X_validation = (
            quantile_transformer
            .transform(X_validation)
        )

        X_test = (
            quantile_transformer
            .transform(X_test)
        )

        # ----------------------------------------------------
        # STANDARDIZATION
        #
        # This is performed after the quantile transformation.
        # Again, fit only on training subjects.
        # ----------------------------------------------------

        scaler = StandardScaler()

        X_train = scaler.fit_transform(
            X_train
        )

        X_validation = scaler.transform(
            X_validation
        )

        X_test = scaler.transform(
            X_test
        )

        # ----------------------------------------------------
        # MODELS
        # ----------------------------------------------------

        models = create_models()

        # ----------------------------------------------------
        # TRAIN / TEST
        # ----------------------------------------------------

        for model_name, model in (
            models.items()
        ):

            model.fit(
                X_train,
                y_train
            )

            predictions = model.predict(
                X_test
            )

            accuracy = (
                accuracy_score(
                    y_test,
                    predictions
                ) * 100
            )

            macro_f1 = (
                f1_score(
                    y_test,
                    predictions,
                    average="macro",
                    zero_division=0
                ) * 100
            )

            balanced_accuracy = (
                balanced_accuracy_score(
                    y_test,
                    predictions
                ) * 100
            )

            all_results.append({

                "fold":
                    fold,

                "test_subject":
                    test_subject,

                "validation_subject":
                    validation_subject,

                "model":
                    model_name,

                "accuracy":
                    accuracy,

                "macro_f1":
                    macro_f1,

                "balanced_accuracy":
                    balanced_accuracy
            })

            # ------------------------------------------------
            # SAVE PREDICTIONS
            # ------------------------------------------------

            trial_ids = (
                test_df[
                    "trial_id"
                ].values
            )

            label_names = (
                test_df[
                    "label"
                ].values
            )

            for i in range(
                len(predictions)
            ):

                all_predictions.append({

                    "fold":
                        fold,

                    "subject_id":
                        test_subject,

                    "trial_id":
                        trial_ids[i],

                    "model":
                        model_name,

                    "true_label":
                        y_test[i],

                    "predicted_label":
                        predictions[i],

                    "true_label_name":
                        label_names[i]
                })

    # ========================================================
    # DATAFRAMES
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    predictions_df = pd.DataFrame(
        all_predictions
    )

    # ========================================================
    # SAVE
    # ========================================================

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    results_df.to_csv(
        RESULTS_FILE,
        index=False
    )

    predictions_df.to_csv(
        PREDICTIONS_FILE,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = (
        results_df
        .groupby("model")
        .agg(
            accuracy_mean=(
                "accuracy",
                "mean"
            ),

            accuracy_std=(
                "accuracy",
                "std"
            ),

            macro_f1_mean=(
                "macro_f1",
                "mean"
            ),

            balanced_accuracy_mean=(
                "balanced_accuracy",
                "mean"
            )
        )
        .sort_values(
            "accuracy_mean",
            ascending=False
        )
    )

    print()

    print("=" * 80)

    print(
        "QUANTILE TRANSFORMATION RESULTS"
    )

    print("=" * 80)

    print(summary)

    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    best_model = summary.index[0]

    best_accuracy = (
        summary.iloc[0][
            "accuracy_mean"
        ]
    )

    print()

    print(
        f"Best model: "
        f"{best_model}"
    )

    print(
        f"Best accuracy: "
        f"{best_accuracy:.3f}%"
    )

    print(
        f"Improvement over chance: "
        f"{best_accuracy - CHANCE_LEVEL:+.3f} pp"
    )

    print(
        f"Improvement vs previous "
        f"best 13.044%: "
        f"{best_accuracy - 13.044:+.3f} pp"
    )

    print(
        f"Runtime: "
        f"{(time.time() - start_time) / 60:.2f} minutes"
    )

    print()

    print(
        f"Results: "
        f"{RESULTS_FILE}"
    )

    print(
        f"Predictions: "
        f"{PREDICTIONS_FILE}"
    )

    print()

    print("Done.")


if __name__ == "__main__":

    main()