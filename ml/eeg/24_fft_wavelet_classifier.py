"""
NEURO-SYMPHONY
Experiment 4 — Channel-Domain FFT + Wavelet/DWT + Classifier

Dataset:
    data/processed/ds004514_eeg_approved.csv

IMPORTANT METHODOLOGICAL NOTE
-----------------------------
The current CSV contains:

    one row = one trial
    64 columns = EEG channels

It does NOT contain temporal EEG samples.

Therefore:

    FFT = applied across the 64-channel dimension
    DWT = applied across the 64-channel dimension

These are CHANNEL-DOMAIN transformations.

They must NOT be interpreted as:

    - temporal EEG FFT
    - temporal EEG DWT
    - delta/theta/alpha/beta/gamma bands
    - true EEG time-frequency analysis

True temporal FFT/DWT requires the original BDF recordings.

Evaluation:
    12-fold subject-independent evaluation

    10 subjects -> TRAIN
    1 subject   -> VALIDATION
    1 subject   -> TEST

Features:
    FFT magnitude + phase
    +
    DWT coefficients

Classifiers:
    Logistic Regression
    Linear SVM
    Extra Trees
"""


# =============================================================================
# IMPORTS
# =============================================================================

from pathlib import Path

import random
import time

import numpy as np
import pandas as pd
import pywt

from sklearn.base import clone

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from sklearn.preprocessing import RobustScaler

from sklearn.svm import LinearSVC


# =============================================================================
# CONFIGURATION
# =============================================================================

SEED = 42

DATA_PATH = Path(
    "data/processed/ds004514_eeg_approved.csv"
)

RESULTS_DIR = Path(
    "results/eeg/fft_wavelet"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

NUM_CLASSES = 8

NUM_CHANNELS = 64

CHANCE_LEVEL = 1.0 / NUM_CLASSES


# =============================================================================
# FFT CONFIGURATION
# =============================================================================

FFT_COMPONENTS = "magnitude_and_phase"


# =============================================================================
# DWT CONFIGURATION
# =============================================================================

WAVELET = "db4"

DWT_LEVEL = 3


# =============================================================================
# CLASSIFIERS
# =============================================================================

CLASSIFIERS = {

    "LogisticRegression": LogisticRegression(
        max_iter=2000,
        C=1.0,
        random_state=SEED,
    ),

    "LinearSVM": LinearSVC(
        C=1.0,
        max_iter=5000,
        random_state=SEED,
    ),

    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=300,
        max_features="sqrt",
        random_state=SEED,
        n_jobs=-1,
    ),
}


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed=SEED):

    random.seed(seed)

    np.random.seed(seed)


# =============================================================================
# DATASET VALIDATION
# =============================================================================

def validate_dataset(df):

    print("=" * 80)
    print("DATASET VALIDATION")
    print("=" * 80)

    eeg_columns = (
        [f"A{i}" for i in range(1, 33)]
        +
        [f"B{i}" for i in range(1, 33)]
    )

    required_columns = (
        [
            "subject_id",
            "trial_id",
            "label_name",
            "label",
        ]
        +
        eeg_columns
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing required columns: "
            f"{missing_columns}"
        )

    if len(eeg_columns) != NUM_CHANNELS:

        raise ValueError(
            f"Expected {NUM_CHANNELS} EEG channels, "
            f"found {len(eeg_columns)}"
        )

    # -------------------------------------------------------------------------
    # Missing values
    # -------------------------------------------------------------------------

    if df[eeg_columns].isna().any().any():

        raise ValueError(
            "Missing values detected in EEG channels."
        )

    # -------------------------------------------------------------------------
    # Infinite values
    # -------------------------------------------------------------------------

    values = df[eeg_columns].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(values).all():

        raise ValueError(
            "Infinite values detected in EEG channels."
        )

    # -------------------------------------------------------------------------
    # Subjects
    # -------------------------------------------------------------------------

    subjects = sorted(
        df["subject_id"].unique()
    )

    if len(subjects) != 12:

        raise ValueError(
            f"Expected 12 subjects, "
            f"found {len(subjects)}"
        )

    # -------------------------------------------------------------------------
    # Labels
    # -------------------------------------------------------------------------

    labels = sorted(
        df["label"].unique()
    )

    expected_labels = list(
        range(1, NUM_CLASSES + 1)
    )

    if labels != expected_labels:

        raise ValueError(
            f"Expected labels {expected_labels}, "
            f"found {labels}"
        )

    # -------------------------------------------------------------------------
    # Subject distribution
    # -------------------------------------------------------------------------

    subject_counts = (
        df.groupby("subject_id")
        .size()
    )

    if not (subject_counts == 720).all():

        raise ValueError(
            "Each subject must contain exactly "
            "720 trials."
        )

    # -------------------------------------------------------------------------
    # Class distribution
    # -------------------------------------------------------------------------

    class_counts = (
        df.groupby("label")
        .size()
    )

    if not (class_counts == 1080).all():

        raise ValueError(
            "Each class must contain exactly "
            "1080 trials."
        )

    # -------------------------------------------------------------------------
    # Subject/class distribution
    # -------------------------------------------------------------------------

    subject_class_counts = (
        df.groupby(
            [
                "subject_id",
                "label",
            ]
        )
        .size()
        .unique()
    )

    if (
        len(subject_class_counts) != 1
        or subject_class_counts[0] != 90
    ):

        raise ValueError(
            "Each subject/class combination "
            "must contain exactly 90 trials."
        )

    # -------------------------------------------------------------------------
    # Duplicate trial check
    # -------------------------------------------------------------------------

    duplicate_trials = (
        df.duplicated(
            subset=[
                "subject_id",
                "trial_id",
            ]
        )
        .sum()
    )

    if duplicate_trials != 0:

        raise ValueError(
            f"Found {duplicate_trials} "
            "duplicate subject/trial pairs."
        )

    print(
        f"Dataset shape: {df.shape}"
    )

    print(
        f"EEG channels: {len(eeg_columns)}"
    )

    print(
        f"Subjects: {len(subjects)}"
    )

    print(
        f"Classes: {len(labels)}"
    )

    print(
        "✓ DATASET VALIDATION PASSED"
    )

    return eeg_columns, subjects


# =============================================================================
# CHANNEL-DOMAIN FFT
# =============================================================================

def channel_domain_fft(X):

    """
    Apply FFT across the 64-channel dimension.

    Input:
        X = [trials, 64]

    rFFT of 64 values produces:

        33 frequency-bin coefficients

    We retain:

        33 magnitude values
        +
        33 phase values

        = 66 features

    IMPORTANT:
        These are channel-domain Fourier coefficients.

        They are NOT temporal EEG frequency bands.
    """

    fft_values = np.fft.rfft(
        X,
        axis=1
    )

    magnitude = np.abs(
        fft_values
    )

    phase = np.angle(
        fft_values
    )

    fft_features = np.concatenate(
        [
            magnitude,
            phase,
        ],
        axis=1
    )

    fft_features = np.nan_to_num(
        fft_features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return fft_features.astype(
        np.float32
    )


# =============================================================================
# CHANNEL-DOMAIN DWT
# =============================================================================

def channel_domain_dwt(X):

    """
    Apply DWT across the 64-channel dimension.

    Wavelet:
        db4

    Level:
        3

    The resulting coefficient vector is flattened.

    IMPORTANT:
        These are channel-domain wavelet coefficients,
        not temporal EEG wavelet coefficients.
    """

    transformed_trials = []

    for trial in X:

        coefficients = pywt.wavedec(
            trial,
            wavelet=WAVELET,
            level=DWT_LEVEL,
            mode="symmetric",
        )

        flattened = np.concatenate(
            [
                coefficient.ravel()
                for coefficient in coefficients
            ]
        )

        transformed_trials.append(
            flattened
        )

    dwt_features = np.asarray(
        transformed_trials,
        dtype=np.float32
    )

    dwt_features = np.nan_to_num(
        dwt_features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return dwt_features


# =============================================================================
# COMBINED FEATURE EXTRACTION
# =============================================================================

def extract_fft_wavelet_features(X):

    """
    Generate:

        FFT features
        +
        DWT features

    and concatenate them.

    Expected:

        FFT = 66
        DWT = 84

        Total = 150 features
    """

    fft_features = channel_domain_fft(
        X
    )

    dwt_features = channel_domain_dwt(
        X
    )

    combined_features = np.concatenate(
        [
            fft_features,
            dwt_features,
        ],
        axis=1
    )

    return (
        fft_features,
        dwt_features,
        combined_features,
    )


# =============================================================================
# PREPARE FOLD
# =============================================================================

def prepare_fold_data(
    df,
    eeg_columns,
    train_subjects,
    validation_subject,
    test_subject,
):

    # -------------------------------------------------------------------------
    # Subject split
    # -------------------------------------------------------------------------

    train_df = df[
        df["subject_id"].isin(
            train_subjects
        )
    ].copy()

    validation_df = df[
        df["subject_id"] == validation_subject
    ].copy()

    test_df = df[
        df["subject_id"] == test_subject
    ].copy()

    # -------------------------------------------------------------------------
    # Raw EEG
    # -------------------------------------------------------------------------

    X_train_raw = train_df[
        eeg_columns
    ].to_numpy(
        dtype=np.float64
    )

    X_validation_raw = validation_df[
        eeg_columns
    ].to_numpy(
        dtype=np.float64
    )

    X_test_raw = test_df[
        eeg_columns
    ].to_numpy(
        dtype=np.float64
    )

    # -------------------------------------------------------------------------
    # Combined FFT + DWT
    # -------------------------------------------------------------------------

    (
        X_train_fft,
        X_train_dwt,
        X_train_combined,
    ) = extract_fft_wavelet_features(
        X_train_raw
    )

    (
        X_validation_fft,
        X_validation_dwt,
        X_validation_combined,
    ) = extract_fft_wavelet_features(
        X_validation_raw
    )

    (
        X_test_fft,
        X_test_dwt,
        X_test_combined,
    ) = extract_fft_wavelet_features(
        X_test_raw
    )

    # -------------------------------------------------------------------------
    # RobustScaler
    #
    # Fit ONLY on training subjects.
    # -------------------------------------------------------------------------

    scaler = RobustScaler()

    X_train = scaler.fit_transform(
        X_train_combined
    ).astype(
        np.float32
    )

    X_validation = scaler.transform(
        X_validation_combined
    ).astype(
        np.float32
    )

    X_test = scaler.transform(
        X_test_combined
    ).astype(
        np.float32
    )

    # -------------------------------------------------------------------------
    # Labels
    #
    # Original:
    #     1..8
    #
    # Model:
    #     0..7
    # -------------------------------------------------------------------------

    y_train = (
        train_df["label"]
        .to_numpy(
            dtype=np.int64
        )
        - 1
    )

    y_validation = (
        validation_df["label"]
        .to_numpy(
            dtype=np.int64
        )
        - 1
    )

    y_test = (
        test_df["label"]
        .to_numpy(
            dtype=np.int64
        )
        - 1
    )

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        X_train_fft,
        X_train_dwt,
    )


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_classifier(
    model,
    X,
    y,
):

    predictions = model.predict(
        X
    )

    accuracy = accuracy_score(
        y,
        predictions
    )

    macro_f1 = f1_score(
        y,
        predictions,
        average="macro",
        zero_division=0,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y,
            predictions
        )
    )

    return (
        accuracy,
        macro_f1,
        balanced_accuracy,
        predictions,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    set_seed()

    print("=" * 80)
    print(
        "NEURO-SYMPHONY — "
        "EXPERIMENT 4: CHANNEL-DOMAIN FFT + WAVELET/DWT"
    )
    print("=" * 80)

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "FFT and DWT are applied across the "
        "64 EEG CHANNELS."
    )

    print(
        "They are NOT temporal EEG transformations."
    )

    print(
        "True temporal FFT/DWT requires the original BDF."
    )

    print()

    # =========================================================================
    # LOAD DATA
    # =========================================================================

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH
    )

    eeg_columns, subjects = (
        validate_dataset(df)
    )

    # =========================================================================
    # FEATURE INFORMATION
    # =========================================================================

    sample_X = df[
        eeg_columns
    ].to_numpy(
        dtype=np.float64
    )[:1]

    (
        sample_fft,
        sample_dwt,
        sample_combined,
    ) = extract_fft_wavelet_features(
        sample_X
    )

    print()
    print(
        "FEATURE INFORMATION"
    )
    print(
        "-" * 80
    )

    print(
        f"Original EEG features: "
        f"{sample_X.shape[1]}"
    )

    print(
        f"FFT features: "
        f"{sample_fft.shape[1]}"
    )

    print(
        f"DWT features: "
        f"{sample_dwt.shape[1]}"
    )

    print(
        f"Combined features: "
        f"{sample_combined.shape[1]}"
    )

    print(
        f"Wavelet: "
        f"{WAVELET}"
    )

    print(
        f"DWT level: "
        f"{DWT_LEVEL}"
    )

    print(
        "FFT layout:"
    )

    print(
        "  Magnitude: 33"
    )

    print(
        "  Phase: 33"
    )

    print(
        "  Total: 66"
    )

    print(
        "DWT layout:"
    )

    print(
        "  cA3: 8"
    )

    print(
        "  cD3: 8"
    )

    print(
        "  cD2: 16"
    )

    print(
        "  cD1: 32"
    )

    print(
        "  Total: 84"
    )

    print(
        f"Total combined: "
        f"{sample_combined.shape[1]}"
    )

    # =========================================================================
    # CLASS DISTRIBUTION
    # =========================================================================

    class_distribution = (
        df.groupby("label")
        .size()
        .reset_index(
            name="trial_count"
        )
    )

    class_distribution.to_csv(
        RESULTS_DIR
        / "class_distribution.csv",
        index=False,
    )

    # =========================================================================
    # STORAGE
    # =========================================================================

    all_fold_results = []

    all_predictions = []

    start_time = time.time()

    # =========================================================================
    # 12-FOLD SUBJECT-INDEPENDENT EVALUATION
    # =========================================================================

    for fold_index, test_subject in enumerate(
        subjects,
        start=1
    ):

        # ---------------------------------------------------------------------
        # Remaining subjects
        # ---------------------------------------------------------------------

        remaining_subjects = [
            subject
            for subject in subjects
            if subject != test_subject
        ]

        # ---------------------------------------------------------------------
        # Deterministic validation subject
        # ---------------------------------------------------------------------

        validation_index = (
            fold_index - 1
        ) % len(remaining_subjects)

        validation_subject = (
            remaining_subjects[
                validation_index
            ]
        )

        # ---------------------------------------------------------------------
        # Training subjects
        # ---------------------------------------------------------------------

        train_subjects = [
            subject
            for subject in remaining_subjects
            if subject != validation_subject
        ]

        print()
        print("=" * 80)
        print(
            f"FOLD {fold_index}/12"
        )
        print("=" * 80)

        print(
            f"Train subjects ({len(train_subjects)}): "
            f"{train_subjects}"
        )

        print(
            f"Validation subject: "
            f"{validation_subject}"
        )

        print(
            f"Test subject: "
            f"{test_subject}"
        )

        # ---------------------------------------------------------------------
        # Prepare data
        # ---------------------------------------------------------------------

        (
            X_train,
            y_train,
            X_validation,
            y_validation,
            X_test,
            y_test,
            X_train_fft,
            X_train_dwt,
        ) = prepare_fold_data(
            df,
            eeg_columns,
            train_subjects,
            validation_subject,
            test_subject,
        )

        print()

        print(
            f"Train trials: "
            f"{len(X_train)}"
        )

        print(
            f"Validation trials: "
            f"{len(X_validation)}"
        )

        print(
            f"Test trials: "
            f"{len(X_test)}"
        )

        print(
            f"FFT features: "
            f"{X_train_fft.shape[1]}"
        )

        print(
            f"DWT features: "
            f"{X_train_dwt.shape[1]}"
        )

        print(
            f"Combined features: "
            f"{X_train.shape[1]}"
        )

        # ---------------------------------------------------------------------
        # Class distribution
        # ---------------------------------------------------------------------

        print()

        print(
            "Class counts:"
        )

        print(
            "Train:",
            np.bincount(
                y_train,
                minlength=NUM_CLASSES
            )
        )

        print(
            "Validation:",
            np.bincount(
                y_validation,
                minlength=NUM_CLASSES
            )
        )

        print(
            "Test:",
            np.bincount(
                y_test,
                minlength=NUM_CLASSES
            )
        )

        # ---------------------------------------------------------------------
        # Classifier loop
        # ---------------------------------------------------------------------

        for (
            classifier_name,
            classifier_template
        ) in CLASSIFIERS.items():

            print()
            print(
                "-" * 80
            )

            print(
                f"CLASSIFIER: "
                f"{classifier_name}"
            )

            # -----------------------------------------------------------------
            # Fresh model
            # -----------------------------------------------------------------

            model = clone(
                classifier_template
            )

            # -----------------------------------------------------------------
            # Train
            # -----------------------------------------------------------------

            model.fit(
                X_train,
                y_train
            )

            # -----------------------------------------------------------------
            # Training
            # -----------------------------------------------------------------

            (
                train_accuracy,
                train_f1,
                train_balanced,
                train_predictions,
            ) = evaluate_classifier(
                model,
                X_train,
                y_train,
            )

            # -----------------------------------------------------------------
            # Validation
            # -----------------------------------------------------------------

            (
                validation_accuracy,
                validation_f1,
                validation_balanced,
                validation_predictions,
            ) = evaluate_classifier(
                model,
                X_validation,
                y_validation,
            )

            # -----------------------------------------------------------------
            # Test
            # -----------------------------------------------------------------

            (
                test_accuracy,
                test_f1,
                test_balanced,
                test_predictions,
            ) = evaluate_classifier(
                model,
                X_test,
                y_test,
            )

            # -----------------------------------------------------------------
            # Print metrics
            # -----------------------------------------------------------------

            print(
                f"Train accuracy: "
                f"{train_accuracy * 100:.3f}%"
            )

            print(
                f"Validation accuracy: "
                f"{validation_accuracy * 100:.3f}%"
            )

            print(
                f"Test accuracy: "
                f"{test_accuracy * 100:.3f}%"
            )

            print(
                f"Macro F1: "
                f"{test_f1 * 100:.3f}%"
            )

            print(
                f"Balanced accuracy: "
                f"{test_balanced * 100:.3f}%"
            )

            # -----------------------------------------------------------------
            # Store results
            # -----------------------------------------------------------------

            all_fold_results.append({

                "fold":
                    fold_index,

                "classifier":
                    classifier_name,

                "train_subject_count":
                    len(train_subjects),

                "validation_subject":
                    validation_subject,

                "test_subject":
                    test_subject,

                "train_trials":
                    len(X_train),

                "validation_trials":
                    len(X_validation),

                "test_trials":
                    len(X_test),

                "fft_features":
                    X_train_fft.shape[1],

                "dwt_features":
                    X_train_dwt.shape[1],

                "combined_features":
                    X_train.shape[1],

                "wavelet":
                    WAVELET,

                "dwt_level":
                    DWT_LEVEL,

                "train_accuracy":
                    train_accuracy,

                "validation_accuracy":
                    validation_accuracy,

                "test_accuracy":
                    test_accuracy,

                "macro_f1":
                    test_f1,

                "balanced_accuracy":
                    test_balanced,
            })

            # -----------------------------------------------------------------
            # Store predictions
            # -----------------------------------------------------------------

            prediction_df = df[
                df["subject_id"] == test_subject
            ][
                [
                    "subject_id",
                    "trial_id",
                    "label_name",
                    "label",
                ]
            ].copy()

            prediction_df[
                "classifier"
            ] = classifier_name

            prediction_df[
                "fold"
            ] = fold_index

            prediction_df[
                "validation_subject"
            ] = validation_subject

            prediction_df[
                "true_label_zero_based"
            ] = y_test

            prediction_df[
                "predicted_label_zero_based"
            ] = test_predictions

            prediction_df[
                "predicted_label"
            ] = test_predictions + 1

            all_predictions.append(
                prediction_df
            )

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================

    results_df = pd.DataFrame(
        all_fold_results
    )

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True
    )

    results_df.to_csv(
        RESULTS_DIR
        / "fft_wavelet_classifier_results.csv",
        index=False,
    )

    predictions_df.to_csv(
        RESULTS_DIR
        / "fft_wavelet_classifier_predictions.csv",
        index=False,
    )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    summary_rows = []

    for classifier_name in CLASSIFIERS.keys():

        classifier_results = results_df[
            results_df["classifier"]
            == classifier_name
        ]

        mean_train_accuracy = (
            classifier_results[
                "train_accuracy"
            ].mean()
        )

        mean_validation_accuracy = (
            classifier_results[
                "validation_accuracy"
            ].mean()
        )

        mean_test_accuracy = (
            classifier_results[
                "test_accuracy"
            ].mean()
        )

        std_test_accuracy = (
            classifier_results[
                "test_accuracy"
            ].std()
        )

        mean_macro_f1 = (
            classifier_results[
                "macro_f1"
            ].mean()
        )

        std_macro_f1 = (
            classifier_results[
                "macro_f1"
            ].std()
        )

        mean_balanced_accuracy = (
            classifier_results[
                "balanced_accuracy"
            ].mean()
        )

        difference_from_chance = (
            mean_test_accuracy
            - CHANCE_LEVEL
        )

        summary_rows.append({

            "experiment":
                "Channel-Domain FFT + Wavelet/DWT + Classifier",

            "classifier":
                classifier_name,

            "folds":
                len(classifier_results),

            "train_subjects_per_fold":
                10,

            "validation_subjects_per_fold":
                1,

            "test_subjects_per_fold":
                1,

            "fft_features":
                int(
                    classifier_results[
                        "fft_features"
                    ].iloc[0]
                ),

            "dwt_features":
                int(
                    classifier_results[
                        "dwt_features"
                    ].iloc[0]
                ),

            "combined_features":
                int(
                    classifier_results[
                        "combined_features"
                    ].iloc[0]
                ),

            "wavelet":
                WAVELET,

            "dwt_level":
                DWT_LEVEL,

            "mean_train_accuracy":
                mean_train_accuracy,

            "mean_validation_accuracy":
                mean_validation_accuracy,

            "mean_test_accuracy":
                mean_test_accuracy,

            "std_test_accuracy":
                std_test_accuracy,

            "mean_macro_f1":
                mean_macro_f1,

            "std_macro_f1":
                std_macro_f1,

            "mean_balanced_accuracy":
                mean_balanced_accuracy,

            "chance_accuracy":
                CHANCE_LEVEL,

            "difference_from_chance":
                difference_from_chance,
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        RESULTS_DIR
        / "fft_wavelet_classifier_summary.csv",
        index=False,
    )

    # =========================================================================
    # CONFUSION MATRICES
    # =========================================================================

    for classifier_name in CLASSIFIERS.keys():

        classifier_predictions = (
            predictions_df[
                predictions_df["classifier"]
                == classifier_name
            ]
        )

        y_true = (
            classifier_predictions[
                "true_label_zero_based"
            ].to_numpy()
        )

        y_pred = (
            classifier_predictions[
                "predicted_label_zero_based"
            ].to_numpy()
        )

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=list(
                range(NUM_CLASSES)
            )
        )

        cm_df = pd.DataFrame(
            cm,
            index=[
                f"true_{i}"
                for i in range(
                    1,
                    NUM_CLASSES + 1
                )
            ],
            columns=[
                f"pred_{i}"
                for i in range(
                    1,
                    NUM_CLASSES + 1
                )
            ],
        )

        safe_name = (
            classifier_name
            .lower()
            .replace(
                " ",
                "_"
            )
        )

        cm_df.to_csv(
            RESULTS_DIR
            / f"confusion_matrix_{safe_name}.csv"
        )

    # =========================================================================
    # CLASSIFICATION REPORT
    # =========================================================================

    report_lines = []

    report_lines.append(
        "NEURO-SYMPHONY — EXPERIMENT 4"
    )

    report_lines.append(
        "CHANNEL-DOMAIN FFT + WAVELET/DWT + CLASSIFIER"
    )

    report_lines.append(
        "=" * 80
    )

    report_lines.append("")

    report_lines.append(
        "IMPORTANT METHODOLOGICAL NOTE:"
    )

    report_lines.append(
        "FFT and DWT are applied across the 64 EEG channels "
        "because the current CSV contains no temporal samples."
    )

    report_lines.append(
        "Therefore these are channel-domain transformations, "
        "not temporal EEG frequency/wavelet features."
    )

    report_lines.append("")

    for classifier_name in CLASSIFIERS.keys():

        classifier_predictions = (
            predictions_df[
                predictions_df["classifier"]
                == classifier_name
            ]
        )

        y_true = (
            classifier_predictions[
                "true_label_zero_based"
            ].to_numpy()
        )

        y_pred = (
            classifier_predictions[
                "predicted_label_zero_based"
            ].to_numpy()
        )

        report = classification_report(
            y_true,
            y_pred,
            labels=list(
                range(NUM_CLASSES)
            ),
            target_names=[
                str(i)
                for i in range(
                    1,
                    NUM_CLASSES + 1
                )
            ],
            zero_division=0,
        )

        report_lines.append(
            f"CLASSIFIER: {classifier_name}"
        )

        report_lines.append(
            "-" * 80
        )

        report_lines.append(
            report
        )

        report_lines.append("")

    with open(
        RESULTS_DIR
        / "fft_wavelet_classification_reports.txt",
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "\n".join(
                report_lines
            )
        )

    # =========================================================================
    # METADATA
    # =========================================================================

    metadata = pd.DataFrame([{

        "experiment":
            "Channel-Domain FFT + Wavelet/DWT + Classifier",

        "dataset":
            str(DATA_PATH),

        "subjects":
            len(subjects),

        "classes":
            NUM_CLASSES,

        "input_channels":
            NUM_CHANNELS,

        "fft_features":
            int(
                sample_fft.shape[1]
            ),

        "dwt_features":
            int(
                sample_dwt.shape[1]
            ),

        "combined_features":
            int(
                sample_combined.shape[1]
            ),

        "wavelet":
            WAVELET,

        "dwt_level":
            DWT_LEVEL,

        "train_subjects_per_fold":
            10,

        "validation_subjects_per_fold":
            1,

        "test_subjects_per_fold":
            1,

        "folds":
            12,

        "chance_accuracy":
            CHANCE_LEVEL,

        "runtime_minutes":
            (
                time.time()
                - start_time
            ) / 60,
    }])

    metadata.to_csv(
        RESULTS_DIR
        / "fft_wavelet_experiment_metadata.csv",
        index=False,
    )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    total_runtime = (
        time.time()
        - start_time
    )

    print()
    print("=" * 80)
    print(
        "EXPERIMENT 4 — FFT + WAVELET/DWT COMPLETE"
    )
    print("=" * 80)

    print()

    print(
        "Remember:"
    )

    print(
        "FFT and DWT are CHANNEL-DOMAIN transformations."
    )

    print(
        "They are NOT temporal EEG frequency/wavelet analysis."
    )

    print()

    print(
        f"FFT features: "
        f"{sample_fft.shape[1]}"
    )

    print(
        f"DWT features: "
        f"{sample_dwt.shape[1]}"
    )

    print(
        f"Combined features: "
        f"{sample_combined.shape[1]}"
    )

    print(
        f"Chance accuracy: "
        f"{CHANCE_LEVEL * 100:.3f}%"
    )

    print()

    print(
        "FINAL CLASSIFIER RESULTS"
    )

    print(
        "-" * 80
    )

    for _, row in summary_df.iterrows():

        print(
            f"{row['classifier']}:"
        )

        print(
            f"  Train accuracy: "
            f"{row['mean_train_accuracy'] * 100:.3f}%"
        )

        print(
            f"  Validation accuracy: "
            f"{row['mean_validation_accuracy'] * 100:.3f}%"
        )

        print(
            f"  Test accuracy: "
            f"{row['mean_test_accuracy'] * 100:.3f}% "
            f"± "
            f"{row['std_test_accuracy'] * 100:.3f}%"
        )

        print(
            f"  Macro F1: "
            f"{row['mean_macro_f1'] * 100:.3f}% "
            f"± "
            f"{row['std_macro_f1'] * 100:.3f}%"
        )

        print(
            f"  Balanced accuracy: "
            f"{row['mean_balanced_accuracy'] * 100:.3f}%"
        )

        print(
            f"  Difference from chance: "
            f"{row['difference_from_chance'] * 100:+.3f} pp"
        )

        print()

    print(
        f"Runtime: "
        f"{total_runtime / 60:.2f} minutes"
    )

    print()

    print(
        "OUTPUT FILES"
    )

    print(
        "-" * 80
    )

    print(
        "Results:",
        RESULTS_DIR
        / "fft_wavelet_classifier_results.csv"
    )

    print(
        "Predictions:",
        RESULTS_DIR
        / "fft_wavelet_classifier_predictions.csv"
    )

    print(
        "Summary:",
        RESULTS_DIR
        / "fft_wavelet_classifier_summary.csv"
    )

    print(
        "Classification reports:",
        RESULTS_DIR
        / "fft_wavelet_classification_reports.txt"
    )

    print(
        "Metadata:",
        RESULTS_DIR
        / "fft_wavelet_experiment_metadata.csv"
    )

    print()

    print(
        "Done."
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()