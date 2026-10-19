"""
NEURO-SYMPHONY
EEG + fNIRS EARLY FEATURE FUSION

Purpose:
    Combine engineered EEG and engineered fNIRS features
    at the trial level and evaluate multimodal classification.

Data:
    EEG:
        data/features/ds004514_eeg_engineered.csv

    fNIRS:
        data/features/ds004514_fnirs_engineered.csv

Alignment:
    subject_id + trial_id

Evaluation:
    Leave-One-Subject-Out (LOSO)

Models:
    - Logistic Regression
    - Linear SVM
    - LDA
    - KNN
    - Random Forest
    - Extra Trees

Important:
    Imputation and scaling are performed inside each
    training fold to prevent data leakage.
"""

import os
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score
)


# ============================================================
# WARNING CONTROL
# ============================================================

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

EEG_FILE = (
    "data/features/"
    "ds004514_eeg_engineered.csv"
)

FNIRS_FILE = (
    "data/features/"
    "ds004514_fnirs_engineered.csv"
)

OUTPUT_DIR = "results/fusion"


RESULTS_FILE = os.path.join(
    OUTPUT_DIR,
    "eeg_fnirs_fusion_model_results.csv"
)

FOLD_RESULTS_FILE = os.path.join(
    OUTPUT_DIR,
    "eeg_fnirs_fusion_fold_results.csv"
)

PREDICTIONS_FILE = os.path.join(
    OUTPUT_DIR,
    "eeg_fnirs_fusion_predictions.csv"
)

ALIGNMENT_FILE = os.path.join(
    OUTPUT_DIR,
    "eeg_fnirs_alignment_report.csv"
)

RANDOM_STATE = 42


# ============================================================
# HEADER
# ============================================================

print("\n" + "=" * 70)
print("NEURO-SYMPHONY — EEG + fNIRS FEATURE FUSION")
print("=" * 70)

start_time = time.time()

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# CHECK INPUT FILES
# ============================================================

print("\n" + "=" * 70)
print("CHECKING INPUT FILES")
print("=" * 70)


if not os.path.exists(EEG_FILE):

    raise FileNotFoundError(
        f"\nEEG engineered file not found:\n{EEG_FILE}"
    )


if not os.path.exists(FNIRS_FILE):

    raise FileNotFoundError(
        f"\nfNIRS engineered file not found:\n{FNIRS_FILE}"
    )


print(
    f"EEG file found:\n{EEG_FILE}"
)

print(
    f"\nfNIRS file found:\n{FNIRS_FILE}"
)


# ============================================================
# LOAD EEG
# ============================================================

print("\n" + "=" * 70)
print("LOADING EEG ENGINEERED DATA")
print("=" * 70)

eeg = pd.read_csv(
    EEG_FILE
)

print(
    f"EEG shape: {eeg.shape}"
)


# ============================================================
# LOAD fNIRS
# ============================================================

print("\n" + "=" * 70)
print("LOADING fNIRS ENGINEERED DATA")
print("=" * 70)

fnirs = pd.read_csv(
    FNIRS_FILE
)

print(
    f"fNIRS shape: {fnirs.shape}"
)


# ============================================================
# STANDARDIZE LABEL COLUMNS
# ============================================================

def standardize_label(df, name):

    if "label" in df.columns:

        return df

    if "Label" in df.columns:

        df = df.rename(
            columns={
                "Label": "label"
            }
        )

        print(
            f"{name}: renamed "
            "'Label' → 'label'"
        )

        return df

    if "label_name" in df.columns:

        raise ValueError(
            f"{name} has no numeric 'label' column "
            "and only contains 'label_name'."
        )

    raise ValueError(
        f"{name}: no label column found."
    )


eeg = standardize_label(
    eeg,
    "EEG"
)

fnirs = standardize_label(
    fnirs,
    "fNIRS"
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "subject_id",
    "trial_id",
    "label"
]


for col in required_columns:

    if col not in eeg.columns:

        raise ValueError(
            f"EEG missing required column: {col}"
        )


for col in required_columns:

    if col not in fnirs.columns:

        raise ValueError(
            f"fNIRS missing required column: {col}"
        )


# ============================================================
# REMOVE DUPLICATE KEYS
# ============================================================

if eeg.duplicated(
    subset=[
        "subject_id",
        "trial_id"
    ]
).any():

    raise ValueError(
        "EEG contains duplicate "
        "(subject_id, trial_id) keys."
    )


if fnirs.duplicated(
    subset=[
        "subject_id",
        "trial_id"
    ]
).any():

    raise ValueError(
        "fNIRS contains duplicate "
        "(subject_id, trial_id) keys."
    )


# ============================================================
# DETERMINE FEATURE COLUMNS
# ============================================================

metadata_columns = {
    "subject_id",
    "trial_id",
    "label",
    "label_name"
}


eeg_features = [
    col
    for col in eeg.columns
    if col not in metadata_columns
    and pd.api.types.is_numeric_dtype(
        eeg[col]
    )
]


fnirs_features = [
    col
    for col in fnirs.columns
    if col not in metadata_columns
    and pd.api.types.is_numeric_dtype(
        fnirs[col]
    )
]


# ============================================================
# FEATURE INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("FEATURE INFORMATION")
print("=" * 70)

print(
    f"EEG ML features: "
    f"{len(eeg_features)}"
)

print(
    f"fNIRS ML features: "
    f"{len(fnirs_features)}"
)

print(
    f"Combined ML features: "
    f"{len(eeg_features) + len(fnirs_features)}"
)


# ============================================================
# VALIDATE LABEL CONSISTENCY
# ============================================================

print("\n" + "=" * 70)
print("VALIDATING LABEL CONSISTENCY")
print("=" * 70)


eeg_labels = (
    eeg[
        [
            "subject_id",
            "trial_id",
            "label"
        ]
    ]
    .rename(
        columns={
            "label": "eeg_label"
        }
    )
)


fnirs_labels = (
    fnirs[
        [
            "subject_id",
            "trial_id",
            "label"
        ]
    ]
    .rename(
        columns={
            "label": "fnirs_label"
        }
    )
)


label_check = pd.merge(
    eeg_labels,
    fnirs_labels,
    on=[
        "subject_id",
        "trial_id"
    ],
    how="outer",
    indicator=True
)


# ------------------------------------------------------------
# Missing alignment keys
# ------------------------------------------------------------

missing_eeg = (
    label_check["_merge"] == "right_only"
).sum()

missing_fnirs = (
    label_check["_merge"] == "left_only"
).sum()


print(
    f"Trials missing from EEG: "
    f"{missing_eeg}"
)

print(
    f"Trials missing from fNIRS: "
    f"{missing_eeg if False else missing_fnirs}"
)


# ------------------------------------------------------------
# Keep only matched trials for label comparison
# ------------------------------------------------------------

matched_labels = label_check[
    label_check["_merge"] == "both"
].copy()


label_mismatches = (
    matched_labels["eeg_label"]
    != matched_labels["fnirs_label"]
)


mismatch_count = int(
    label_mismatches.sum()
)


print(
    f"Matched trials: "
    f"{len(matched_labels)}"
)

print(
    f"Label mismatches: "
    f"{mismatch_count}"
)


if missing_eeg != 0 or missing_fnirs != 0:

    raise ValueError(
        "EEG and fNIRS do not contain exactly "
        "the same subject/trial keys."
    )


if mismatch_count != 0:

    raise ValueError(
        "EEG and fNIRS labels do not match."
    )


# ============================================================
# ALIGN DATA
# ============================================================

print("\n" + "=" * 70)
print("ALIGNING EEG + fNIRS")
print("=" * 70)


eeg_selected = eeg[
    [
        "subject_id",
        "trial_id",
        "label"
    ]
    + eeg_features
].copy()


fnirs_selected = fnirs[
    [
        "subject_id",
        "trial_id"
    ]
    + fnirs_features
].copy()


# ------------------------------------------------------------
# Rename fNIRS feature columns if collision occurs
# ------------------------------------------------------------

overlapping_features = (
    set(eeg_features)
    .intersection(
        set(fnirs_features)
    )
)


if overlapping_features:

    print(
        "\nFeature name collisions detected:"
    )

    for col in sorted(
        overlapping_features
    ):

        print(
            f"  {col}"
        )

    fnirs_selected = (
        fnirs_selected.rename(
            columns={
                col: f"fnirs_{col}"
                for col in overlapping_features
            }
        )
    )

    fnirs_features = [
        (
            f"fnirs_{col}"
            if col in overlapping_features
            else col
        )
        for col in fnirs_features
    ]


# ------------------------------------------------------------
# Merge
# ------------------------------------------------------------

fusion_df = pd.merge(

    eeg_selected,

    fnirs_selected,

    on=[
        "subject_id",
        "trial_id"
    ],

    how="inner",

    validate="one_to_one"
)


# ============================================================
# FUSION VALIDATION
# ============================================================

print(
    f"\nFusion shape: "
    f"{fusion_df.shape}"
)

print(
    f"Fusion samples: "
    f"{len(fusion_df)}"
)

print(
    f"Fusion features: "
    f"{len(eeg_features) + len(fnirs_features)}"
)


# ------------------------------------------------------------
# Verify labels against EEG
# ------------------------------------------------------------

fusion_label_counts = (
    fusion_df["label"]
    .value_counts()
    .sort_index()
)


print("\nFusion class distribution:")

for label, count in fusion_label_counts.items():

    print(
        f"  Class {label}: "
        f"{count}"
    )


# ------------------------------------------------------------
# Verify subjects
# ------------------------------------------------------------

fusion_subject_counts = (
    fusion_df["subject_id"]
    .value_counts()
    .sort_index()
)


print("\nFusion subject distribution:")

for subject, count in (
    fusion_subject_counts.items()
):

    print(
        f"  {subject}: "
        f"{count}"
    )


# ------------------------------------------------------------
# Duplicate fusion keys
# ------------------------------------------------------------

duplicate_fusion_keys = (
    fusion_df.duplicated(
        subset=[
            "subject_id",
            "trial_id"
        ]
    ).sum()
)


print(
    f"\nDuplicate fusion keys: "
    f"{duplicate_fusion_keys}"
)


if duplicate_fusion_keys != 0:

    raise ValueError(
        "Duplicate fusion keys detected."
    )


# ============================================================
# SAVE ALIGNMENT REPORT
# ============================================================

alignment_rows = []

for subject in sorted(
    fusion_df["subject_id"].unique()
):

    subject_data = fusion_df[
        fusion_df["subject_id"] == subject
    ]

    alignment_rows.append({

        "subject_id":
            subject,

        "fusion_trials":
            len(subject_data),

        "unique_trial_ids":
            subject_data[
                "trial_id"
            ].nunique(),

        "classes":
            subject_data[
                "label"
            ].nunique(),

        "alignment_valid":
            (
                len(subject_data) == 720
                and
                subject_data[
                    "trial_id"
                ].nunique() == 720
            )
    })


alignment_df = pd.DataFrame(
    alignment_rows
)


alignment_df.to_csv(
    ALIGNMENT_FILE,
    index=False
)


# ============================================================
# FINAL ALIGNMENT CHECK
# ============================================================

if len(fusion_df) != 8640:

    raise ValueError(
        f"Expected 8640 aligned trials, "
        f"but found {len(fusion_df)}."
    )


if (
    fusion_df["subject_id"]
    .nunique()
    != 12
):

    raise ValueError(
        "Expected 12 subjects."
    )


if (
    fusion_df["label"]
    .nunique()
    != 8
):

    raise ValueError(
        "Expected 8 classes."
    )


print(
    "\n✓ EEG/fNIRS trial alignment verified."
)

print(
    "✓ 8640/8640 trials matched."
)

print(
    "✓ 12/12 subjects matched."
)

print(
    "✓ 8/8 classes present."
)

print(
    "✓ No duplicate subject/trial keys."
)


# ============================================================
# PREPARE ML DATA
# ============================================================

feature_columns = (
    eeg_features
    + fnirs_features
)


X = fusion_df[
    feature_columns
]

y = fusion_df[
    "label"
]

groups = fusion_df[
    "subject_id"
]


# ============================================================
# VALIDATE NUMERIC FEATURES
# ============================================================

print("\n" + "=" * 70)
print("FUSION DATA VALIDATION")
print("=" * 70)


print(
    f"Samples: {len(X)}"
)

print(
    f"Subjects: "
    f"{groups.nunique()}"
)

print(
    f"Classes: "
    f"{y.nunique()}"
)

print(
    f"Features: "
    f"{len(feature_columns)}"
)

print(
    f"Missing values: "
    f"{int(X.isna().sum().sum())}"
)


numeric_X = X.select_dtypes(
    include=np.number
)


infinite_count = np.isinf(
    numeric_X.to_numpy()
).sum()


print(
    f"Infinite values: "
    f"{int(infinite_count)}"
)


if infinite_count != 0:

    raise ValueError(
        "Infinite values found in fusion features."
    )


non_numeric_features = [
    col
    for col in feature_columns
    if not pd.api.types.is_numeric_dtype(
        X[col]
    )
]


if non_numeric_features:

    raise ValueError(
        "Non-numeric ML features detected: "
        + str(non_numeric_features)
    )


# ============================================================
# LOSO SETUP
# ============================================================

subjects = sorted(
    groups.unique()
)

classes = sorted(
    y.unique()
)


print("\n" + "=" * 70)
print("LEAVE-ONE-SUBJECT-OUT SETUP")
print("=" * 70)


print(
    f"Number of subjects: "
    f"{len(subjects)}"
)

print(
    f"Number of LOSO folds: "
    f"{len(subjects)}"
)


for subject in subjects:

    train_mask = (
        groups != subject
    )

    test_mask = (
        groups == subject
    )

    print(
        f"{subject}: "
        f"train={train_mask.sum()}, "
        f"test={test_mask.sum()}"
    )


# ============================================================
# MODEL DEFINITIONS
# ============================================================

models = {

    # --------------------------------------------------------
    # LOGISTIC REGRESSION
    # --------------------------------------------------------

    "Logistic Regression": Pipeline([

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",
            StandardScaler()
        ),

        (
            "model",
            LogisticRegression(
                max_iter=3000,
                random_state=RANDOM_STATE
            )
        )
    ]),


    # --------------------------------------------------------
    # LINEAR SVM
    # --------------------------------------------------------

    "Linear SVM": Pipeline([

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",
            StandardScaler()
        ),

        (
            "model",
            SVC(
                kernel="linear",
                random_state=RANDOM_STATE
            )
        )
    ]),


    # --------------------------------------------------------
    # LDA
    # --------------------------------------------------------

    "LDA": Pipeline([

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",
            StandardScaler()
        ),

        (
            "model",
            LinearDiscriminantAnalysis(
                solver="lsqr",
                shrinkage="auto"
            )
        )
    ]),


    # --------------------------------------------------------
    # KNN
    # --------------------------------------------------------

    "KNN": Pipeline([

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",
            StandardScaler()
        ),

        (
            "model",
            KNeighborsClassifier(
                n_neighbors=5
            )
        )
    ]),


    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    "Random Forest": Pipeline([

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "model",
            RandomForestClassifier(
                n_estimators=300,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ]),


    # --------------------------------------------------------
    # EXTRA TREES
    # --------------------------------------------------------

    "Extra Trees": Pipeline([

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "model",
            ExtraTreesClassifier(
                n_estimators=300,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ])
}


# ============================================================
# START LOSO FUSION
# ============================================================

print("\n" + "=" * 70)
print("STARTING EEG + fNIRS LOSO FUSION")
print("=" * 70)


all_results = []

all_fold_results = []

all_predictions = []


# ============================================================
# MODEL LOOP
# ============================================================

for model_name, model in models.items():

    print("\n")

    print("-" * 70)

    print(
        f"MODEL: {model_name}"
    )

    print("-" * 70)


    # --------------------------------------------------------
    # SUBJECT LOOP
    # --------------------------------------------------------

    for subject in subjects:

        print(
            f"\nTesting subject: "
            f"{subject}"
        )


        # ----------------------------------------------------
        # Train/test masks
        # ----------------------------------------------------

        train_mask = (
            groups != subject
        )

        test_mask = (
            groups == subject
        )


        X_train = X.loc[
            train_mask
        ]

        X_test = X.loc[
            test_mask
        ]

        y_train = y.loc[
            train_mask
        ]

        y_test = y.loc[
            test_mask
        ]


        # ----------------------------------------------------
        # TRAIN
        #
        # All preprocessing occurs inside the pipeline.
        # ----------------------------------------------------

        model.fit(
            X_train,
            y_train
        )


        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        y_pred = model.predict(
            X_test
        )


        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            y_pred
        )


        macro_f1 = f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )


        balanced_accuracy = (
            balanced_accuracy_score(
                y_test,
                y_pred
            )
        )


        # ----------------------------------------------------
        # PRINT
        # ----------------------------------------------------

        print(
            f"  Accuracy: "
            f"{accuracy * 100:.2f}%"
        )

        print(
            f"  Macro F1: "
            f"{macro_f1 * 100:.2f}%"
        )

        print(
            f"  Balanced Accuracy: "
            f"{balanced_accuracy * 100:.2f}%"
        )


        # ----------------------------------------------------
        # STORE FOLD RESULT
        # ----------------------------------------------------

        fold_result = {

            "model":
                model_name,

            "subject_id":
                subject,

            "accuracy":
                accuracy,

            "macro_f1":
                macro_f1,

            "balanced_accuracy":
                balanced_accuracy,

            "train_samples":
                len(X_train),

            "test_samples":
                len(X_test)
        }


        all_fold_results.append(
            fold_result
        )


        # ----------------------------------------------------
        # STORE PREDICTIONS
        # ----------------------------------------------------

        test_rows = fusion_df.loc[
            test_mask
        ]


        for (
            (_, row),
            predicted_label
        ) in zip(
            test_rows.iterrows(),
            y_pred
        ):

            all_predictions.append({

                "model":
                    model_name,

                "subject_id":
                    row["subject_id"],

                "trial_id":
                    row["trial_id"],

                "true_label":
                    row["label"],

                "predicted_label":
                    predicted_label
            })


# ============================================================
# CREATE RESULTS DATAFRAME
# ============================================================

fold_results_df = pd.DataFrame(
    all_fold_results
)


predictions_df = pd.DataFrame(
    all_predictions
)


# ============================================================
# MODEL SUMMARY
# ============================================================

model_summary = (

    fold_results_df

    .groupby("model")

    .agg(

        mean_accuracy=(
            "accuracy",
            "mean"
        ),

        std_accuracy=(
            "accuracy",
            "std"
        ),

        mean_macro_f1=(
            "macro_f1",
            "mean"
        ),

        std_macro_f1=(
            "macro_f1",
            "std"
        ),

        mean_balanced_accuracy=(
            "balanced_accuracy",
            "mean"
        ),

        std_balanced_accuracy=(
            "balanced_accuracy",
            "std"
        )
    )

    .reset_index()
)


# ============================================================
# CHANCE LEVEL
# ============================================================

chance = 1 / len(classes)


model_summary[
    "improvement_over_chance"
] = (

    model_summary[
        "mean_accuracy"
    ]
    - chance
)


# ============================================================
# SORT
# ============================================================

model_summary = (

    model_summary

    .sort_values(
        "mean_accuracy",
        ascending=False
    )

    .reset_index(
        drop=True
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

model_summary.to_csv(
    RESULTS_FILE,
    index=False
)


fold_results_df.to_csv(
    FOLD_RESULTS_FILE,
    index=False
)


predictions_df.to_csv(
    PREDICTIONS_FILE,
    index=False
)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL EEG + fNIRS FUSION RESULTS")
print("=" * 70)


print(
    model_summary.to_string(
        index=False
    )
)


# ============================================================
# BEST MODEL
# ============================================================

best_idx = (
    model_summary[
        "mean_accuracy"
    ].idxmax()
)


best_model = (
    model_summary.loc[
        best_idx
    ]
)


print("\n" + "=" * 70)
print("BEST EEG + fNIRS FUSION MODEL")
print("=" * 70)


print(
    f"Model: "
    f"{best_model['model']}"
)


print(
    f"Mean Accuracy: "
    f"{best_model['mean_accuracy'] * 100:.2f}%"
)


print(
    f"Std Accuracy: "
    f"{best_model['std_accuracy'] * 100:.2f}%"
)


print(
    f"Mean Macro F1: "
    f"{best_model['mean_macro_f1'] * 100:.2f}%"
)


print(
    f"Mean Balanced Accuracy: "
    f"{best_model['mean_balanced_accuracy'] * 100:.2f}%"
)


print(
    f"Chance Level: "
    f"{chance * 100:.2f}%"
)


print(
    f"Improvement over chance: "
    f"{best_model['improvement_over_chance'] * 100:.2f} "
    f"percentage points"
)


# ============================================================
# PER-SUBJECT BEST PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("BEST FUSION MODEL FOR EACH SUBJECT")
print("=" * 70)


best_subject_results = []


for subject in subjects:

    subject_data = fold_results_df[
        fold_results_df[
            "subject_id"
        ] == subject
    ]


    best_idx = (
        subject_data[
            "accuracy"
        ].idxmax()
    )


    best = (
        subject_data.loc[
            best_idx
        ]
    )


    best_subject_results.append({

        "subject_id":
            subject,

        "best_model":
            best["model"],

        "accuracy":
            best["accuracy"],

        "macro_f1":
            best["macro_f1"],

        "balanced_accuracy":
            best["balanced_accuracy"]
    })


    print(
        f"{subject}: "
        f"{best['model']} — "
        f"{best['accuracy'] * 100:.2f}% "
        f"(F1: "
        f"{best['macro_f1'] * 100:.2f}%)"
    )


best_subject_df = pd.DataFrame(
    best_subject_results
)


best_subject_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "eeg_fnirs_fusion_best_per_subject.csv"
    ),
    index=False
)


# ============================================================
# OVERALL BEST-SUBJECT STATISTICS
# ============================================================

best_subject_mean = (
    best_subject_df[
        "accuracy"
    ].mean()
)


best_subject_std = (
    best_subject_df[
        "accuracy"
    ].std()
)


print("\n" + "=" * 70)
print("BEST-SUBJECT FUSION STATISTICS")
print("=" * 70)


print(
    f"Mean best-subject accuracy: "
    f"{best_subject_mean * 100:.2f}%"
)


print(
    f"Std best-subject accuracy: "
    f"{best_subject_std * 100:.2f}%"
)


print(
    f"Chance level: "
    f"{chance * 100:.2f}%"
)


print(
    f"Improvement over chance: "
    f"{(best_subject_mean - chance) * 100:.2f} "
    f"percentage points"
)


# ============================================================
# OUTPUT FILES
# ============================================================

print("\n" + "=" * 70)
print("OUTPUT FILES")
print("=" * 70)


print(
    "1. results/fusion/"
    "eeg_fnirs_fusion_model_results.csv"
)


print(
    "2. results/fusion/"
    "eeg_fnirs_fusion_fold_results.csv"
)


print(
    "3. results/fusion/"
    "eeg_fnirs_fusion_predictions.csv"
)


print(
    "4. results/fusion/"
    "eeg_fnirs_alignment_report.csv"
)


print(
    "5. results/fusion/"
    "eeg_fnirs_fusion_best_per_subject.csv"
)


# ============================================================
# FINISH
# ============================================================

elapsed = (
    time.time()
    - start_time
)


print(
    f"\nTotal experiment time: "
    f"{elapsed:.2f} seconds"
)


print("\n" + "=" * 70)
print(
    "EEG + fNIRS FUSION EXPERIMENT COMPLETED"
)
print("=" * 70)