import os
import time

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/features/ds004514_fnirs.csv"

RESULTS_DIR = "results/fnirs"

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

RANDOM_STATE = 42


# ============================================================
# HELPER
# ============================================================

def print_section(title):

    print("\n")
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print_section(
    "NEURO-SYMPHONY — fNIRS BASELINE MODELS"
)

print("\nLoading:")
print(DATA_PATH)

if not os.path.exists(DATA_PATH):

    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}"
    )

df = pd.read_csv(
    DATA_PATH
)

print("\nDataset shape:")
print(df.shape)


# ============================================================
# STANDARDIZE LABEL COLUMN
# ============================================================

if "Label" in df.columns and "label" not in df.columns:

    df = df.rename(
        columns={
            "Label": "label"
        }
    )

elif "label" not in df.columns:

    raise ValueError(
        "Neither 'label' nor 'Label' exists."
    )


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "subject_id",
    "trial_id",
    "label_name",
    "label"
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Missing required column: {column}"
        )


# ============================================================
# FEATURE COLUMNS
# ============================================================

metadata_columns = [
    "subject_id",
    "trial_id",
    "label_name",
    "label"
]

feature_columns = [
    column
    for column in df.columns
    if column not in metadata_columns
]

print("\nNumber of fNIRS features:")
print(len(feature_columns))

print("\nFeatures:")

for feature in feature_columns:

    print(" ", feature)


# ============================================================
# DATA QUALITY
# ============================================================

print_section(
    "DATA QUALITY"
)

X_all = df[
    feature_columns
]

y_all = df[
    "label"
]

groups = df[
    "subject_id"
]

print(
    "Missing values:",
    int(X_all.isna().sum().sum())
)

print(
    "Infinite values:",
    int(
        np.isinf(
            X_all.to_numpy(
                dtype=float
            )
        ).sum()
    )
)

print(
    "Subjects:",
    groups.nunique()
)

print(
    "Classes:",
    y_all.nunique()
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print_section(
    "CLASS DISTRIBUTION"
)

class_distribution = (
    df.groupby(
        [
            "label",
            "label_name"
        ]
    )
    .size()
    .reset_index(
        name="samples"
    )
    .sort_values(
        "label"
    )
)

print(
    class_distribution.to_string(
        index=False
    )
)


# ============================================================
# LOSO SUBJECTS
# ============================================================

subjects = sorted(
    groups.unique()
)

print_section(
    "LEAVE-ONE-SUBJECT-OUT SETUP"
)

print(
    "Number of LOSO folds:",
    len(subjects)
)

for subject in subjects:

    train_count = (
        groups != subject
    ).sum()

    test_count = (
        groups == subject
    ).sum()

    print(
        f"{subject}: "
        f"train={train_count}, "
        f"test={test_count}"
    )


# ============================================================
# MODELS
# ============================================================

models = {

    "Logistic Regression":
        Pipeline([
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
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE
                )
            )
        ]),

    "Linear SVM":
        Pipeline([
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
                "classifier",
                LinearSVC(
                    max_iter=5000,
                    random_state=RANDOM_STATE
                )
            )
        ]),

    "LDA":
        Pipeline([
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
                "classifier",
                LinearDiscriminantAnalysis(
                    solver="lsqr",
                    shrinkage="auto"
                )
            )
        ]),

    "KNN":
        Pipeline([
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
                "classifier",
                KNeighborsClassifier(
                    n_neighbors=5
                )
            )
        ]),

    "Random Forest":
        Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                )
            )
        ]),

    "Extra Trees":
        Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "classifier",
                ExtraTreesClassifier(
                    n_estimators=300,
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                )
            )
        ])
}


# ============================================================
# TRAINING
# ============================================================

print_section(
    "STARTING fNIRS LOSO BASELINE"
)

all_results = []
all_predictions = []

overall_start = time.time()


for model_name, base_model in models.items():

    print("\n")
    print("-" * 70)
    print(
        f"MODEL: {model_name}"
    )
    print("-" * 70)

    model_start = time.time()

    fold_results = []

    for test_subject in subjects:

        print(
            f"\nTesting subject: "
            f"{test_subject}"
        )

        train_mask = (
            groups != test_subject
        )

        test_mask = (
            groups == test_subject
        )

        X_train = X_all.loc[
            train_mask
        ]

        X_test = X_all.loc[
            test_mask
        ]

        y_train = y_all.loc[
            train_mask
        ]

        y_test = y_all.loc[
            test_mask
        ]

        # ----------------------------------------------------
        # CLONE MODEL
        # ----------------------------------------------------

        model = clone(
            base_model
        )

        # ----------------------------------------------------
        # TRAIN
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

        correct = int(
            (
                y_test.to_numpy()
                ==
                y_pred
            ).sum()
        )

        fold_results.append({

            "model":
                model_name,

            "test_subject":
                test_subject,

            "accuracy":
                accuracy,

            "macro_f1":
                macro_f1,

            "balanced_accuracy":
                balanced_accuracy,

            "correct":
                correct,

            "test_samples":
                len(y_test)

        })

        # ----------------------------------------------------
        # SAVE PREDICTIONS
        # ----------------------------------------------------

        prediction_subset = df.loc[
            test_mask,
            [
                "subject_id",
                "trial_id",
                "label_name",
                "label"
            ]
        ].copy()

        prediction_subset[
            "predicted_label"
        ] = y_pred

        label_to_name = (
            df[
                [
                    "label",
                    "label_name"
                ]
            ]
            .drop_duplicates()
            .set_index(
                "label"
            )[
                "label_name"
            ]
            .to_dict()
        )

        prediction_subset[
            "predicted_label_name"
        ] = (
            prediction_subset[
                "predicted_label"
            ]
            .map(
                label_to_name
            )
        )

        prediction_subset[
            "correct"
        ] = (
            prediction_subset[
                "label"
            ]
            ==
            prediction_subset[
                "predicted_label"
            ]
        )

        prediction_subset[
            "model"
        ] = model_name

        all_predictions.append(
            prediction_subset
        )

        print(
            f"  Accuracy: "
            f"{accuracy * 100:.2f}%"
        )

    # --------------------------------------------------------
    # MODEL SUMMARY
    # --------------------------------------------------------

    fold_df = pd.DataFrame(
        fold_results
    )

    mean_accuracy = (
        fold_df["accuracy"]
        .mean()
    )

    std_accuracy = (
        fold_df["accuracy"]
        .std()
    )

    mean_f1 = (
        fold_df["macro_f1"]
        .mean()
    )

    std_f1 = (
        fold_df["macro_f1"]
        .std()
    )

    mean_balanced = (
        fold_df["balanced_accuracy"]
        .mean()
    )

    std_balanced = (
        fold_df["balanced_accuracy"]
        .std()
    )

    elapsed = (
        time.time()
        - model_start
    )

    all_results.append({

        "model":
            model_name,

        "mean_accuracy":
            mean_accuracy,

        "std_accuracy":
            std_accuracy,

        "mean_macro_f1":
            mean_f1,

        "std_macro_f1":
            std_f1,

        "mean_balanced_accuracy":
            mean_balanced,

        "std_balanced_accuracy":
            std_balanced,

        "training_time_seconds":
            elapsed

    })

    print(
        "\nModel summary:"
    )

    print(
        f"Mean Accuracy: "
        f"{mean_accuracy * 100:.2f}% "
        f"± {std_accuracy * 100:.2f}%"
    )

    print(
        f"Mean Macro F1: "
        f"{mean_f1 * 100:.2f}% "
        f"± {std_f1 * 100:.2f}%"
    )

    print(
        f"Mean Balanced Accuracy: "
        f"{mean_balanced * 100:.2f}% "
        f"± {std_balanced * 100:.2f}%"
    )

    print(
        f"Time: "
        f"{elapsed:.2f} seconds"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)

results_path = os.path.join(
    RESULTS_DIR,
    "baseline_model_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# ============================================================
# SAVE FOLD RESULTS
# ============================================================

# Reconstruct fold-level results
# from predictions and metrics

fold_results_all = []

for model_name in models.keys():

    model_predictions = [
        x
        for x in all_predictions
        if x["model"].iloc[0]
        == model_name
    ]

    for prediction_df in model_predictions:

        y_true = prediction_df[
            "label"
        ]

        y_pred = prediction_df[
            "predicted_label"
        ]

        fold_results_all.append({

            "model":
                model_name,

            "test_subject":
                prediction_df[
                    "subject_id"
                ].iloc[0],

            "accuracy":
                accuracy_score(
                    y_true,
                    y_pred
                ),

            "macro_f1":
                f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0
                ),

            "balanced_accuracy":
                balanced_accuracy_score(
                    y_true,
                    y_pred
                ),

            "correct":
                int(
                    prediction_df[
                        "correct"
                    ].sum()
                ),

            "test_samples":
                len(prediction_df)

        })

fold_results_df = pd.DataFrame(
    fold_results_all
)

fold_results_path = os.path.join(
    RESULTS_DIR,
    "baseline_fold_results.csv"
)

fold_results_df.to_csv(
    fold_results_path,
    index=False
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

predictions_df = pd.concat(
    all_predictions,
    ignore_index=True
)

predictions_path = os.path.join(
    RESULTS_DIR,
    "baseline_predictions.csv"
)

predictions_df.to_csv(
    predictions_path,
    index=False
)


# ============================================================
# FINAL COMPARISON
# ============================================================

print_section(
    "FINAL fNIRS BASELINE RESULTS"
)

chance = 1 / df[
    "label"
].nunique()

results_df[
    "chance_accuracy"
] = chance

results_df[
    "improvement_over_chance"
] = (
    results_df[
        "mean_accuracy"
    ]
    - chance
)

results_df = results_df.sort_values(
    "mean_accuracy",
    ascending=False
)

print(
    results_df[
        [
            "model",
            "mean_accuracy",
            "std_accuracy",
            "mean_macro_f1",
            "mean_balanced_accuracy",
            "improvement_over_chance"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# BEST MODEL
# ============================================================

best_model = results_df.iloc[0]

print("\n")
print("=" * 70)
print("BEST fNIRS BASELINE MODEL")
print("=" * 70)

print(
    "Model:",
    best_model["model"]
)

print(
    f"Mean Accuracy: "
    f"{best_model['mean_accuracy'] * 100:.2f}%"
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
# TOTAL TIME
# ============================================================

total_time = (
    time.time()
    - overall_start
)

print("\n")
print("=" * 70)
print("OUTPUT FILES")
print("=" * 70)

print(
    "1.",
    results_path
)

print(
    "2.",
    fold_results_path
)

print(
    "3.",
    predictions_path
)

print(
    "\nTotal experiment time:",
    f"{total_time:.2f} seconds"
)

print("\n")
print("=" * 70)
print("fNIRS BASELINE EXPERIMENT COMPLETED")
print("=" * 70)