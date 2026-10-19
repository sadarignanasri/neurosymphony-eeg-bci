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

DATA_PATH = "data/features/ds004514_fnirs_engineered.csv"

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
    "NEURO-SYMPHONY — fNIRS ENGINEERED MODELS"
)

print("\nLoading:")
print(DATA_PATH)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print("\nDataset shape:")
print(df.shape)


# ============================================================
# STANDARDIZE LABEL
# ============================================================

if "Label" in df.columns and "label" not in df.columns:

    df = df.rename(
        columns={"Label": "label"}
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
# IDENTIFY FEATURES
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

print_section(
    "FEATURE INFORMATION"
)

print(
    "Total ML features:",
    len(feature_columns)
)

print(
    "Raw fNIRS features:",
    sum(
        1
        for c in feature_columns
        if not (
            c.startswith("fnirs_")
            or c.startswith("wavelength_")
            or "_785_830_" in c
            or "_830_to_785_" in c
            or any(
                c.startswith(
                    f"S{i}_"
                )
                for i in range(1, 9)
            )
        )
    )
)


# ============================================================
# DATA
# ============================================================

X = df[
    feature_columns
]

y = df[
    "label"
]

groups = df[
    "subject_id"
]


# ============================================================
# VALIDATION
# ============================================================

print_section(
    "DATA VALIDATION"
)

print(
    "Samples:",
    len(df)
)

print(
    "Subjects:",
    groups.nunique()
)

print(
    "Classes:",
    y.nunique()
)

print(
    "Features:",
    len(feature_columns)
)

print(
    "Missing values:",
    int(
        X.isna()
        .sum()
        .sum()
    )
)

print(
    "Infinite values:",
    int(
        np.isinf(
            X.select_dtypes(
                include=np.number
            ).to_numpy()
        ).sum()
    )
)

print(
    "Duplicate feature rows:",
    X.duplicated().sum()
)


# ============================================================
# LOSO SETUP
# ============================================================

subjects = sorted(
    groups.unique()
)

print_section(
    "LEAVE-ONE-SUBJECT-OUT SETUP"
)

print(
    "Number of subjects:",
    len(subjects)
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
                    max_iter=3000,
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
    "STARTING fNIRS ENGINEERED LOSO"
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

    for test_subject in subjects:

        print(
            f"\nTesting subject: "
            f"{test_subject}"
        )

        # ----------------------------------------------------
        # LOSO SPLIT
        # ----------------------------------------------------

        train_mask = (
            groups != test_subject
        )

        test_mask = (
            groups == test_subject
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
        # FRESH MODEL
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

        all_results.append({

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
                int(
                    (
                        y_test.to_numpy()
                        ==
                        y_pred
                    ).sum()
                ),

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


# ============================================================
# FOLD RESULTS
# ============================================================

fold_results_df = pd.DataFrame(
    all_results
)

fold_results_path = os.path.join(
    RESULTS_DIR,
    "engineered_fold_results.csv"
)

fold_results_df.to_csv(
    fold_results_path,
    index=False
)


# ============================================================
# MODEL SUMMARY
# ============================================================

summary_rows = []

for model_name in models.keys():

    model_data = fold_results_df[
        fold_results_df["model"]
        ==
        model_name
    ]

    summary_rows.append({

        "model":
            model_name,

        "mean_accuracy":
            model_data[
                "accuracy"
            ].mean(),

        "std_accuracy":
            model_data[
                "accuracy"
            ].std(),

        "mean_macro_f1":
            model_data[
                "macro_f1"
            ].mean(),

        "std_macro_f1":
            model_data[
                "macro_f1"
            ].std(),

        "mean_balanced_accuracy":
            model_data[
                "balanced_accuracy"
            ].mean(),

        "std_balanced_accuracy":
            model_data[
                "balanced_accuracy"
            ].std(),

        "total_correct":
            model_data[
                "correct"
            ].sum(),

        "total_test_samples":
            model_data[
                "test_samples"
            ].sum()

    })


results_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# CHANCE LEVEL
# ============================================================

num_classes = y.nunique()

chance = 1.0 / num_classes

results_df[
    "chance_accuracy"
] = chance

results_df[
    "improvement_over_chance"
] = (
    results_df[
        "mean_accuracy"
    ]
    -
    chance
)


# ============================================================
# SORT RESULTS
# ============================================================

results_df = results_df.sort_values(
    "mean_accuracy",
    ascending=False
)


# ============================================================
# SAVE SUMMARY
# ============================================================

results_path = os.path.join(
    RESULTS_DIR,
    "engineered_model_results.csv"
)

results_df.to_csv(
    results_path,
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
    "engineered_predictions.csv"
)

predictions_df.to_csv(
    predictions_path,
    index=False
)


# ============================================================
# FINAL RESULTS
# ============================================================

print_section(
    "FINAL fNIRS ENGINEERED RESULTS"
)

display_columns = [
    "model",
    "mean_accuracy",
    "std_accuracy",
    "mean_macro_f1",
    "std_macro_f1",
    "mean_balanced_accuracy",
    "improvement_over_chance"
]

print(
    results_df[
        display_columns
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
print("BEST fNIRS ENGINEERED MODEL")
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
# COMPARE AGAINST fNIRS BASELINE
# ============================================================

baseline_path = os.path.join(
    RESULTS_DIR,
    "baseline_model_results.csv"
)

if os.path.exists(baseline_path):

    baseline_df = pd.read_csv(
        baseline_path
    )

    baseline_best = baseline_df.loc[
        baseline_df[
            "mean_accuracy"
        ].idxmax()
    ]

    engineered_best = results_df.iloc[0]

    print("\n")
    print("=" * 70)
    print("BASELINE vs ENGINEERED COMPARISON")
    print("=" * 70)

    print(
        f"Baseline best model: "
        f"{baseline_best['model']}"
    )

    print(
        f"Baseline accuracy: "
        f"{baseline_best['mean_accuracy'] * 100:.2f}%"
    )

    print(
        f"Engineered best model: "
        f"{engineered_best['model']}"
    )

    print(
        f"Engineered accuracy: "
        f"{engineered_best['mean_accuracy'] * 100:.2f}%"
    )

    improvement = (
        engineered_best[
            "mean_accuracy"
        ]
        -
        baseline_best[
            "mean_accuracy"
        ]
    )

    print(
        f"Change in accuracy: "
        f"{improvement * 100:.2f} "
        f"percentage points"
    )


# ============================================================
# TOTAL TIME
# ============================================================

total_time = (
    time.time()
    -
    overall_start
)


# ============================================================
# OUTPUT FILES
# ============================================================

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


# ============================================================
# COMPLETED
# ============================================================

print("\n")
print("=" * 70)
print(
    "fNIRS ENGINEERED MODEL EXPERIMENT COMPLETED"
)
print("=" * 70)