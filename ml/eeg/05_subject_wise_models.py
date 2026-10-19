import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/features/ds004514_eeg_engineered.csv"

RESULTS_DIR = "results/eeg/subject_wise"
os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("SUBJECT-WISE EEG ACTION CLASSIFICATION")
print("=" * 70)

print("\nLoading dataset:")
print(DATA_PATH)

df = pd.read_csv(DATA_PATH)

print("\nDataset shape:", df.shape)


# ============================================================
# BASIC VALIDATION
# ============================================================

required_columns = ["subject_id", "trial_id", "label_name", "label"]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

print("\nSubjects:")
print(sorted(df["subject_id"].unique()))

print("\nNumber of subjects:", df["subject_id"].nunique())

print("\nClasses:")
print(df["label_name"].value_counts().sort_index())


# ============================================================
# IDENTIFY FEATURE COLUMNS
# ============================================================

metadata_columns = [
    "subject_id",
    "trial_id",
    "label_name",
    "label"
]

feature_columns = [
    col for col in df.columns
    if col not in metadata_columns
]

print("\nNumber of ML features:", len(feature_columns))

X_all = df[feature_columns]
y_all = df["label"]


# ============================================================
# CHECK DATA QUALITY
# ============================================================

print("\nChecking data quality...")

missing_values = X_all.isna().sum().sum()
infinite_values = np.isinf(X_all.to_numpy()).sum()

print("Missing feature values:", missing_values)
print("Infinite feature values:", infinite_values)

if missing_values > 0:
    raise ValueError("Dataset contains missing feature values.")

if infinite_values > 0:
    raise ValueError("Dataset contains infinite feature values.")


# ============================================================
# MODEL
# ============================================================

model = Pipeline([
    ("scaler", StandardScaler()),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE
        )
    )
])


# ============================================================
# SUBJECT-WISE TRAINING
# ============================================================

subjects = sorted(df["subject_id"].unique())

subject_results = []
all_predictions = []

print("\n")
print("=" * 70)
print("STARTING SUBJECT-WISE EXPERIMENT")
print("=" * 70)


for subject in subjects:

    print("\n" + "-" * 70)
    print(f"SUBJECT: {subject}")
    print("-" * 70)

    subject_df = df[df["subject_id"] == subject].copy()

    X = subject_df[feature_columns]
    y = subject_df["label"]

    print("Total samples:", len(subject_df))
    print("Classes:", y.nunique())

    print("\nClass distribution:")
    print(y.value_counts().sort_index().to_dict())


    # --------------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        subject_df.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print("\nTraining model...")

    model.fit(X_train, y_train)


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    y_pred = model.predict(X_test)


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro"
    )

    balanced_acc = balanced_accuracy_score(
        y_test,
        y_pred
    )

    correct = int((y_test == y_pred).sum())
    total = len(y_test)

    chance_accuracy = 1 / y.nunique()

    improvement = accuracy - chance_accuracy

    print("\nRESULTS")
    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Macro F1:", round(macro_f1 * 100, 2), "%")
    print("Balanced Accuracy:", round(balanced_acc * 100, 2), "%")
    print("Correct:", correct, "/", total)
    print("Chance level:", round(chance_accuracy * 100, 2), "%")
    print(
        "Improvement over chance:",
        round(improvement * 100, 2),
        "percentage points"
    )


    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    report_path = os.path.join(
        RESULTS_DIR,
        f"{subject}_classification_report.csv"
    )

    report_df.to_csv(report_path)


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=sorted(y.unique())
    )

    cm_df = pd.DataFrame(
        cm,
        index=sorted(y.unique()),
        columns=sorted(y.unique())
    )

    cm_path = os.path.join(
        RESULTS_DIR,
        f"{subject}_confusion_matrix.csv"
    )

    cm_df.to_csv(cm_path)


    # --------------------------------------------------------
    # SAVE PREDICTIONS
    # --------------------------------------------------------

    prediction_df = subject_df.loc[idx_test, [
        "subject_id",
        "trial_id",
        "label_name",
        "label"
    ]].copy()

    prediction_df["predicted_label"] = y_pred

    label_to_name = (
        subject_df[["label", "label_name"]]
        .drop_duplicates()
        .set_index("label")["label_name"]
        .to_dict()
    )

    prediction_df["predicted_label_name"] = (
        prediction_df["predicted_label"]
        .map(label_to_name)
    )

    prediction_df["correct"] = (
        prediction_df["label"] ==
        prediction_df["predicted_label"]
    )

    all_predictions.append(prediction_df)


    # --------------------------------------------------------
    # STORE SUBJECT RESULT
    # --------------------------------------------------------

    subject_results.append({
        "subject_id": subject,
        "total_samples": len(subject_df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "num_classes": y.nunique(),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_acc,
        "correct_predictions": correct,
        "chance_accuracy": chance_accuracy,
        "improvement_over_chance": improvement
    })


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(subject_results)

results_path = os.path.join(
    RESULTS_DIR,
    "subject_wise_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# ============================================================
# SAVE ALL PREDICTIONS
# ============================================================

predictions_df = pd.concat(
    all_predictions,
    ignore_index=True
)

predictions_path = os.path.join(
    RESULTS_DIR,
    "subject_wise_predictions.csv"
)

predictions_df.to_csv(
    predictions_path,
    index=False
)


# ============================================================
# OVERALL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("FINAL SUBJECT-WISE SUMMARY")
print("=" * 70)

print(
    results_df[
        [
            "subject_id",
            "accuracy",
            "macro_f1",
            "balanced_accuracy",
            "correct_predictions",
            "test_samples"
        ]
    ].to_string(index=False)
)


mean_accuracy = results_df["accuracy"].mean()
std_accuracy = results_df["accuracy"].std()

mean_f1 = results_df["macro_f1"].mean()
std_f1 = results_df["macro_f1"].std()

mean_balanced = results_df["balanced_accuracy"].mean()
std_balanced = results_df["balanced_accuracy"].std()

chance = 1 / results_df["num_classes"].iloc[0]


print("\n")
print("=" * 70)
print("AGGREGATE RESULTS")
print("=" * 70)

print(
    f"Mean Accuracy: "
    f"{mean_accuracy * 100:.2f}% ± {std_accuracy * 100:.2f}%"
)

print(
    f"Mean Macro F1: "
    f"{mean_f1 * 100:.2f}% ± {std_f1 * 100:.2f}%"
)

print(
    f"Mean Balanced Accuracy: "
    f"{mean_balanced * 100:.2f}% ± {std_balanced * 100:.2f}%"
)

print(
    f"Chance Level: "
    f"{chance * 100:.2f}%"
)

print(
    f"Mean Improvement Over Chance: "
    f"{(mean_accuracy - chance) * 100:.2f} percentage points"
)


# ============================================================
# BEST / WORST SUBJECT
# ============================================================

best_subject = results_df.loc[
    results_df["accuracy"].idxmax()
]

worst_subject = results_df.loc[
    results_df["accuracy"].idxmin()
]

print("\n")
print("=" * 70)
print("BEST SUBJECT")
print("=" * 70)

print(
    best_subject[
        [
            "subject_id",
            "accuracy",
            "macro_f1",
            "balanced_accuracy"
        ]
    ]
)


print("\n")
print("=" * 70)
print("WORST SUBJECT")
print("=" * 70)

print(
    worst_subject[
        [
            "subject_id",
            "accuracy",
            "macro_f1",
            "balanced_accuracy"
        ]
    ]
)


# ============================================================
# OUTPUT FILES
# ============================================================

print("\n")
print("=" * 70)
print("OUTPUT FILES")
print("=" * 70)

print(results_path)
print(predictions_path)

print("\nIndividual subject reports:")
for subject in subjects:
    print(
        os.path.join(
            RESULTS_DIR,
            f"{subject}_classification_report.csv"
        )
    )

print("\nExperiment completed successfully!")