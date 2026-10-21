
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

PROJECT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT / "experiments" / "ds004022_experiment114_results"
OUT = PROJECT / "experiments" / "ds004022_experiment115_results"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["reaching", "grasping", "lifting", "twisting"]


def find_column(df, candidates, description):
    for name in candidates:
        if name in df.columns:
            return name
    raise ValueError(
        f"Cannot find {description}. Available columns: {list(df.columns)}"
    )


def main():
    pred_path = SOURCE / "trial_level_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing file: {pred_path}")

    df = pd.read_csv(pred_path)

    subject_col = find_column(
        df, ["subject_from_epochs", "subject", "subject_id"], "subject column"
    )
    run_col = find_column(
        df, ["run_from_epochs", "run", "run_id"], "run column"
    )
    true_col = find_column(
        df, ["true_class", "true_label", "class_name"], "true-class column"
    )
    pred_col = find_column(
        df, ["predicted_class", "predicted_label", "prediction"],
        "predicted-class column"
    )

    df[subject_col] = df[subject_col].astype(str).str.strip()
    df[run_col] = df[run_col].astype(str).str.strip()
    df[true_col] = df[true_col].astype(str).str.strip().str.lower()
    df[pred_col] = df[pred_col].astype(str).str.strip().str.lower()

    unknown_true = sorted(set(df[true_col]) - set(CLASSES))
    unknown_pred = sorted(set(df[pred_col]) - set(CLASSES))
    if unknown_true or unknown_pred:
        raise ValueError(
            f"Unexpected labels. True: {unknown_true}; predicted: {unknown_pred}"
        )

    df["correct"] = df[true_col] == df[pred_col]
    df["error_type"] = np.where(df["correct"], "correct", "misclassified")

    # 1. Overall summary
    overall = {
        "experiment": 115,
        "source_experiment": 114,
        "trials": int(len(df)),
        "subjects": int(df[subject_col].nunique()),
        "runs": int(df[[subject_col, run_col]].drop_duplicates().shape[0]),
        "correct": int(df["correct"].sum()),
        "incorrect": int((~df["correct"]).sum()),
        "accuracy": float(df["correct"].mean()),
    }

    # 2. Performance by held-out subject
    subject_rows = []
    for subject, part in df.groupby(subject_col, sort=True):
        subject_rows.append({
            "subject": subject,
            "trials": int(len(part)),
            "correct": int(part["correct"].sum()),
            "incorrect": int((~part["correct"]).sum()),
            "accuracy": float(part["correct"].mean()),
        })
    subject_metrics = pd.DataFrame(subject_rows)
    subject_metrics.to_csv(OUT / "subject_error_analysis.csv", index=False)

    # 3. Performance by subject and run
    run_rows = []
    for (subject, run), part in df.groupby([subject_col, run_col], sort=True):
        run_rows.append({
            "subject": subject,
            "run": run,
            "trials": int(len(part)),
            "correct": int(part["correct"].sum()),
            "incorrect": int((~part["correct"]).sum()),
            "accuracy": float(part["correct"].mean()),
        })
    run_metrics = pd.DataFrame(run_rows)
    run_metrics.to_csv(OUT / "run_error_analysis.csv", index=False)

    # 4. True class vs predicted class, overall and per subject
    overall_cm = confusion_matrix(
        df[true_col], df[pred_col], labels=CLASSES
    )
    cm_df = pd.DataFrame(
        overall_cm,
        index=[f"true_{c}" for c in CLASSES],
        columns=[f"pred_{c}" for c in CLASSES],
    )
    cm_df.to_csv(OUT / "overall_confusion_matrix.csv")

    cm_rows = []
    for subject, part in df.groupby(subject_col, sort=True):
        cm = confusion_matrix(part[true_col], part[pred_col], labels=CLASSES)
        for i, true_class in enumerate(CLASSES):
            for j, predicted_class in enumerate(CLASSES):
                cm_rows.append({
                    "subject": subject,
                    "true_class": true_class,
                    "predicted_class": predicted_class,
                    "count": int(cm[i, j]),
                })
    pd.DataFrame(cm_rows).to_csv(
        OUT / "subject_confusion_matrices.csv", index=False
    )

    # 5. Most frequent misclassification directions
    mistakes = df.loc[~df["correct"]].copy()
    pairs = (
        mistakes.groupby([true_col, pred_col])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .rename(columns={
            true_col: "true_class",
            pred_col: "predicted_class",
        })
    )
    pairs.to_csv(OUT / "most_common_confusions.csv", index=False)

    # 6. Class-level recall and prediction counts
    class_rows = []
    for class_name in CLASSES:
        true_mask = df[true_col] == class_name
        pred_mask = df[pred_col] == class_name
        tp = int((true_mask & pred_mask).sum())
        true_count = int(true_mask.sum())
        pred_count = int(pred_mask.sum())
        class_rows.append({
            "class": class_name,
            "true_trials": true_count,
            "predicted_trials": pred_count,
            "correct": tp,
            "recall": tp / true_count if true_count else 0.0,
            "precision": tp / pred_count if pred_count else 0.0,
        })
    class_metrics = pd.DataFrame(class_rows)
    class_metrics.to_csv(OUT / "class_error_analysis.csv", index=False)

    # 7. Check whether trial counts are balanced by class and subject-run
    counts = df.groupby([subject_col, run_col, true_col]).size()
    count_report = {
        "minimum_trials_per_subject_run_class": int(counts.min()),
        "maximum_trials_per_subject_run_class": int(counts.max()),
        "all_subject_run_class_counts_equal": bool(counts.nunique() == 1),
    }

    # 8. Identify runs that warrant review. These are descriptive flags,
    # not evidence of a faulty recording.
    mean_run_accuracy = float(run_metrics["accuracy"].mean())
    run_metrics["below_mean_run_accuracy"] = (
        run_metrics["accuracy"] < mean_run_accuracy
    )
    run_metrics.to_csv(OUT / "run_error_analysis.csv", index=False)

    overall["mean_run_accuracy"] = mean_run_accuracy
    overall["run_count_audit"] = count_report
    overall["most_common_confusions"] = pairs.head(8).to_dict(orient="records")
    overall["lowest_accuracy_runs"] = (
        run_metrics.sort_values("accuracy")
        .head(5)[["subject", "run", "trials", "accuracy"]]
        .to_dict(orient="records")
    )

    with open(OUT / "experiment115_summary.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)

    print("\nEXPERIMENT 115 COMPLETE")
    print("=" * 60)
    print("\nOverall summary:")
    print(json.dumps(overall, indent=2))

    print("\nSubject errors:")
    print(subject_metrics.to_string(index=False))

    print("\nRun errors (lowest accuracy first):")
    print(
        run_metrics.sort_values("accuracy").to_string(index=False)
    )

    print("\nClass errors:")
    print(class_metrics.to_string(index=False))

    print("\nMost common misclassification directions:")
    print(pairs.head(10).to_string(index=False))

    print("\nOverall confusion matrix:")
    print(cm_df.to_string())

    print("\nSaved to:", OUT)


if __name__ == "__main__":
    main()
