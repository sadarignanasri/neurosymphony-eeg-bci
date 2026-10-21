
"""
Experiment 142: Final Results Audit — Dataset 3 (28955456)

Purpose:
    - Audit existing experiment outputs without retraining.
    - Compare saved metrics across experiments.
    - Verify OOF prediction coverage and recompute metrics when possible.
    - Save a consolidated report in a new output directory.

This script does not modify source datasets or existing experiment outputs.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)

ROOT = Path(r"D:\neurosymphony-eeg-bci")

BASE = ROOT / "experiments" / "dataset3_28955456_training"

OUTPUT_DIR = BASE / "experiment142_final_audit"

EXPERIMENTS = {
    130: BASE,
    131: BASE / "experiment131_csp_lda",
    132: BASE / "experiment132_time_windows",
    134: BASE / "experiment134_corrected_features",
    135: BASE / "experiment135_validation",
    136: BASE / "experiment136_significance",
    137: BASE / "experiment137_group_comparison",
    138: BASE / "experiment138_error_analysis",
    139: BASE / "experiment139_preprocessing",
    140: BASE / "experiment140_csp_svm",
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path):
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        warnings.warn(f"Could not read {path}: {exc}")
        return None


def first_existing(folder, filenames):
    for filename in filenames:
        path = folder / filename
        if path.exists():
            return path
    return None


def metric_value(data, keys):
    """Return the first available metric key from a JSON dictionary."""
    if not isinstance(data, dict):
        return np.nan

    for key in keys:
        value = data.get(key)
        if isinstance(value, (int, float)) and np.isfinite(value):
            return float(value)

    return np.nan


def find_prediction_file(folder):
    return first_existing(
        folder,
        [
            "out_of_fold_predictions.csv",
            "corrected_out_of_fold_predictions.csv",
        ],
    )


def find_summary_file(folder):
    return first_existing(
        folder,
        [
            "corrected_feature_summary.json",
            "preprocessing_summary.json",
            "csp_svm_summary.json",
            "summary.json",
            "validation_summary.json",
            "significance_summary.json",
        ],
    )


def audit_predictions(exp_id, path):
    """
    Recompute metrics from saved predictions.

    Prediction files can contain several model configurations.
    If model/configuration columns exist, audit each configuration
    separately instead of combining their predictions.
    """
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return [], [{
            "experiment": exp_id,
            "file": str(path),
            "status": f"READ_ERROR: {exc}",
        }]

    required = {"y_true", "y_pred"}

    if not required.issubset(df.columns):
        return [], [{
            "experiment": exp_id,
            "file": str(path),
            "status": "MISSING y_true/y_pred columns",
        }]

    # Identify configuration columns to avoid mixing different models.
    possible_config_cols = [
        col for col in [
            "feature_set",
            "feature_name",
            "feature_type",
            "model",
            "model_name",
            "classifier",
            "configuration",
            "config",
        ]
        if col in df.columns
    ]

    grouping_cols = possible_config_cols

    if grouping_cols:
        groups = df.groupby(
            grouping_cols,
            dropna=False,
            sort=False,
        )
    else:
        groups = [(("single_configuration",), df)]

    metric_rows = []
    coverage_rows = []

    for config, part in groups:
        if not isinstance(config, tuple):
            config = (config,)

        config_label = "; ".join(
            f"{col}={value}"
            for col, value in zip(
                grouping_cols or ["configuration"],
                config,
            )
        )

        part = part.dropna(subset=["y_true", "y_pred"]).copy()

        if part.empty:
            coverage_rows.append({
                "experiment": exp_id,
                "configuration": config_label,
                "file": str(path),
                "status": "NO VALID PREDICTIONS",
            })
            continue

        y_true = part["y_true"].astype(int).to_numpy()
        y_pred = part["y_pred"].astype(int).to_numpy()

        pooled_acc = accuracy_score(y_true, y_pred)
        pooled_bal_acc = balanced_accuracy_score(y_true, y_pred)
        pooled_f1 = f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )

        row = {
            "experiment": exp_id,
            "configuration": config_label,
            "prediction_rows": int(len(part)),
            "pooled_accuracy": float(pooled_acc),
            "pooled_balanced_accuracy": float(pooled_bal_acc),
            "pooled_macro_f1": float(pooled_f1),
            "file": str(path),
        }

        if "participant" in part.columns:
            participant_counts = part["participant"].value_counts()
            unique_participants = part["participant"].nunique()

            row["n_participants"] = int(unique_participants)
            row["min_trials_per_participant"] = int(
                participant_counts.min()
            )
            row["max_trials_per_participant"] = int(
                participant_counts.max()
            )

            participant_acc = []
            participant_f1 = []

            for _, subject_df in part.groupby("participant"):
                sy = subject_df["y_true"].astype(int).to_numpy()
                sp = subject_df["y_pred"].astype(int).to_numpy()

                participant_acc.append(accuracy_score(sy, sp))
                participant_f1.append(
                    f1_score(
                        sy,
                        sp,
                        average="macro",
                        zero_division=0,
                    )
                )

            row["mean_participant_accuracy"] = float(
                np.mean(participant_acc)
            )
            row["mean_participant_macro_f1"] = float(
                np.mean(participant_f1)
            )

            if "correct" in part.columns:
                stored_correct = (
                    part["correct"].astype(str).str.lower()
                    .isin(["true", "1", "yes"])
                )
                recomputed_correct = y_true == y_pred

                row["correct_column_matches"] = bool(
                    np.array_equal(
                        stored_correct.to_numpy(),
                        recomputed_correct,
                    )
                )

            if "group" in part.columns:
                row["group_counts"] = json.dumps(
                    part.groupby("group")["participant"]
                    .nunique()
                    .to_dict()
                )

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        row["confusion_matrix"] = json.dumps(cm.tolist())
        metric_rows.append(row)

        # A prediction file cannot prove every expected trial is present
        # without an independent trial inventory. Report observed coverage.
        coverage_rows.append({
            "experiment": exp_id,
            "configuration": config_label,
            "file": str(path),
            "status": "READ",
            "prediction_rows": int(len(part)),
            "unique_participants": (
                int(part["participant"].nunique())
                if "participant" in part.columns
                else np.nan
            ),
            "duplicate_participant_true_label_rows": (
                int(
                    part.duplicated(
                        subset=["participant", "y_true"],
                        keep=False,
                    ).sum()
                )
                if "participant" in part.columns
                else np.nan
            ),
            "note": (
                "Duplicate participant/label combinations are only a "
                "warning signal, not proof of duplicate trials."
            ),
        })

    return metric_rows, coverage_rows


def main():
    summary_rows = []
    prediction_metric_rows = []
    coverage_rows = []
    missing_rows = []

    print("=" * 72)
    print("EXPERIMENT 142 — FINAL RESULTS AUDIT")
    print("=" * 72)
    print("No models will be retrained.")
    print("Original datasets and previous experiment outputs remain unchanged.")

    for exp_id, folder in EXPERIMENTS.items():
        if not folder.exists():
            missing_rows.append({
                "experiment": exp_id,
                "expected_folder": str(folder),
                "status": "FOLDER_NOT_FOUND",
            })
            continue

        summary_path = find_summary_file(folder)
        summary = load_json(summary_path) if summary_path else None

        row = {
            "experiment": exp_id,
            "folder": str(folder),
            "summary_file": (
                str(summary_path) if summary_path else ""
            ),
            "pooled_accuracy": metric_value(
                summary,
                [
                    "pooled_accuracy",
                    "accuracy",
                    "pooled_accuracy_mean",
                ],
            ),
            "pooled_balanced_accuracy": metric_value(
                summary,
                [
                    "pooled_balanced_accuracy",
                    "balanced_accuracy",
                ],
            ),
            "pooled_macro_f1": metric_value(
                summary,
                [
                    "pooled_macro_f1",
                    "macro_f1",
                ],
            ),
            "mean_participant_accuracy": metric_value(
                summary,
                [
                    "mean_participant_accuracy",
                    "mean_subject_accuracy",
                ],
            ),
            "mean_participant_macro_f1": metric_value(
                summary,
                [
                    "mean_participant_macro_f1",
                    "mean_subject_macro_f1",
                ],
            ),
            "n_participants": metric_value(
                summary,
                [
                    "n_participants",
                    "n_subjects",
                    "participant_count",
                ],
            ),
            "n_trials": metric_value(
                summary,
                [
                    "n_trials",
                    "total_trials",
                    "n_epochs",
                ],
            ),
        }

        pred_path = find_prediction_file(folder)

        if pred_path:
            pred_metrics, pred_coverage = audit_predictions(
                exp_id, pred_path
            )
            prediction_metric_rows.extend(pred_metrics)
            coverage_rows.extend(pred_coverage)

            row["prediction_file"] = str(pred_path)
        else:
            row["prediction_file"] = ""

        if summary is None and pred_path is None:
            row["status"] = "NO_READABLE_SUMMARY_OR_PREDICTIONS"
        elif summary is None:
            row["status"] = "PREDICTIONS_ONLY"
        elif pred_path is None:
            row["status"] = "SUMMARY_ONLY"
        else:
            row["status"] = "SUMMARY_AND_PREDICTIONS"

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    prediction_df = pd.DataFrame(prediction_metric_rows)
    coverage_df = pd.DataFrame(coverage_rows)
    missing_df = pd.DataFrame(missing_rows)

    summary_csv = OUTPUT_DIR / "experiment_summary_audit.csv"
    prediction_csv = OUTPUT_DIR / "prediction_metrics_recomputed.csv"
    coverage_csv = OUTPUT_DIR / "prediction_coverage_audit.csv"
    missing_csv = OUTPUT_DIR / "missing_outputs.csv"
    report_txt = OUTPUT_DIR / "final_audit_report.txt"
    report_json = OUTPUT_DIR / "final_audit_summary.json"

    summary_df.to_csv(summary_csv, index=False)
    prediction_df.to_csv(prediction_csv, index=False)
    coverage_df.to_csv(coverage_csv, index=False)
    missing_df.to_csv(missing_csv, index=False)

    report = []
    report.append("DATASET 3 — FINAL RESULTS AUDIT")
    report.append("=" * 72)
    report.append("")
    report.append("This audit reads existing outputs only; it does not retrain models.")
    report.append("")

    report.append("EXPERIMENT SUMMARY")
    report.append("-" * 72)

    if summary_df.empty:
        report.append("No experiment folders were successfully inspected.")
    else:
        cols = [
            col for col in [
                "experiment",
                "pooled_accuracy",
                "pooled_balanced_accuracy",
                "pooled_macro_f1",
                "mean_participant_accuracy",
                "mean_participant_macro_f1",
                "n_participants",
                "n_trials",
                "status",
            ]
            if col in summary_df.columns
        ]

        report.append(
            summary_df[cols].to_string(index=False)
        )

    report.append("")
    report.append("RECOMPUTED METRICS FROM SAVED PREDICTIONS")
    report.append("-" * 72)

    if prediction_df.empty:
        report.append(
            "No compatible saved out-of-fold prediction files were found."
        )
    else:
        preferred = [
            col for col in [
                "experiment",
                "configuration",
                "prediction_rows",
                "pooled_accuracy",
                "pooled_balanced_accuracy",
                "pooled_macro_f1",
                "mean_participant_accuracy",
                "mean_participant_macro_f1",
                "n_participants",
            ]
            if col in prediction_df.columns
        ]
        report.append(
            prediction_df[preferred].to_string(index=False)
        )

    report.append("")
    report.append("INTERPRETATION NOTES")
    report.append("-" * 72)
    report.append(
        "1. Metrics recomputed from predictions are distinct from metrics "
        "read from summary JSON files."
    )
    report.append(
        "2. Multiple configurations in one prediction file are audited "
        "separately when configuration columns are present."
    )
    report.append(
        "3. A saved prediction file alone cannot prove complete trial "
        "coverage without an independent expected-trial inventory."
    )
    report.append(
        "4. Participant-level validation does not replace validation on "
        "an independent participant cohort."
    )
    report.append(
        "5. Results from earlier experiments should only be compared "
        "directly when task, labels, folds, and metric definitions match."
    )

    if not missing_df.empty:
        report.append("")
        report.append("MISSING EXPERIMENT OUTPUTS")
        report.append("-" * 72)
        report.append(missing_df.to_string(index=False))

    report_text = "\n".join(report)

    with open(report_txt, "w", encoding="utf-8") as f:
        f.write(report_text)

    audit_summary = {
        "experiment": 142,
        "purpose": "final_results_audit",
        "retraining_performed": False,
        "experiment_folders_inspected": int(len(summary_df)),
        "prediction_configurations_audited": int(len(prediction_df)),
        "missing_or_unreadable_folders": int(len(missing_df)),
        "output_directory": str(OUTPUT_DIR),
        "outputs": [
            str(summary_csv),
            str(prediction_csv),
            str(coverage_csv),
            str(missing_csv),
            str(report_txt),
        ],
        "limitations": [
            "Saved files may not contain every expected metric.",
            "Prediction completeness requires an independent trial inventory.",
            "Previous model selection reused the same participant cohort.",
        ],
    }

    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    print("\n" + "=" * 72)
    print("AUDIT COMPLETE")
    print("=" * 72)
    print(f"Experiment folders inspected: {len(summary_df)}")
    print(f"Prediction configurations audited: {len(prediction_df)}")
    print(f"Missing/unreadable folders: {len(missing_df)}")

    print("\nOutput files:")
    print(f"  {summary_csv}")
    print(f"  {prediction_csv}")
    print(f"  {coverage_csv}")
    print(f"  {missing_csv}")
    print(f"  {report_txt}")
    print(f"  {report_json}")

    print("\nAudit report preview:\n")
    print(report_text)


if __name__ == "__main__":
    main()
