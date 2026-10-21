
"""
Experiment 143: Final Results Consolidation — Dataset 3

Reads existing experiment outputs only.
Does not retrain models or modify previous experiment outputs.

Produces:
    final_model_comparison.csv
    prediction_audit.csv
    final_results_report.txt
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
OUT = BASE / "experiment143_final_consolidation"

OUT.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = {
    130: BASE,
    131: BASE / "experiment131_csp_lda",
    134: BASE / "experiment134_corrected_features",
    139: BASE / "experiment139_preprocessing",
    140: BASE / "experiment140_csp_svm",
}

PREDICTION_NAMES = [
    "out_of_fold_predictions.csv",
    "corrected_out_of_fold_predictions.csv",
]

SUMMARY_NAMES = [
    "corrected_feature_summary.json",
    "preprocessing_summary.json",
    "csp_svm_summary.json",
    "summary.json",
]


def find_file(folder, names):
    for name in names:
        candidate = folder / name
        if candidate.exists():
            return candidate
    return None


def read_json(path):
    if path is None:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        warnings.warn(f"Cannot read JSON {path}: {exc}")
        return None


def flatten_json(obj, prefix=""):
    """Flatten nested JSON keys to help locate metrics safely."""
    result = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                result.update(flatten_json(value, name))
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                if np.isfinite(value):
                    result[name] = float(value)
    return result


def find_metric(flat, candidates):
    """Find a numeric metric by exact or suffix key."""
    for candidate in candidates:
        for key, value in flat.items():
            if key.lower() == candidate.lower():
                return value

    for candidate in candidates:
        matches = [
            value for key, value in flat.items()
            if key.lower().endswith("." + candidate.lower())
        ]
        if matches:
            return matches[0]

    return np.nan


def configuration_columns(df):
    return [
        c for c in [
            "feature_set",
            "feature_name",
            "feature_type",
            "model",
            "model_name",
            "classifier",
            "configuration",
            "config",
        ]
        if c in df.columns
    ]


def audit_predictions(exp_id, path):
    df = pd.read_csv(path)

    if not {"y_true", "y_pred"}.issubset(df.columns):
        return [], [{
            "experiment": exp_id,
            "file": str(path),
            "status": "Missing y_true/y_pred",
        }]

    config_cols = configuration_columns(df)

    if config_cols:
        grouped = df.groupby(config_cols, dropna=False, sort=False)
    else:
        grouped = [("single_configuration", df)]

    rows = []
    audits = []

    for config, part in grouped:
        if not isinstance(config, tuple):
            config = (config,)

        config_label = (
            " | ".join(
                f"{col}={val}"
                for col, val in zip(config_cols, config)
            )
            if config_cols else "single_configuration"
        )

        part = part.dropna(subset=["y_true", "y_pred"]).copy()

        if part.empty:
            continue

        y_true = part["y_true"].astype(int).to_numpy()
        y_pred = part["y_pred"].astype(int).to_numpy()

        row = {
            "experiment": exp_id,
            "configuration": config_label,
            "prediction_rows": len(part),
            "pooled_accuracy": accuracy_score(y_true, y_pred),
            "pooled_balanced_accuracy": balanced_accuracy_score(
                y_true, y_pred
            ),
            "pooled_macro_f1": f1_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "confusion_matrix_0_1": json.dumps(
                confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
            ),
            "prediction_file": str(path),
        }

        if "participant" in part.columns:
            row["n_participants"] = int(part["participant"].nunique())

            subject_acc = []
            subject_f1 = []

            for _, subject in part.groupby("participant"):
                sy = subject["y_true"].astype(int).to_numpy()
                sp = subject["y_pred"].astype(int).to_numpy()

                subject_acc.append(accuracy_score(sy, sp))
                subject_f1.append(
                    f1_score(sy, sp, average="macro", zero_division=0)
                )

            row["mean_participant_accuracy"] = float(np.mean(subject_acc))
            row["mean_participant_macro_f1"] = float(np.mean(subject_f1))

            if "group" in part.columns:
                row["groups"] = ";".join(
                    sorted(part["group"].astype(str).unique())
                )

        rows.append(row)

        audits.append({
            "experiment": exp_id,
            "configuration": config_label,
            "file": str(path),
            "status": "READ",
            "prediction_rows": int(len(part)),
            "n_participants": (
                int(part["participant"].nunique())
                if "participant" in part.columns else np.nan
            ),
            "has_participant_column": "participant" in part.columns,
            "has_trial_id_column": any(
                c.lower() in {
                    "trial_id", "trial", "epoch_id", "epoch",
                    "sample_id", "event_id"
                }
                for c in part.columns
            ),
        })

    return rows, audits


def main():
    comparison_rows = []
    prediction_rows = []
    audit_rows = []
    file_rows = []

    print("=" * 72)
    print("EXPERIMENT 143 — FINAL RESULTS CONSOLIDATION")
    print("=" * 72)
    print("Reading saved outputs only. No model training will occur.")

    for exp_id, folder in EXPERIMENTS.items():
        print(f"\nInspecting Experiment {exp_id}: {folder}")

        pred_path = find_file(folder, PREDICTION_NAMES)
        summary_path = find_file(folder, SUMMARY_NAMES)
        summary = read_json(summary_path)
        flat = flatten_json(summary) if summary else {}

        file_rows.append({
            "experiment": exp_id,
            "folder_exists": folder.exists(),
            "summary_path": str(summary_path) if summary_path else "",
            "prediction_path": str(pred_path) if pred_path else "",
            "summary_readable": summary is not None,
            "prediction_file_found": pred_path is not None,
        })

        comparison = {
            "experiment": exp_id,
            "summary_file": str(summary_path) if summary_path else "",
            "prediction_file": str(pred_path) if pred_path else "",
            "summary_accuracy": find_metric(
                flat, [
                    "pooled_accuracy", "accuracy",
                    "pooled_accuracy_mean",
                ]
            ),
            "summary_balanced_accuracy": find_metric(
                flat, ["pooled_balanced_accuracy", "balanced_accuracy"]
            ),
            "summary_macro_f1": find_metric(
                flat, ["pooled_macro_f1", "macro_f1"]
            ),
            "summary_mean_participant_accuracy": find_metric(
                flat, [
                    "mean_participant_accuracy",
                    "mean_subject_accuracy",
                ]
            ),
            "summary_mean_participant_macro_f1": find_metric(
                flat, [
                    "mean_participant_macro_f1",
                    "mean_subject_macro_f1",
                ]
            ),
            "summary_n_participants": find_metric(
                flat, ["n_participants", "n_subjects", "participant_count"]
            ),
            "summary_n_trials": find_metric(
                flat, ["n_trials", "total_trials", "n_epochs"]
            ),
        }

        if pred_path:
            try:
                metrics, audits = audit_predictions(exp_id, pred_path)
                prediction_rows.extend(metrics)
                audit_rows.extend(audits)
            except Exception as exc:
                audit_rows.append({
                    "experiment": exp_id,
                    "file": str(pred_path),
                    "status": f"ERROR: {exc}",
                })

        comparison_rows.append(comparison)

    comparison_df = pd.DataFrame(comparison_rows)
    predictions_df = pd.DataFrame(prediction_rows)
    audits_df = pd.DataFrame(audit_rows)
    files_df = pd.DataFrame(file_rows)

    # Merge recomputed metrics only when a single configuration exists.
    # Never choose the best configuration silently.
    if not predictions_df.empty:
        for exp_id in predictions_df["experiment"].unique():
            exp_rows = predictions_df[
                predictions_df["experiment"] == exp_id
            ]

            if len(exp_rows) == 1:
                idx = comparison_df.index[
                    comparison_df["experiment"] == exp_id
                ]
                if len(idx) == 1:
                    i = idx[0]
                    r = exp_rows.iloc[0]
                    comparison_df.loc[i, "recomputed_accuracy"] = (
                        r["pooled_accuracy"]
                    )
                    comparison_df.loc[i, "recomputed_balanced_accuracy"] = (
                        r["pooled_balanced_accuracy"]
                    )
                    comparison_df.loc[i, "recomputed_macro_f1"] = (
                        r["pooled_macro_f1"]
                    )
                    comparison_df.loc[i, "recomputed_n_participants"] = (
                        r.get("n_participants", np.nan)
                    )
                    comparison_df.loc[i, "recomputed_n_trials"] = (
                        r["prediction_rows"]
                    )
                    comparison_df.loc[i, "recomputed_configuration"] = (
                        r["configuration"]
                    )

    comparison_path = OUT / "final_model_comparison.csv"
    predictions_path = OUT / "prediction_metrics_recomputed.csv"
    audit_path = OUT / "prediction_audit.csv"
    files_path = OUT / "inspected_files.csv"
    report_path = OUT / "final_results_report.txt"

    comparison_df.to_csv(comparison_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    audits_df.to_csv(audit_path, index=False)
    files_df.to_csv(files_path, index=False)

    report = [
        "DATASET 3 — FINAL RESULTS CONSOLIDATION",
        "=" * 72,
        "",
        "No models were retrained.",
        "Original datasets and existing experiment outputs were not modified.",
        "",
        "SUMMARY METRICS FROM SAVED JSON FILES",
        "-" * 72,
        comparison_df.to_string(index=False),
        "",
        "METRICS RECOMPUTED FROM SAVED PREDICTIONS",
        "-" * 72,
        (
            predictions_df.to_string(index=False)
            if not predictions_df.empty
            else "No compatible prediction files found."
        ),
        "",
        "IMPORTANT INTERPRETATION NOTES",
        "-" * 72,
        "1. Missing metrics are not zero; they may use unrecognized JSON keys.",
        "2. Multi-configuration prediction files are reported separately.",
        "3. Comparisons are valid only when task, labels, preprocessing, and",
        "   evaluation protocol are compatible.",
        "4. Prediction row counts alone do not prove exact trial coverage.",
        "5. Prior model selection reused this participant cohort; results are",
        "   exploratory until validated on an independent cohort.",
        "",
        "OUTPUT FILES",
        "-" * 72,
        str(comparison_path),
        str(predictions_path),
        str(audit_path),
        str(files_path),
    ]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("\n" + "=" * 72)
    print("CONSOLIDATION COMPLETE")
    print("=" * 72)
    print(f"Experiments inspected: {len(comparison_df)}")
    print(f"Prediction configurations: {len(predictions_df)}")
    print("\nGenerated files:")
    print(f"  {comparison_path}")
    print(f"  {predictions_path}")
    print(f"  {audit_path}")
    print(f"  {files_path}")
    print(f"  {report_path}")

    print("\nFinal comparison preview:")
    preview_cols = [
        c for c in [
            "experiment",
            "summary_accuracy",
            "summary_balanced_accuracy",
            "summary_macro_f1",
            "recomputed_accuracy",
            "recomputed_balanced_accuracy",
            "recomputed_macro_f1",
        ]
        if c in comparison_df.columns
    ]
    print(comparison_df[preview_cols].to_string(index=False))


if __name__ == "__main__":
    main()
