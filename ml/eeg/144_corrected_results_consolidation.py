
"""
Experiment 144: Corrected Final Results Consolidation — Dataset 3.

Reads existing experiment summaries and predictions.
Does not retrain models or modify original data / previous outputs.

Correctly handles:
    - JSON summaries with nested "results" lists
    - true_label / predicted_label columns
    - y_true / y_pred columns
    - subject_id / participant columns
    - Multiple feature and preprocessing configurations

Outputs are written to a new Experiment 144 folder.
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
OUT = BASE / "experiment144_corrected_consolidation"
OUT.mkdir(parents=True, exist_ok=True)

# Paths are based on the experiment output locations established so far.
EXPERIMENTS = {
    130: BASE,
    131: BASE / "experiment131_csp_lda",
    134: BASE / "experiment134_corrected_features",
    139: BASE / "experiment139_preprocessing",
    140: BASE / "experiment140_csp_svm",
}

SUMMARY_NAMES = [
    "corrected_feature_summary.json",
    "preprocessing_summary.json",
    "csp_svm_summary.json",
    "summary.json",
    "validation_summary.json",
    "significance_summary.json",
]

PREDICTION_NAMES = [
    "out_of_fold_predictions.csv",
    "corrected_out_of_fold_predictions.csv",
]

TRUE_COLS = ["true_label", "y_true", "y", "actual_label"]
PRED_COLS = ["predicted_label", "y_pred", "prediction", "pred_label"]
SUBJECT_COLS = ["participant", "subject_id", "subject", "participant_id"]
GROUP_COLS = ["feature_family", "model", "preprocessing", "configuration", "config"]


def find_file(folder, filenames):
    for filename in filenames:
        path = folder / filename
        if path.is_file():
            return path
    return None


def first_column(df, candidates):
    for name in candidates:
        if name in df.columns:
            return name
    return None


def flatten_numeric_json(obj, prefix=""):
    """Flatten numeric JSON fields without discarding nested result lists."""
    output = {}

    if isinstance(obj, dict):
        for key, value in obj.items():
            name = f"{prefix}.{key}" if prefix else str(key)

            if isinstance(value, dict):
                output.update(flatten_numeric_json(value, name))
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                if np.isfinite(value):
                    output[name] = float(value)

    return output


def read_json(path):
    if path is None:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        warnings.warn(f"Could not read {path}: {exc}")
        return None


def read_summary_records(exp_id, folder):
    """Read top-level metrics and each record inside a results list."""
    summary_path = find_file(folder, SUMMARY_NAMES)
    summary = read_json(summary_path)

    if summary is None:
        return [], summary_path

    records = summary.get("results", [])
    rows = []

    if isinstance(records, list) and records:
        for item in records:
            if not isinstance(item, dict):
                continue

            row = {
                "experiment": exp_id,
                "source": "summary_json",
                "summary_file": str(summary_path),
            }

            # Keep identifying configuration fields.
            for key in [
                "feature_family",
                "model",
                "preprocessing",
                "configuration",
            ]:
                if key in item:
                    row[key] = item[key]

            # Preserve metrics as they appear in the source JSON.
            for key, value in item.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    row[key] = value

            row["n_participants"] = item.get(
                "participants",
                item.get("n_participants", np.nan),
            )

            rows.append(row)

    else:
        flat = flatten_numeric_json(summary)
        row = {
            "experiment": exp_id,
            "source": "summary_json",
            "summary_file": str(summary_path),
        }

        for key, value in flat.items():
            row[key] = value

        rows.append(row)

    return rows, summary_path


def metric_row_from_predictions(exp_id, path):
    df = pd.read_csv(path)

    true_col = first_column(df, TRUE_COLS)
    pred_col = first_column(df, PRED_COLS)
    subject_col = first_column(df, SUBJECT_COLS)

    if true_col is None or pred_col is None:
        return [], [{
            "experiment": exp_id,
            "file": str(path),
            "status": "Missing recognized true/predicted label columns",
            "columns_found": ", ".join(df.columns.astype(str)),
        }]

    # Group separately by every available configuration column.
    config_cols = [c for c in GROUP_COLS if c in df.columns]

    if config_cols:
        grouped = df.groupby(config_cols, dropna=False, sort=False)
    else:
        grouped = [("single_configuration", df)]

    metric_rows = []
    audit_rows = []

    for config, part in grouped:
        if not isinstance(config, tuple):
            config = (config,)

        config_values = dict(zip(config_cols, config))
        part = part.dropna(subset=[true_col, pred_col]).copy()

        if part.empty:
            continue

        y_true = part[true_col].astype(int).to_numpy()
        y_pred = part[pred_col].astype(int).to_numpy()

        row = {
            "experiment": exp_id,
            "source": "recomputed_from_predictions",
            "prediction_file": str(path),
            "configuration": json.dumps(
                config_values if config_cols else {"name": "single_configuration"},
                sort_keys=True,
            ),
            "prediction_rows": int(len(part)),
            "pooled_accuracy": float(accuracy_score(y_true, y_pred)),
            "pooled_balanced_accuracy": float(
                balanced_accuracy_score(y_true, y_pred)
            ),
            "pooled_macro_f1": float(
                f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0,
                )
            ),
            "confusion_matrix_labels_0_1": json.dumps(
                confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
            ),
        }

        if subject_col is not None:
            row["n_participants"] = int(part[subject_col].nunique())

            participant_acc = []
            participant_f1 = []

            for _, subject_df in part.groupby(subject_col, dropna=False):
                sy = subject_df[true_col].astype(int).to_numpy()
                sp = subject_df[pred_col].astype(int).to_numpy()

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

        # Check trial-key uniqueness when useful identifiers are available.
        trial_col = first_column(
            part,
            ["trial_number", "trial_id", "trial", "epoch_id"],
        )
        source_col = first_column(
            part,
            ["source_file", "file"],
        )

        if subject_col and trial_col:
            key_cols = [subject_col, trial_col]
            if source_col:
                key_cols.append(source_col)

            row["duplicate_trial_key_rows"] = int(
                part.duplicated(subset=key_cols, keep=False).sum()
            )
        else:
            row["duplicate_trial_key_rows"] = np.nan

        metric_rows.append(row)

        audit_rows.append({
            "experiment": exp_id,
            "file": str(path),
            "configuration": row["configuration"],
            "status": "READ",
            "true_label_column": true_col,
            "predicted_label_column": pred_col,
            "participant_column": subject_col or "",
            "prediction_rows": int(len(part)),
            "n_participants": (
                int(part[subject_col].nunique())
                if subject_col else np.nan
            ),
            "duplicate_trial_key_rows": row["duplicate_trial_key_rows"],
        })

    return metric_rows, audit_rows


def main():
    summary_rows = []
    prediction_rows = []
    audit_rows = []
    file_rows = []

    print("=" * 72)
    print("EXPERIMENT 144 — CORRECTED RESULTS CONSOLIDATION")
    print("=" * 72)
    print("No model training will be performed.")
    print("Original data and previous results will not be modified.")

    for exp_id, folder in EXPERIMENTS.items():
        print(f"\nInspecting Experiment {exp_id}")

        summary_records, summary_path = read_summary_records(exp_id, folder)

        for row in summary_records:
            summary_rows.append(row)

        pred_path = find_file(folder, PREDICTION_NAMES)

        file_rows.append({
            "experiment": exp_id,
            "folder_exists": folder.exists(),
            "summary_file": str(summary_path) if summary_path else "",
            "prediction_file": str(pred_path) if pred_path else "",
        })

        if pred_path:
            try:
                metrics, audits = metric_row_from_predictions(
                    exp_id, pred_path
                )
                prediction_rows.extend(metrics)
                audit_rows.extend(audits)
            except Exception as exc:
                audit_rows.append({
                    "experiment": exp_id,
                    "file": str(pred_path),
                    "status": f"ERROR: {exc}",
                })
        else:
            audit_rows.append({
                "experiment": exp_id,
                "file": "",
                "status": "No recognized prediction file found",
            })

    summaries_df = pd.DataFrame(summary_rows)
    predictions_df = pd.DataFrame(prediction_rows)
    audits_df = pd.DataFrame(audit_rows)
    files_df = pd.DataFrame(file_rows)

    summaries_path = OUT / "summary_metrics_extracted.csv"
    predictions_path = OUT / "prediction_metrics_recomputed.csv"
    audit_path = OUT / "prediction_file_audit.csv"
    files_path = OUT / "inspected_files.csv"
    report_path = OUT / "final_results_report.txt"

    summaries_df.to_csv(summaries_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    audits_df.to_csv(audit_path, index=False)
    files_df.to_csv(files_path, index=False)

    report_lines = [
        "DATASET 3 — FINAL RESULTS REPORT",
        "=" * 72,
        "",
        "This report consolidates existing summaries and saved predictions.",
        "No models were retrained.",
        "",
        "METRICS EXTRACTED FROM SUMMARY JSON FILES",
        "-" * 72,
        (
            summaries_df.to_string(index=False)
            if not summaries_df.empty
            else "No readable summary records found."
        ),
        "",
        "METRICS RECOMPUTED FROM PREDICTIONS",
        "-" * 72,
        (
            predictions_df.to_string(index=False)
            if not predictions_df.empty
            else "No compatible prediction files found."
        ),
        "",
        "PREDICTION FILE AUDIT",
        "-" * 72,
        (
            audits_df.to_string(index=False)
            if not audits_df.empty
            else "No prediction files inspected."
        ),
        "",
        "INTERPRETATION",
        "-" * 72,
        "1. Summary metrics and prediction-recomputed metrics are kept separate.",
        "2. Multiple feature/model configurations are reported individually.",
        "3. A duplicate trial key is a warning requiring inspection, not automatic",
        "   proof of invalid data.",
        "4. Metrics should only be compared when the task, labels, evaluation",
        "   protocol, and metric definitions are compatible.",
        "5. The selected best result is exploratory because the same cohort was",
        "   reused for model/feature comparisons.",
        "",
        "OUTPUT FILES",
        "-" * 72,
        str(summaries_path),
        str(predictions_path),
        str(audit_path),
        str(files_path),
    ]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("\n" + "=" * 72)
    print("EXPERIMENT 144 COMPLETE")
    print("=" * 72)
    print(f"Summary configurations extracted: {len(summaries_df)}")
    print(f"Prediction configurations recomputed: {len(predictions_df)}")
    print(f"Prediction audit rows: {len(audits_df)}")

    print("\nSaved files:")
    print(f"  {summaries_path}")
    print(f"  {predictions_path}")
    print(f"  {audit_path}")
    print(f"  {files_path}")
    print(f"  {report_path}")

    print("\nRecomputed prediction metrics preview:")
    if predictions_df.empty:
        print("No metrics available.")
    else:
        preview_cols = [
            c for c in [
                "experiment",
                "configuration",
                "prediction_rows",
                "n_participants",
                "pooled_accuracy",
                "pooled_balanced_accuracy",
                "pooled_macro_f1",
                "mean_participant_accuracy",
                "mean_participant_macro_f1",
            ]
            if c in predictions_df.columns
        ]
        print(predictions_df[preview_cols].to_string(index=False))


if __name__ == "__main__":
    main()
