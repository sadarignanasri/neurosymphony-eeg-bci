
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch


# ============================================================
# EXPERIMENT 104: EEG FEATURE AND LABEL ALIGNMENT AUDIT
# Dataset: ds004022
# Read-only audit; previous experiments remain untouched.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT
    / "experiments"
    / "ds004022_verified_epochs"
    / "eeg_epochs.npz"
)

EXTRACTION_AUDIT = (
    ROOT
    / "experiments"
    / "ds004022_verified_epochs"
    / "trial_audit.csv"
)

OUT_DIR = ROOT / "experiments" / "ds004022_experiment104_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FS = 500

BANDS = {
    "delta_1_4": (1, 4),
    "theta_4_8": (4, 8),
    "alpha_8_13": (8, 13),
    "beta_13_30": (13, 30),
    "gamma_30_45": (30, 45),
}


def get_array(data, candidates, required=True):
    for key in candidates:
        if key in data.files:
            return data[key]

    if required:
        raise KeyError(
            f"Could not find {candidates}. "
            f"Available keys: {data.files}"
        )

    return None


def normalize_label_names(class_names, labels):
    if class_names is None:
        return {str(label): str(label) for label in np.unique(labels)}

    values = np.asarray(class_names).tolist()
    return {
        str(i): str(name)
        for i, name in enumerate(values)
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    with np.load(DATA_PATH, allow_pickle=True) as data:
        print("NPZ keys:", data.files)

        X = get_array(data, ["X", "epochs", "eeg_epochs"])
        y = get_array(data, ["y", "labels", "targets"]).reshape(-1)
        subjects = get_array(
            data, ["subjects", "subject_ids", "subject"]
        ).reshape(-1).astype(str)

        runs = get_array(
            data, ["runs", "run_ids", "run"], required=False
        )
        class_names = get_array(
            data, ["class_names", "classes", "label_names"],
            required=False
        )

    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 3:
        raise ValueError(f"Expected 3-D epochs, got {X.shape}")

    if not (len(X) == len(y) == len(subjects)):
        raise ValueError("Epoch, label, and subject counts differ.")

    if runs is not None:
        runs = np.asarray(runs).reshape(-1).astype(str)
        if len(runs) != len(y):
            raise ValueError("Run count differs from epoch count.")

    label_names = normalize_label_names(class_names, y)
    labels = np.unique(y)

    print("\nDATASET SUMMARY")
    print("----------------")
    print("Shape:", X.shape)
    print("Subjects:", np.unique(subjects).tolist())
    print("Labels:", labels.tolist())
    print("Trials per class:", pd.Series(y).value_counts().to_dict())
    print("Sampling rate:", FS, "Hz")

    # --------------------------------------------------------
    # 1. Trial-level signal quality
    # --------------------------------------------------------
    print("\nCalculating signal quality...")

    finite = np.isfinite(X)
    safe_X = np.where(finite, X, 0.0)

    trial_rows = []

    for i in range(len(X)):
        epoch = safe_X[i]
        channel_std = np.std(epoch, axis=1)
        channel_ptp = np.ptp(epoch, axis=1)

        trial_rows.append({
            "trial_index": i,
            "subject": subjects[i],
            "run": runs[i] if runs is not None else "unknown",
            "label": str(y[i]),
            "class_name": label_names.get(str(y[i]), str(y[i])),
            "all_values_finite": bool(finite[i].all()),
            "nonfinite_value_count": int((~finite[i]).sum()),
            "epoch_mean": float(np.mean(epoch)),
            "epoch_std": float(np.std(epoch)),
            "epoch_min": float(np.min(epoch)),
            "epoch_max": float(np.max(epoch)),
            "mean_channel_std": float(np.mean(channel_std)),
            "minimum_channel_std": float(np.min(channel_std)),
            "minimum_channel_ptp": float(np.min(channel_ptp)),
            "channels_with_near_zero_std": int(
                np.sum(channel_std < 1e-12)
            ),
        })

    quality_df = pd.DataFrame(trial_rows)
    quality_df.to_csv(OUT_DIR / "trial_signal_quality.csv", index=False)

    # --------------------------------------------------------
    # 2. Class counts per subject and run
    # --------------------------------------------------------
    count_df = pd.DataFrame({
        "subject": subjects,
        "label": y.astype(str),
        "class_name": [
            label_names.get(str(label), str(label)) for label in y
        ],
        "run": runs if runs is not None else ["unknown"] * len(y),
    })

    counts = (
        count_df.groupby(["subject", "run", "label", "class_name"])
        .size()
        .reset_index(name="trial_count")
    )
    counts.to_csv(OUT_DIR / "class_counts_by_subject_run.csv", index=False)

    subject_class_counts = (
        count_df.groupby(["subject", "label", "class_name"])
        .size()
        .reset_index(name="trial_count")
    )
    subject_class_counts.to_csv(
        OUT_DIR / "class_counts_by_subject.csv", index=False
    )

    # --------------------------------------------------------
    # 3. Band power using Welch PSD
    # --------------------------------------------------------
    print("Calculating band-power features...")

    feature_rows = []

    for i in range(len(X)):
        epoch = safe_X[i]

        # Welch PSD per channel; average power across channels.
        freqs, psd = welch(
            epoch,
            fs=FS,
            nperseg=min(1000, epoch.shape[-1]),
            axis=-1,
        )

        row = {
            "trial_index": i,
            "subject": subjects[i],
            "run": runs[i] if runs is not None else "unknown",
            "label": str(y[i]),
            "class_name": label_names.get(str(y[i]), str(y[i])),
        }

        for band_name, (low, high) in BANDS.items():
            mask = (freqs >= low) & (freqs < high)

            if np.any(mask):
                channel_band_power = np.trapezoid(
                    psd[:, mask], x=freqs[mask], axis=1
                )
                row[f"{band_name}_mean_power"] = float(
                    np.mean(channel_band_power)
                )
                row[f"{band_name}_median_power"] = float(
                    np.median(channel_band_power)
                )
            else:
                row[f"{band_name}_mean_power"] = np.nan
                row[f"{band_name}_median_power"] = np.nan

        feature_rows.append(row)

    features_df = pd.DataFrame(feature_rows)
    features_df.to_csv(OUT_DIR / "band_power_features.csv", index=False)

    # --------------------------------------------------------
    # 4. Feature distribution by class and subject
    # --------------------------------------------------------
    feature_columns = [
        column for column in features_df.columns
        if column.endswith("_mean_power")
        or column.endswith("_median_power")
    ]

    class_summary = (
        features_df.groupby(["label", "class_name"])[feature_columns]
        .agg(["mean", "median", "std"])
    )
    class_summary.to_csv(OUT_DIR / "band_power_by_class.csv")

    subject_summary = (
        features_df.groupby(["subject", "label", "class_name"])[feature_columns]
        .mean()
        .reset_index()
    )
    subject_summary.to_csv(
        OUT_DIR / "band_power_by_subject_and_class.csv", index=False
    )

    # --------------------------------------------------------
    # 5. Compare trial ordering to extraction audit if available
    # --------------------------------------------------------
    alignment_info = {
        "extraction_audit_exists": EXTRACTION_AUDIT.exists(),
        "extraction_audit_row_count": None,
        "warning": None,
    }

    if EXTRACTION_AUDIT.exists():
        audit_df = pd.read_csv(EXTRACTION_AUDIT)
        alignment_info["extraction_audit_row_count"] = int(len(audit_df))

        # Do not assume audit column names or ordering semantics.
        # Save its column names for a manual, schema-aware comparison.
        alignment_info["extraction_audit_columns"] = audit_df.columns.tolist()

        if len(audit_df) != len(X):
            alignment_info["warning"] = (
                "Audit row count differs from epoch count. "
                "Inspect audit schema and extraction ordering."
            )
        else:
            alignment_info["warning"] = (
                "Row counts match, but row-count agreement alone does "
                "not prove that each epoch is aligned with its label."
            )
    else:
        alignment_info["warning"] = (
            "Extraction audit file not found; alignment could not "
            "be compared against that file."
        )

    # --------------------------------------------------------
    # 6. Summary report
    # --------------------------------------------------------
    summary = {
        "experiment": 104,
        "dataset": "ds004022",
        "epochs_shape": list(X.shape),
        "n_trials": int(len(X)),
        "n_channels": int(X.shape[1]),
        "n_samples_per_epoch": int(X.shape[2]),
        "sampling_frequency_hz": FS,
        "subjects": np.unique(subjects).tolist(),
        "labels": [str(label) for label in labels],
        "class_names": label_names,
        "class_counts": {
            str(label): int(np.sum(y == label)) for label in labels
        },
        "all_data_finite": bool(finite.all()),
        "nonfinite_value_count": int((~finite).sum()),
        "trials_with_nonfinite_values": int(
            np.sum(~finite.reshape(len(X), -1).all(axis=1))
        ),
        "trials_with_near_zero_std_channel": int(
            (quality_df["channels_with_near_zero_std"] > 0).sum()
        ),
        "signal_amplitude_quantiles": {
            str(q): float(np.quantile(np.abs(safe_X), q))
            for q in [0.5, 0.9, 0.99, 1.0]
        },
        "alignment_check": alignment_info,
        "interpretation_warning": (
            "Band-power differences are descriptive only. They do not "
            "prove class separability or establish causal relationships. "
            "The extraction audit row count alone cannot prove trial alignment."
        ),
    }

    with open(
        OUT_DIR / "experiment104_report.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    print("\nEXPERIMENT 104 COMPLETE")
    print("-----------------------")
    print(json.dumps(summary, indent=2))
    print("\nOutputs:", OUT_DIR)


if __name__ == "__main__":
    main()
