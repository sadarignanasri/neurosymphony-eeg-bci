
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.signal import welch

# ============================================================
# EXPERIMENT 100: EEG PREPROCESSING AUDIT
# Dataset: ds004022
# Purpose: Validate labels, signal quality and band power
# ============================================================

ROOT = Path(r"D:\neurosymphony-eeg-bci")

INPUT_DIR = ROOT / "experiments" / "ds004022_verified_epochs"
NPZ_PATH = INPUT_DIR / "eeg_epochs.npz"
METADATA_PATH = INPUT_DIR / "metadata.json"

OUTPUT_DIR = ROOT / "experiments" / "ds004022_experiment100_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FS = 500.0

# Frequency bands in Hz
BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}

# Expected mapping from the established extraction pipeline.
# Metadata will be inspected before this mapping is accepted.
EXPECTED_CLASS_MAP = {
    0: "reaching",
    1: "grasping",
    2: "lifting",
    3: "twisting",
}


def find_key(data, candidates, required=True):
    """Find an array using common key names."""
    for key in candidates:
        if key in data:
            return key

    if required:
        raise KeyError(
            f"Could not find any of {candidates}. "
            f"Available NPZ keys: {list(data.keys())}"
        )

    return None


def load_metadata():
    if not METADATA_PATH.exists():
        print("WARNING: metadata.json was not found.")
        return {}

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def normalize_class_mapping(metadata, labels):
    """
    Use an explicit class mapping from metadata if available.
    Otherwise use the established extraction-pipeline mapping,
    but only when labels are exactly 0, 1, 2, 3.
    """
    possible_maps = [
        metadata.get("class_map"),
        metadata.get("class_mapping"),
        metadata.get("label_mapping"),
    ]

    for mapping in possible_maps:
        if isinstance(mapping, dict) and mapping:
            try:
                normalized = {
                    int(key): str(value)
                    for key, value in mapping.items()
                }
                if set(np.unique(labels)).issubset(set(normalized)):
                    return normalized, "metadata"
            except (TypeError, ValueError):
                pass

    unique_labels = set(np.unique(labels).tolist())

    if unique_labels == {0, 1, 2, 3}:
        return EXPECTED_CLASS_MAP.copy(), "expected extraction mapping"

    return {
        label: str(label)
        for label in np.unique(labels).tolist()
    }, "raw label values; mapping needs manual verification"


def bandpower(signal, low, high):
    """Calculate mean Welch PSD within a frequency band."""
    frequencies, psd = welch(
        signal,
        fs=FS,
        nperseg=min(1024, len(signal)),
        detrend="constant",
    )

    mask = (frequencies >= low) & (frequencies < high)

    if not np.any(mask):
        return np.nan

    return float(np.trapezoid(psd[mask], frequencies[mask]))


def main():
    if not NPZ_PATH.exists():
        raise FileNotFoundError(
            f"Input epochs not found: {NPZ_PATH}"
        )

    metadata = load_metadata()

    with np.load(NPZ_PATH, allow_pickle=True) as archive:
        data = {key: archive[key] for key in archive.files}

    print("\n========== INPUT INSPECTION ==========")
    print("NPZ keys:", list(data.keys()))

    x_key = find_key(data, ["X", "x", "epochs", "eeg_epochs"])
    y_key = find_key(data, ["y", "labels", "target", "targets"])
    subject_key = find_key(
        data, ["subjects", "subject_ids", "subject", "participant_ids"]
    )
    run_key = find_key(
        data, ["runs", "run_ids", "run"], required=False
    )

    X = np.asarray(data[x_key], dtype=np.float64)
    y = np.asarray(data[y_key]).reshape(-1)
    subjects = np.asarray(data[subject_key]).astype(str).reshape(-1)

    runs = (
        np.asarray(data[run_key]).astype(str).reshape(-1)
        if run_key is not None
        else np.array(["unknown"] * len(y))
    )

    if X.ndim != 3:
        raise ValueError(
            f"Expected EEG shape (trials, channels, samples); got {X.shape}"
        )

    n_trials, n_channels, n_samples = X.shape

    if not (len(y) == len(subjects) == len(runs) == n_trials):
        raise ValueError(
            "Mismatch between EEG trials, labels, subjects and runs."
        )

    if not np.isfinite(X).all():
        bad_values = int(np.size(X) - np.isfinite(X).sum())
        raise ValueError(
            f"EEG contains {bad_values} NaN/Inf values. "
            "Investigate before model training."
        )

    class_map, mapping_source = normalize_class_mapping(metadata, y)

    print("EEG shape:", X.shape)
    print("Trials:", n_trials)
    print("Channels:", n_channels)
    print("Samples per trial:", n_samples)
    print("Sampling rate:", FS, "Hz")
    print("Trial duration:", round(n_samples / FS, 3), "seconds")
    print("Subjects:", sorted(np.unique(subjects).tolist()))
    print("Runs:", len(np.unique(runs)))
    print("Class mapping source:", mapping_source)
    print("Class mapping:", class_map)

    # --------------------------------------------------------
    # 1. TRIAL-LEVEL SIGNAL QUALITY
    # --------------------------------------------------------
    trial_rows = []

    for i in range(n_trials):
        trial = X[i]

        channel_means = np.mean(trial, axis=1)
        channel_stds = np.std(trial, axis=1)
        channel_ptp = np.ptp(trial, axis=1)

        row = {
            "trial_index": i,
            "subject": subjects[i],
            "run": runs[i],
            "label_id": str(y[i]),
            "class_name": class_map.get(y[i], str(y[i])),
            "mean_channel_std": float(np.mean(channel_stds)),
            "median_channel_std": float(np.median(channel_stds)),
            "max_channel_std": float(np.max(channel_stds)),
            "mean_channel_peak_to_peak": float(np.mean(channel_ptp)),
            "max_channel_peak_to_peak": float(np.max(channel_ptp)),
            "mean_absolute_amplitude": float(np.mean(np.abs(trial))),
            "fraction_zero_std_channels": float(np.mean(channel_stds == 0)),
            "has_extreme_amplitude": bool(np.max(np.abs(trial)) > 500),
        }

        for band, (low, high) in BANDS.items():
            values = [
                bandpower(channel, low, high)
                for channel in trial
            ]
            row[f"{band}_mean_power"] = float(np.mean(values))

        trial_rows.append(row)

    trial_df = pd.DataFrame(trial_rows)
    trial_df.to_csv(OUTPUT_DIR / "trial_signal_quality.csv", index=False)

    # --------------------------------------------------------
    # 2. CLASS COUNTS
    # --------------------------------------------------------
    class_counts = (
        trial_df.groupby(["label_id", "class_name"])
        .size()
        .reset_index(name="trial_count")
    )
    class_counts.to_csv(OUTPUT_DIR / "class_counts.csv", index=False)

    # --------------------------------------------------------
    # 3. SUBJECT AND CLASS SUMMARY
    # --------------------------------------------------------
    summary_columns = [
        "mean_channel_std",
        "median_channel_std",
        "max_channel_std",
        "mean_channel_peak_to_peak",
        "mean_absolute_amplitude",
        "delta_mean_power",
        "theta_mean_power",
        "alpha_mean_power",
        "beta_mean_power",
        "gamma_mean_power",
    ]

    subject_class_summary = (
        trial_df.groupby(["subject", "class_name"])[summary_columns]
        .agg(["mean", "median", "std"])
        .reset_index()
    )

    subject_class_summary.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in subject_class_summary.columns
    ]

    subject_class_summary.to_csv(
        OUTPUT_DIR / "subject_class_signal_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 4. OUTLIER FLAGS
    # These are screening flags, not automatic trial rejection.
    # --------------------------------------------------------
    quality_column = "mean_channel_std"
    q1 = trial_df[quality_column].quantile(0.25)
    q3 = trial_df[quality_column].quantile(0.75)
    iqr = q3 - q1

    lower = max(0, q1 - 3 * iqr)
    upper = q3 + 3 * iqr

    trial_df["std_outlier_flag"] = (
        (trial_df[quality_column] < lower)
        | (trial_df[quality_column] > upper)
    )

    flagged = trial_df[
        trial_df["std_outlier_flag"]
        | trial_df["has_extreme_amplitude"]
        | (trial_df["fraction_zero_std_channels"] > 0)
    ]

    flagged.to_csv(OUTPUT_DIR / "trials_to_review.csv", index=False)

    # --------------------------------------------------------
    # 5. SUMMARY REPORT
    # --------------------------------------------------------
    report = {
        "dataset": "ds004022",
        "input_file": str(NPZ_PATH),
        "shape": list(X.shape),
        "sampling_rate_hz": FS,
        "trial_duration_seconds": n_samples / FS,
        "number_of_trials": int(n_trials),
        "number_of_channels": int(n_channels),
        "number_of_subjects": int(len(np.unique(subjects))),
        "number_of_runs": int(len(np.unique(runs))),
        "class_mapping_source": mapping_source,
        "class_mapping": {
            str(key): value for key, value in class_map.items()
        },
        "labels_and_counts": {
            str(label): int(np.sum(y == label))
            for label in np.unique(y)
        },
        "nonfinite_values": int(np.size(X) - np.isfinite(X).sum()),
        "trials_flagged_for_review": int(len(flagged)),
        "outlier_rule": "3 times IQR on mean channel standard deviation",
        "warning": (
            "Flags are for review only. Do not automatically remove trials. "
            "Verify class mapping against the original dataset metadata."
        ),
    }

    with open(
        OUTPUT_DIR / "audit_report.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, indent=2)

    print("\n========== AUDIT SUMMARY ==========")
    print("Output folder:", OUTPUT_DIR)
    print("Class counts:")
    print(class_counts.to_string(index=False))
    print("Trials flagged for review:", len(flagged))
    print("\nFiles generated:")
    print("1. trial_signal_quality.csv")
    print("2. class_counts.csv")
    print("3. subject_class_signal_summary.csv")
    print("4. trials_to_review.csv")
    print("5. audit_report.json")
    print("\nIMPORTANT: Flags do not mean trials must be deleted.")
    print("Review label mapping and signal statistics before retraining.")


if __name__ == "__main__":
    main()
