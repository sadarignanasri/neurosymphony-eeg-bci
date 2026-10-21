
from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, welch, detrend
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# =========================================================
# 1. PATHS — ORIGINAL DATASET IS READ-ONLY
# =========================================================
ROOT = Path(r"D:\neurosymphony-eeg-bci")

DATA_ROOT = ROOT / "major datasets" / "28955456" / "epo_data"

OUTPUT_ROOT = (
    ROOT
    / "experiments"
    / "dataset3_28955456_training"
    / "experiment134_corrected_features"
)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# =========================================================
# 2. EXPERIMENT CONFIGURATION
# =========================================================
FS = 256

# Epoch-relative seconds; approximately cue +3s to cue +12s.
WINDOW_START_SEC = 13.0
WINDOW_END_SEC = 22.0

LOW_CUT = 8.0
HIGH_CUT = 30.0

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

RANDOM_STATE = 42
EPS = 1e-20

# =========================================================
# 3. FIND EEG FILES — RECURSIVE SUBJECT-FOLDER SEARCH
# =========================================================
def find_dataset_files():
    files = []

    for group in ("Normals", "Patients"):
        eeg_dir = DATA_ROOT / group / "EEG"

        if not eeg_dir.is_dir():
            raise FileNotFoundError(
                f"Missing EEG directory: {eeg_dir}"
            )

        for path in sorted(eeg_dir.rglob("*_epo.mat")):
            # Subject IDs are unique across groups.
            participant = f"{group}_{path.parent.name}"
            files.append((path, group, participant))

    if not files:
        raise FileNotFoundError(
            f"No *_epo.mat files found under {DATA_ROOT}"
        )

    unique_participants = {
        participant for _, _, participant in files
    }

    print(f"Discovered EEG epoch files: {len(files)}")
    print(f"Discovered participants: {len(unique_participants)}")

    return files


# =========================================================
# 4. LOAD MATLAB EPOCH FILE
# =========================================================
def load_epoch_file(path):
    mat = loadmat(
        path,
        squeeze_me=False,
        struct_as_record=False,
    )

    if "epo" not in mat:
        raise ValueError(f"Missing 'epo' structure: {path}")

    epo = mat["epo"][0, 0]

    X = np.asarray(epo.X, dtype=np.float64)
    y = np.asarray(epo.y).reshape(-1).astype(int)
    fs = float(np.asarray(epo.fs).reshape(-1)[0])

    # Expected original shape: samples x channels x trials.
    if X.ndim != 3:
        raise ValueError(f"Unexpected EEG shape {X.shape}: {path}")

    if X.shape[2] != len(y):
        raise ValueError(
            f"Trial/label mismatch in {path}: "
            f"X={X.shape}, y={y.shape}"
        )

    if not np.isclose(fs, FS):
        raise ValueError(
            f"Expected {FS} Hz but found {fs} Hz in {path}"
        )

    if not np.isfinite(X).all():
        raise ValueError(f"NaN/Inf EEG values found in {path}")

    if not np.isin(y, [0, 1]).all():
        raise ValueError(
            f"Unexpected labels in {path}: {np.unique(y)}"
        )

    # Convert to trials x channels x samples.
    X = np.transpose(X, (2, 1, 0))

    return X, y


# =========================================================
# 5. BAND-POWER INTEGRATION WITH INTERPOLATED BOUNDARIES
# =========================================================
def integrate_band_power(freqs, psd, low, high):
    """
    Integrate PSD between exact frequency boundaries.

    Interpolates PSD at low/high edges before trapezoidal
    integration, instead of integrating only interior bins.

    psd shape: trials x channels x frequency_bins.
    """
    if low < freqs[0] or high > freqs[-1]:
        raise ValueError(
            f"Requested band {low}-{high} Hz exceeds "
            f"PSD range {freqs[0]}-{freqs[-1]} Hz"
        )

    interior = (freqs > low) & (freqs < high)

    low_values = np.interp(
        low, freqs, np.arange(len(freqs), dtype=float)
    )
    high_values = np.interp(
        high, freqs, np.arange(len(freqs), dtype=float)
    )

    # Interpolate PSD values along the final frequency axis.
    low_psd = np.apply_along_axis(
        lambda row: np.interp(low, freqs, row),
        -1,
        psd,
    )
    high_psd = np.apply_along_axis(
        lambda row: np.interp(high, freqs, row),
        -1,
        psd,
    )

    selected_freqs = np.concatenate(
        ([low], freqs[interior], [high])
    )

    selected_psd = np.concatenate(
        (
            low_psd[..., None],
            psd[..., interior],
            high_psd[..., None],
        ),
        axis=-1,
    )

    power = np.trapezoid(
        selected_psd,
        x=selected_freqs,
        axis=-1,
    )

    return np.maximum(power, EPS)


# =========================================================
# 6. FEATURE EXTRACTION
# =========================================================
def extract_features(trials):
    """
    Input:
        trials x channels x samples

    Feature families:
      1. log_variance: variance after 8-30 Hz filtering
      2. log_band_power: PSD band powers from detrended EEG
      3. relative_band_power: band power / total 1-45 Hz power
      4. combined: log_variance + log_band_power

    Spectral power is computed from the detrended, unfiltered
    EEG window, not from the 8-30 Hz filtered signal.
    """
    start = int(round(WINDOW_START_SEC * FS))
    end = int(round(WINDOW_END_SEC * FS))

    if trials.shape[2] < end:
        raise ValueError(
            f"Epoch has {trials.shape[2]} samples; "
            f"at least {end} are required."
        )

    n_trials, n_channels, _ = trials.shape

    # Baseline log variance from 8-30 Hz EEG.
    sos = butter(
        4,
        [LOW_CUT, HIGH_CUT],
        btype="bandpass",
        fs=FS,
        output="sos",
    )

    filtered = sosfiltfilt(sos, trials, axis=-1)
    filtered_window = filtered[:, :, start:end]

    log_variance = np.log(
        np.var(filtered_window, axis=-1) + EPS
    )

    # PSD from the original EEG window after linear detrending.
    raw_window = trials[:, :, start:end]
    raw_window = detrend(
        raw_window,
        axis=-1,
        type="linear",
    )

    freqs, psd = welch(
        raw_window,
        fs=FS,
        nperseg=512,
        noverlap=256,
        axis=-1,
        scaling="density",
    )

    band_power_list = []

    for band_name, (low, high) in BANDS.items():
        power = integrate_band_power(
            freqs, psd, low, high
        )
        band_power_list.append(power)

    # trials x channels x 5 bands
    band_power = np.stack(band_power_list, axis=-1)

    total_power = integrate_band_power(
        freqs, psd, 1.0, 45.0
    )

    relative_power = (
        band_power / total_power[:, :, None]
    )

    log_band_power = np.log(
        band_power
    ).reshape(n_trials, -1)

    relative_band_power = relative_power.reshape(
        n_trials, -1
    )

    combined = np.concatenate(
        [log_variance, log_band_power],
        axis=1,
    )

    return {
        "log_variance": log_variance,
        "log_band_power": log_band_power,
        "relative_band_power": relative_band_power,
        "combined": combined,
    }


# =========================================================
# 7. BUILD DATASET
# =========================================================
def build_dataset():
    feature_parts = {
        "log_variance": [],
        "log_band_power": [],
        "relative_band_power": [],
        "combined": [],
    }

    labels = []
    participants = []
    groups = []
    source_files = []
    trial_numbers = []
    inventory = []

    files = find_dataset_files()

    for file_index, (path, group, participant) in enumerate(
        files, start=1
    ):
        X, y = load_epoch_file(path)
        features = extract_features(X)

        for name in feature_parts:
            feature_parts[name].append(features[name])

        labels.extend(y.tolist())
        participants.extend([participant] * len(y))
        groups.extend([group] * len(y))
        source_files.extend([str(path)] * len(y))
        trial_numbers.extend(range(len(y)))

        inventory.append({
            "file": str(path),
            "group": group,
            "participant": participant,
            "trials": int(len(y)),
            "channels": int(X.shape[1]),
            "sampling_rate_hz": FS,
            "label_0": int(np.sum(y == 0)),
            "label_1": int(np.sum(y == 1)),
        })

        print(
            f"[{file_index}/{len(files)}] "
            f"{participant} | {path.name} | "
            f"trials={len(y)} | "
            f"class_counts={dict(zip(*np.unique(y, return_counts=True)))}"
        )

    X_features = {
        name: np.concatenate(parts, axis=0)
        for name, parts in feature_parts.items()
    }

    metadata = pd.DataFrame({
        "label": labels,
        "participant": participants,
        "group": groups,
        "source_file": source_files,
        "trial_number": trial_numbers,
    })

    pd.DataFrame(inventory).to_csv(
        OUTPUT_ROOT / "file_inventory.csv",
        index=False,
    )

    return X_features, metadata


# =========================================================
# 8. MODELS
# =========================================================
def build_models():
    return {
        "RBF_SVM": make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight="balanced",
            ),
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


# =========================================================
# 9. LEAVE-ONE-PARTICIPANT-OUT EVALUATION
# =========================================================
def evaluate(X_features, metadata):
    y = metadata["label"].to_numpy()
    participant_ids = metadata["participant"].to_numpy()
    unique_participants = np.unique(participant_ids)

    fold_records = []
    prediction_records = []
    start_time = time.time()

    for feature_name, X in X_features.items():
        print(
            f"\nFEATURE FAMILY: {feature_name} "
            f"| feature shape={X.shape}"
        )

        for model_name, model_template in build_models().items():
            print(f"\nMODEL: {model_name}")

            for fold_number, held_out in enumerate(
                unique_participants, start=1
            ):
                test_mask = participant_ids == held_out
                train_mask = ~test_mask

                y_train = y[train_mask]
                y_test = y[test_mask]

                if np.unique(y_train).size != 2:
                    raise ValueError(
                        f"Training fold for {held_out} "
                        "does not contain both classes."
                    )

                # Fresh model for every held-out participant.
                from sklearn.base import clone
                model = clone(model_template)

                model.fit(X[train_mask], y_train)
                pred = model.predict(X[test_mask])

                fold_records.append({
                    "feature_family": feature_name,
                    "model": model_name,
                    "held_out_participant": held_out,
                    "group": metadata.loc[
                        test_mask, "group"
                    ].iloc[0],
                    "n_trials": int(test_mask.sum()),
                    "accuracy": accuracy_score(y_test, pred),
                    "balanced_accuracy": balanced_accuracy_score(
                        y_test, pred
                    ),
                    "macro_f1": f1_score(
                        y_test,
                        pred,
                        average="macro",
                        zero_division=0,
                    ),
                })

                test_indices = np.flatnonzero(test_mask)

                for idx, prediction in zip(test_indices, pred):
                    prediction_records.append({
                        "feature_family": feature_name,
                        "model": model_name,
                        "participant": participant_ids[idx],
                        "group": metadata.iloc[idx]["group"],
                        "source_file": metadata.iloc[idx]["source_file"],
                        "trial_number": int(
                            metadata.iloc[idx]["trial_number"]
                        ),
                        "true_label": int(y[idx]),
                        "predicted_label": int(prediction),
                    })

                if fold_number % 5 == 0 or fold_number == len(
                    unique_participants
                ):
                    print(
                        f"  Completed participant fold "
                        f"{fold_number}/{len(unique_participants)}"
                    )

    folds = pd.DataFrame(fold_records)
    predictions = pd.DataFrame(prediction_records)

    folds.to_csv(
        OUTPUT_ROOT / "participant_fold_metrics.csv",
        index=False,
    )

    predictions.to_csv(
        OUTPUT_ROOT / "out_of_fold_predictions.csv",
        index=False,
    )

    # Summarize each model-feature combination.
    summary_rows = []

    for (feature_name, model_name), fold_group in folds.groupby(
        ["feature_family", "model"],
        sort=False,
    ):
        pred_group = predictions[
            (predictions["feature_family"] == feature_name)
            & (predictions["model"] == model_name)
        ]

        summary_rows.append({
            "feature_family": feature_name,
            "model": model_name,
            "pooled_accuracy": accuracy_score(
                pred_group["true_label"],
                pred_group["predicted_label"],
            ),
            "pooled_balanced_accuracy": balanced_accuracy_score(
                pred_group["true_label"],
                pred_group["predicted_label"],
            ),
            "pooled_macro_f1": f1_score(
                pred_group["true_label"],
                pred_group["predicted_label"],
                average="macro",
                zero_division=0,
            ),
            "mean_participant_accuracy":
                fold_group["accuracy"].mean(),
            "std_participant_accuracy":
                fold_group["accuracy"].std(),
            "mean_participant_balanced_accuracy":
                fold_group["balanced_accuracy"].mean(),
            "mean_participant_macro_f1":
                fold_group["macro_f1"].mean(),
            "participants":
                int(fold_group["held_out_participant"].nunique()),
        })

    summary = pd.DataFrame(summary_rows).sort_values(
        ["pooled_balanced_accuracy", "pooled_macro_f1"],
        ascending=False,
    )

    summary.to_csv(
        OUTPUT_ROOT / "corrected_feature_comparison.csv",
        index=False,
    )

    summary_json = {
        "experiment": 134,
        "description": (
            "Corrected PSD band-power integration and "
            "participant-wise leave-one-participant-out evaluation."
        ),
        "original_dataset_modified": False,
        "sampling_rate_hz": FS,
        "epoch_relative_window_seconds": [
            WINDOW_START_SEC,
            WINDOW_END_SEC,
        ],
        "approximate_cue_relative_window_seconds": [3.0, 12.0],
        "log_variance_filter_hz": [LOW_CUT, HIGH_CUT],
        "spectral_features_source": (
            "Detrended original EEG window; Welch PSD"
        ),
        "welch_nperseg": 512,
        "welch_noverlap": 256,
        "frequency_bands_hz": BANDS,
        "participants": int(metadata["participant"].nunique()),
        "trials": int(len(metadata)),
        "class_counts": {
            str(k): int(v)
            for k, v in metadata["label"]
            .value_counts()
            .sort_index()
            .items()
        },
        "elapsed_seconds": round(time.time() - start_time, 2),
        "results": summary.to_dict(orient="records"),
        "limitations": [
            "Time-window physiological validity is not independently established.",
            "Model and feature comparisons are exploratory.",
            "LOSO scores do not establish real-world BCI performance.",
            "The best result selected from these comparisons is not an independent test.",
        ],
    }

    with open(
        OUTPUT_ROOT / "corrected_feature_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary_json, f, indent=2)

    print("\n" + "=" * 78)
    print("EXPERIMENT 134 COMPLETE")
    print("=" * 78)
    print(summary.to_string(index=False))
    print(f"\nElapsed seconds: {time.time() - start_time:.1f}")
    print(f"Outputs saved to:\n{OUTPUT_ROOT}")


# =========================================================
# 10. MAIN
# =========================================================
def main():
    print("=" * 78)
    print("EXPERIMENT 134 — CORRECTED EEG FREQUENCY FEATURES")
    print("=" * 78)
    print(f"Dataset: {DATA_ROOT}")
    print(f"Output:  {OUTPUT_ROOT}")
    print("Original dataset files will only be read.\n")

    X_features, metadata = build_dataset()

    print(f"\nTotal trials: {len(metadata)}")
    print(
        f"Unique participants: "
        f"{metadata['participant'].nunique()}"
    )
    print(
        "Class counts:\n"
        f"{metadata['label'].value_counts().sort_index()}"
    )

    evaluate(X_features, metadata)


if __name__ == "__main__":
    main()
