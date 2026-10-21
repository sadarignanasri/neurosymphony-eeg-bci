
"""
Experiment 140: Dataset 3 — CSP + RBF-SVM with LOSO validation.

Dataset:
    major datasets/28955456/epo_data

Protocol:
    - Original MAT files are read-only.
    - 8–30 Hz Butterworth bandpass filter.
    - Epoch-relative window: 13–22 seconds.
    - CSP is fitted only on training subjects in each fold.
    - RBF-SVM classifier.
    - Leave-One-Subject-Out (LOSO) evaluation.
    - Results saved in a separate Experiment 140 directory.

Labels:
    0 = Left motor imagery
    1 = Right motor imagery
"""

from pathlib import Path
import inspect
import json
import time
import warnings

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from mne.decoding import CSP

warnings.filterwarnings("default")


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(r"D:\neurosymphony-eeg-bci")

DATA_ROOT = ROOT / "major datasets" / "28955456" / "epo_data"

OUTPUT_DIR = (
    ROOT
    / "experiments"
    / "dataset3_28955456_training"
    / "experiment140_csp_svm"
)

FS = 256

WINDOW_START_SECONDS = 13.0
WINDOW_END_SECONDS = 22.0

FILTER_LOW_HZ = 8.0
FILTER_HIGH_HZ = 30.0
FILTER_ORDER = 4

CSP_COMPONENTS = 6
CSP_REGULARIZATION = "ledoit_wolf"

SVM_C = 1.0
SVM_GAMMA = "scale"

RANDOM_STATE = 42


# ============================================================
# 2. LOAD MATLAB EPOCH FILE
# ============================================================

def load_epoch_file(mat_path: Path):
    """Load one MATLAB epo structure and return X, y, sampling rate."""

    mat = loadmat(
        mat_path,
        struct_as_record=True,
        squeeze_me=False,
    )

    if "epo" not in mat:
        raise ValueError(f"'epo' structure not found in {mat_path}")

    epo = mat["epo"][0, 0]

    X_raw = np.asarray(epo["X"], dtype=np.float64)
    y = np.asarray(epo["y"]).reshape(-1).astype(int)

    fs_value = np.asarray(epo["fs"]).reshape(-1)
    fs = float(fs_value[0])

    if X_raw.ndim != 3:
        raise ValueError(
            f"Expected 3D EEG data in {mat_path}; "
            f"received shape {X_raw.shape}"
        )

    # Dataset normally stores X as:
    # samples x channels x trials
    #
    # MNE CSP requires:
    # trials x channels x samples
    if X_raw.shape[2] == len(y):
        X = np.transpose(X_raw, (2, 1, 0))

    elif X_raw.shape[0] == len(y):
        # Defensive fallback if trials are already the first dimension.
        X = X_raw

    else:
        raise ValueError(
            f"Cannot match trial count in {mat_path}: "
            f"X={X_raw.shape}, labels={len(y)}"
        )

    if X.shape[0] != len(y):
        raise ValueError(
            f"Trial/label mismatch in {mat_path}: "
            f"X={X.shape[0]}, y={len(y)}"
        )

    if not np.isclose(fs, FS):
        raise ValueError(
            f"Unexpected sampling frequency {fs} Hz in {mat_path}; "
            f"expected {FS} Hz."
        )

    if not np.isfinite(X).all():
        raise ValueError(f"Non-finite EEG values found in {mat_path}")

    if set(np.unique(y)) != {0, 1}:
        raise ValueError(
            f"Expected labels 0 and 1 in {mat_path}; "
            f"found {np.unique(y).tolist()}"
        )

    return X, y, fs


# ============================================================
# 3. FILTER AND SELECT THE WINDOW
# ============================================================

def preprocess_epochs(X):
    """
    Filter each complete epoch and then select the epoch-relative
    13–22 second window.

    Input:
        trials x channels x samples

    Output:
        trials x channels x window_samples
    """

    start_sample = int(round(WINDOW_START_SECONDS * FS))
    end_sample = int(round(WINDOW_END_SECONDS * FS))

    if X.shape[-1] < end_sample:
        raise ValueError(
            f"Epoch is too short: {X.shape[-1]} samples; "
            f"need at least {end_sample}."
        )

    sos = butter(
        FILTER_ORDER,
        [FILTER_LOW_HZ, FILTER_HIGH_HZ],
        btype="bandpass",
        fs=FS,
        output="sos",
    )

    # Filter the full epoch before selecting the window.
    filtered = sosfiltfilt(sos, X, axis=-1)

    window = filtered[:, :, start_sample:end_sample]

    return np.ascontiguousarray(window, dtype=np.float64)


# ============================================================
# 4. DISCOVER FILES AND IDENTIFY PARTICIPANTS
# ============================================================

def discover_files():
    """Discover EEG epoch MAT files in Normals and Patients."""

    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {DATA_ROOT}"
        )

    files = sorted(
        path
        for path in DATA_ROOT.rglob("*_epo.mat")
        if path.is_file()
        and "EEG" in path.parts
        and (
            "Normals" in path.parts
            or "Patients" in path.parts
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No *_epo.mat files found under {DATA_ROOT}"
        )

    return files


def get_group_and_participant(path: Path):
    """
    Extract group and participant ID.

    IMPORTANT:
    subject1 under Normals and subject1 under Patients must be
    treated as different participants.

    IDs therefore look like:
        Normals_subject1
        Patients_subject1
    """

    if "Normals" in path.parts:
        group = "Normals"
    elif "Patients" in path.parts:
        group = "Patients"
    else:
        raise ValueError(f"Unknown group in path: {path}")

    subject_dirs = [
        part
        for part in path.parts
        if part.lower().startswith("subject")
    ]

    if not subject_dirs:
        raise ValueError(
            f"Could not identify subject directory: {path}"
        )

    subject_name = subject_dirs[-1]

    # Group-qualified ID prevents subject1 from the two groups
    # from being incorrectly merged into a single person.
    participant = f"{group}_{subject_name}"

    return group, participant


# ============================================================
# 5. LOAD ALL DATA
# ============================================================

def load_all_data():
    all_X = []
    all_y = []
    all_participants = []
    all_groups = []
    inventory = []

    files = discover_files()

    print(f"Dataset directory : {DATA_ROOT}")
    print(f"Epoch files found : {len(files)}")
    print("Loading and preprocessing EEG files...", flush=True)

    for file_index, path in enumerate(files, start=1):
        group, participant = get_group_and_participant(path)

        X, y, fs = load_epoch_file(path)
        X = preprocess_epochs(X)

        all_X.append(X)
        all_y.append(y)

        all_participants.extend([participant] * len(y))
        all_groups.extend([group] * len(y))

        inventory.append({
            "file": str(path),
            "group": group,
            "participant": participant,
            "trials": int(len(y)),
            "channels": int(X.shape[1]),
            "window_samples": int(X.shape[2]),
            "sampling_frequency_hz": fs,
            "class_0_trials": int(np.sum(y == 0)),
            "class_1_trials": int(np.sum(y == 1)),
        })

        print(
            f"[{file_index:03d}/{len(files):03d}] "
            f"{participant}: {len(y)} trials, "
            f"shape={X.shape}",
            flush=True,
        )

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)

    participant_ids = np.asarray(all_participants)
    group_ids = np.asarray(all_groups)

    if not np.isfinite(X_all).all():
        raise ValueError(
            "Non-finite values detected after preprocessing."
        )

    if set(np.unique(y_all)) != {0, 1}:
        raise ValueError(
            f"Unexpected combined labels: {np.unique(y_all)}"
        )

    inventory_df = pd.DataFrame(inventory)

    # Check that each participant belongs to exactly one group.
    participant_group_counts = (
        inventory_df[["participant", "group"]]
        .drop_duplicates()
        .groupby("participant")["group"]
        .nunique()
    )

    if (participant_group_counts > 1).any():
        raise ValueError(
            "Participant ID collision detected across groups."
        )

    print("\nParticipant counts by group:")
    print(
        inventory_df[["participant", "group"]]
        .drop_duplicates()["group"]
        .value_counts()
        .to_string()
    )

    print(
        "\nUnique participants:",
        len(np.unique(participant_ids)),
    )

    print(
        "Unique participant IDs:",
        sorted(np.unique(participant_ids).tolist()),
    )

    return (
        X_all,
        y_all,
        participant_ids,
        group_ids,
        inventory_df,
    )


# ============================================================
# 6. BUILD A VERSION-COMPATIBLE CSP INSTANCE
# ============================================================

def make_csp():
    """
    Create an MNE CSP instance compatible with the installed
    MNE version by passing only supported constructor arguments.
    """

    requested_params = {
        "n_components": CSP_COMPONENTS,
        "reg": CSP_REGULARIZATION,
        "log": True,
        "norm_trace": False,
        "cov_est": "concat",
        "transform_into": "average_power",
        "component_order": "mutual_info",
        "rank": "full",
    }

    supported_params = inspect.signature(CSP).parameters

    compatible_params = {
        key: value
        for key, value in requested_params.items()
        if key in supported_params
    }

    missing_core = [
        name
        for name in ("n_components", "reg", "log", "norm_trace")
        if name not in compatible_params
    ]

    if missing_core:
        raise RuntimeError(
            "Installed MNE CSP API lacks required parameters: "
            f"{missing_core}. Please share your MNE version."
        )

    print(
        "CSP parameters supported by installed MNE:",
        sorted(compatible_params.keys()),
        flush=True,
    )

    return CSP(**compatible_params)


# ============================================================
# 7. LOSO TRAINING AND EVALUATION
# ============================================================

def evaluate_loso(X, y, participant_ids, group_ids):
    """
    Fit CSP + StandardScaler + RBF-SVM with LOSO.

    CSP is part of the sklearn Pipeline. Therefore, it is fitted
    only on training subjects in each fold.
    """

    participants = np.unique(participant_ids)

    if len(participants) < 2:
        raise ValueError(
            "LOSO requires at least two participants."
        )

    csp_template = make_csp()

    model_template = Pipeline([
        ("csp", csp_template),
        ("scaler", StandardScaler()),
        (
            "svm",
            SVC(
                kernel="rbf",
                C=SVM_C,
                gamma=SVM_GAMMA,
                class_weight=None,
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    oof_rows = []
    fold_rows = []

    start_time = time.time()

    for fold_number, held_out in enumerate(participants, start=1):
        test_mask = participant_ids == held_out
        train_mask = ~test_mask

        X_train = X[train_mask]
        y_train = y[train_mask]

        X_test = X[test_mask]
        y_test = y[test_mask]

        training_participants = np.unique(
            participant_ids[train_mask]
        )

        if len(np.unique(y_train)) != 2:
            raise ValueError(
                f"Training fold for {held_out} does not contain "
                "both classes."
            )

        if len(np.unique(y_test)) != 2:
            warnings.warn(
                f"Held-out participant {held_out} does not contain "
                "both classes. Interpret balanced accuracy and "
                "macro F1 carefully."
            )

        print(
            f"\nFold {fold_number}/{len(participants)} | "
            f"held-out={held_out} | "
            f"train trials={len(y_train)} | "
            f"test trials={len(y_test)} | "
            f"training subjects={len(training_participants)}",
            flush=True,
        )

        model = clone(model_template)

        fold_start = time.time()

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        fold_seconds = time.time() - fold_start

        fold_acc = accuracy_score(y_test, predictions)
        fold_bal_acc = balanced_accuracy_score(
            y_test, predictions
        )
        fold_f1 = f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        )

        test_group = group_ids[test_mask]
        unique_groups = np.unique(test_group)

        group_label = (
            unique_groups[0]
            if len(unique_groups) == 1
            else "Mixed"
        )

        fold_rows.append({
            "participant": held_out,
            "group": group_label,
            "trials": int(len(y_test)),
            "accuracy": float(fold_acc),
            "balanced_accuracy": float(fold_bal_acc),
            "macro_f1": float(fold_f1),
            "fit_predict_seconds": float(fold_seconds),
        })

        for i in range(len(y_test)):
            oof_rows.append({
                "participant": held_out,
                "group": test_group[i],
                "y_true": int(y_test[i]),
                "y_pred": int(predictions[i]),
                "correct": bool(y_test[i] == predictions[i]),
            })

        print(
            f"  Accuracy={fold_acc:.4f} | "
            f"Balanced accuracy={fold_bal_acc:.4f} | "
            f"Macro F1={fold_f1:.4f} | "
            f"Time={fold_seconds:.1f}s",
            flush=True,
        )

    total_seconds = time.time() - start_time

    folds = pd.DataFrame(fold_rows)
    oof = pd.DataFrame(oof_rows)

    # Every trial must be tested exactly once.
    if len(oof) != len(y):
        raise RuntimeError(
            f"OOF coverage mismatch: expected {len(y)}, "
            f"received {len(oof)}."
        )

    if oof["participant"].nunique() != len(participants):
        raise RuntimeError(
            "OOF predictions do not cover every participant."
        )

    pooled_accuracy = accuracy_score(
        oof["y_true"], oof["y_pred"]
    )

    pooled_balanced_accuracy = balanced_accuracy_score(
        oof["y_true"], oof["y_pred"]
    )

    pooled_macro_f1 = f1_score(
        oof["y_true"],
        oof["y_pred"],
        average="macro",
        zero_division=0,
    )

    cm = confusion_matrix(
        oof["y_true"],
        oof["y_pred"],
        labels=[0, 1],
    )

    summary = {
        "experiment": 140,
        "dataset": "28955456",
        "task": "binary_left_vs_right_motor_imagery",
        "model": "CSP + StandardScaler + RBF-SVM",
        "evaluation": "Leave-One-Subject-Out",
        "n_files": int(len(discover_files())),
        "n_participants": int(len(participants)),
        "n_trials": int(len(y)),
        "n_channels": int(X.shape[1]),
        "sampling_frequency_hz": FS,
        "filter_hz": [FILTER_LOW_HZ, FILTER_HIGH_HZ],
        "filter_order": FILTER_ORDER,
        "epoch_relative_window_seconds": [
            WINDOW_START_SECONDS,
            WINDOW_END_SECONDS,
        ],
        "window_samples": int(X.shape[-1]),
        "csp_components": CSP_COMPONENTS,
        "csp_regularization": CSP_REGULARIZATION,
        "svm_kernel": "rbf",
        "svm_C": SVM_C,
        "svm_gamma": SVM_GAMMA,
        "pooled_accuracy": float(pooled_accuracy),
        "pooled_balanced_accuracy": float(
            pooled_balanced_accuracy
        ),
        "pooled_macro_f1": float(pooled_macro_f1),
        "mean_participant_accuracy": float(
            folds["accuracy"].mean()
        ),
        "std_participant_accuracy": float(
            folds["accuracy"].std(ddof=1)
        ),
        "mean_participant_balanced_accuracy": float(
            folds["balanced_accuracy"].mean()
        ),
        "mean_participant_macro_f1": float(
            folds["macro_f1"].mean()
        ),
        "confusion_matrix_labels_0_1": cm.tolist(),
        "total_training_evaluation_seconds": float(
            total_seconds
        ),
        "random_state": RANDOM_STATE,
        "notes": [
            "CSP is fitted inside each LOSO training fold.",
            "Participant IDs include group names to avoid collisions.",
            "Original dataset files are read-only.",
            "Results are exploratory because participants were reused "
            "during prior model selection.",
            "The selected time window has not been independently "
            "validated physiologically.",
        ],
    }

    return folds, oof, summary


# ============================================================
# 8. SAVE RESULTS
# ============================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("EXPERIMENT 140: DATASET 3 — CSP + RBF-SVM")
    print("=" * 70)

    (
        X,
        y,
        participant_ids,
        group_ids,
        inventory,
    ) = load_all_data()

    n_participants = len(np.unique(participant_ids))

    print("\nDataset summary")
    print("-" * 50)
    print(f"Trials       : {len(y)}")
    print(f"Participants : {n_participants}")
    print(f"Channels     : {X.shape[1]}")
    print(f"Window shape : {X.shape}")
    print(
        "Label counts :",
        {
            int(label): int(count)
            for label, count in zip(
                *np.unique(y, return_counts=True)
            )
        },
    )

    # Save inventory before training so the audit remains available
    # even if a later training fold fails.
    inventory_path = OUTPUT_DIR / "file_inventory.csv"
    inventory.to_csv(inventory_path, index=False)

    folds, oof, summary = evaluate_loso(
        X,
        y,
        participant_ids,
        group_ids,
    )

    folds_path = OUTPUT_DIR / "participant_fold_metrics.csv"
    oof_path = OUTPUT_DIR / "out_of_fold_predictions.csv"
    summary_path = OUTPUT_DIR / "csp_svm_summary.json"
    report_path = OUTPUT_DIR / "classification_report.txt"

    folds.to_csv(folds_path, index=False)
    oof.to_csv(oof_path, index=False)

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    report = classification_report(
        oof["y_true"],
        oof["y_pred"],
        labels=[0, 1],
        target_names=[
            "Left imagery (0)",
            "Right imagery (1)",
        ],
        zero_division=0,
        digits=4,
    )

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report)
        file.write("\n\nConfusion matrix (labels 0, 1):\n")
        file.write(
            str(
                confusion_matrix(
                    oof["y_true"],
                    oof["y_pred"],
                    labels=[0, 1],
                )
            )
        )
        file.write("\n")

    print("\n" + "=" * 70)
    print("EXPERIMENT 140 COMPLETE")
    print("=" * 70)

    print(
        f"Pooled accuracy           : "
        f"{summary['pooled_accuracy']:.4%}"
    )
    print(
        f"Pooled balanced accuracy  : "
        f"{summary['pooled_balanced_accuracy']:.4%}"
    )
    print(
        f"Pooled macro F1            : "
        f"{summary['pooled_macro_f1']:.4%}"
    )
    print(
        f"Mean participant accuracy : "
        f"{summary['mean_participant_accuracy']:.4%}"
    )
    print(
        f"Mean participant macro F1 : "
        f"{summary['mean_participant_macro_f1']:.4%}"
    )
    print(
        f"Total runtime              : "
        f"{summary['total_training_evaluation_seconds']:.1f}s"
    )

    print(
        "\nConfusion matrix "
        "(rows=true, columns=predicted; labels 0, 1):"
    )
    print(
        confusion_matrix(
            oof["y_true"],
            oof["y_pred"],
            labels=[0, 1],
        )
    )

    print("\nClassification report:")
    print(report)

    print("\nSaved outputs:")
    print(f"  {inventory_path}")
    print(f"  {folds_path}")
    print(f"  {oof_path}")
    print(f"  {summary_path}")
    print(f"  {report_path}")


if __name__ == "__main__":
    main()
