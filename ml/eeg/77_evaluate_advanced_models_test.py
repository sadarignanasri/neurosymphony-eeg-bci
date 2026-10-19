
"""
Step 77: Evaluate trained MLP and CNN1D models on the held-out test splits.

Project: NeuroSymphony BCI
Dataset: EEG spectral band-power features
Subjects: 01–12
Classes: 8

Important:
- Loads existing Step 76 checkpoints; does NOT retrain models.
- Uses training-set mean and standard deviation from each checkpoint.
- Evaluates the test split only.
- Checkpoints must be trusted local files because weights_only=False
  is required to load the stored NumPy preprocessing arrays.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURES_ROOT = PROJECT_ROOT / "data" / "features"
SPLITS_ROOT = PROJECT_ROOT / "data" / "splits"

CHECKPOINT_ROOT = (
    PROJECT_ROOT / "results" / "eeg" / "advanced_models" / "step76"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "eeg"
    / "advanced_models"
    / "step77_test_evaluation"
)

SUBJECTS = range(1, 13)
NUM_CLASSES = 8
NUM_FEATURES = 320

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*pin_memory.*",
)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# ============================================================
# MODEL ARCHITECTURES
# These must match the models used in Step 76.
# ============================================================

class MLP(nn.Module):
    def __init__(self, input_dim=NUM_FEATURES, num_classes=NUM_CLASSES):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.35),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.network(x)


class EEGConv1D(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.AdaptiveAvgPool1d(20),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 20, 128),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ============================================================
# PATH RESOLUTION
# Supports the subject naming variations in this project.
# ============================================================

def first_existing_path(candidates, description):
    for path in candidates:
        if path.exists():
            return path

    candidate_text = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        f"\nCould not find {description}.\n"
        f"Checked these paths:\n{candidate_text}"
    )


def get_subject_paths(subject_num):
    subject_id = f"{subject_num:02d}"

    # Subject 01 uses a slightly different directory naming pattern.
    if subject_num == 1:
        feature_candidates = [
            FEATURES_ROOT
            / "temporal_spectral_sub01"
            / "sub-01_band_power.npy",
        ]

        label_candidates = [
            FEATURES_ROOT
            / "temporal_spectral_sub01"
            / "sub-01_spectral_metadata.csv",
        ]

        split_dir_candidates = [
            SPLITS_ROOT / "sub01",
            SPLITS_ROOT / "sub-01",
        ]

        checkpoint_subject_candidates = [
            CHECKPOINT_ROOT / "sub01",
            CHECKPOINT_ROOT / "sub-01",
        ]

    else:
        feature_candidates = [
            FEATURES_ROOT
            / f"temporal_spectral_sub-{subject_id}"
            / f"sub-{subject_id}_band_power.npy",
        ]

        label_candidates = [
            FEATURES_ROOT
            / f"temporal_eeg_sub-{subject_id}"
            / f"sub-{subject_id}_temporal_eeg_preprocessed_metadata.csv",
        ]

        split_dir_candidates = [
            SPLITS_ROOT / f"sub-{subject_id}",
            SPLITS_ROOT / f"sub{subject_id}",
        ]

        checkpoint_subject_candidates = [
            CHECKPOINT_ROOT / f"sub-{subject_id}",
            CHECKPOINT_ROOT / f"sub{subject_id}",
        ]

    feature_path = first_existing_path(
        feature_candidates, f"features for subject {subject_id}"
    )

    label_path = first_existing_path(
        label_candidates, f"metadata labels for subject {subject_id}"
    )

    split_dir = first_existing_path(
        split_dir_candidates, f"split directory for subject {subject_id}"
    )

    checkpoint_subject_dir = first_existing_path(
        checkpoint_subject_candidates,
        f"checkpoint directory for subject {subject_id}",
    )

    split_paths = {
        "train": split_dir / "train_indices.npy",
        "val": split_dir / "val_indices.npy",
        "test": split_dir / "test_indices.npy",
    }

    for split_name, path in split_paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {split_name} split for subject {subject_id}: {path}"
            )

    return {
        "subject_id": subject_id,
        "features": feature_path,
        "labels": label_path,
        "splits": split_paths,
        "checkpoint_dir": checkpoint_subject_dir,
    }


# ============================================================
# LABEL LOADING
# ============================================================

def load_labels(metadata_path):
    metadata = pd.read_csv(metadata_path)

    possible_columns = [
        "label",
        "label_id",
        "class_label",
        "class_id",
    ]

    label_column = next(
        (column for column in possible_columns if column in metadata.columns),
        None,
    )

    if label_column is None:
        raise ValueError(
            f"No recognized label column in {metadata_path}.\n"
            f"Available columns: {metadata.columns.tolist()}"
        )

    labels = metadata[label_column].to_numpy()

    # Labels must be numeric class IDs from 0 to 7.
    try:
        labels = labels.astype(np.int64)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Labels in {metadata_path} are not numeric class IDs. "
            f"Found column: {label_column}"
        ) from exc

    unique_labels = set(np.unique(labels).tolist())

    if not unique_labels.issubset(set(range(NUM_CLASSES))):
        raise ValueError(
            f"Unexpected labels in {metadata_path}: "
            f"{sorted(unique_labels)}. Expected IDs 0–7."
        )

    return labels, label_column


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_checkpoint(checkpoint_path):
    # weights_only=False is intentional because Step 76 checkpoints
    # contain NumPy preprocessing arrays as well as model weights.
    # Only load checkpoints created by your trusted local training script.
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Unexpected checkpoint format: {checkpoint_path}"
        )

    return checkpoint


def get_checkpoint_value(checkpoint, possible_keys):
    for key in possible_keys:
        if key in checkpoint:
            return checkpoint[key]

    return None


def to_numpy_1d(value, name, checkpoint_path):
    if value is None:
        raise KeyError(
            f"Checkpoint is missing {name}: {checkpoint_path}\n"
            "The training mean and standard deviation must be recovered "
            "from the Step 76 checkpoint or training artifacts."
        )

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    array = np.asarray(value, dtype=np.float32).reshape(-1)

    if array.size != NUM_FEATURES:
        raise ValueError(
            f"{name} has {array.size} values in {checkpoint_path}; "
            f"expected {NUM_FEATURES}."
        )

    return array


def extract_model_state(checkpoint, checkpoint_path):
    state = get_checkpoint_value(
        checkpoint,
        [
            "model_state_dict",
            "state_dict",
            "model_state",
            "model",
        ],
    )

    # Some checkpoints may contain only a raw state dictionary.
    if state is None and checkpoint and all(
        isinstance(value, torch.Tensor)
        for value in checkpoint.values()
    ):
        state = checkpoint

    if not isinstance(state, dict):
        raise KeyError(
            f"Could not find model weights in {checkpoint_path}. "
            f"Available checkpoint keys: {list(checkpoint.keys())}"
        )

    return state


def build_model(model_name):
    if model_name == "MLP":
        return MLP()

    if model_name == "CNN1D":
        return EEGConv1D()

    raise ValueError(f"Unknown model: {model_name}")


def load_trained_model(model_name, checkpoint_path):
    checkpoint = load_checkpoint(checkpoint_path)

    model = build_model(model_name)
    state_dict = extract_model_state(checkpoint, checkpoint_path)

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    mean = get_checkpoint_value(
        checkpoint,
        [
            "feature_mean",
            "train_mean",
            "mean",
            "scaler_mean",
            "X_mean",
        ],
    )

    std = get_checkpoint_value(
        checkpoint,
        [
            "feature_std",
            "train_std",
            "std",
            "scaler_std",
            "X_std",
        ],
    )

    mean = to_numpy_1d(mean, "training feature mean", checkpoint_path)
    std = to_numpy_1d(std, "training feature standard deviation", checkpoint_path)

    # Protect against zero standard deviations.
    std = np.where(std < 1e-8, 1.0, std)

    return model, mean, std


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=list(range(NUM_CLASSES)),
            average="macro",
            zero_division=0,
        ),
    }


# ============================================================
# TEST EVALUATION
# ============================================================

def evaluate_model(
    model,
    model_name,
    features,
    labels,
    test_indices,
    feature_mean,
    feature_std,
):
    X_test = features[test_indices].reshape(len(test_indices), -1)
    y_test = labels[test_indices]

    if X_test.shape[1] != NUM_FEATURES:
        raise ValueError(
            f"Expected {NUM_FEATURES} features, got {X_test.shape[1]}."
        )

    # Apply training-only normalization parameters from the checkpoint.
    X_test = (X_test.astype(np.float32) - feature_mean) / feature_std
    X_test = np.nan_to_num(
        X_test,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    X_tensor = torch.from_numpy(X_test).to(DEVICE)

    if model_name == "CNN1D":
        X_tensor = X_tensor.unsqueeze(1)

    with torch.no_grad():
        logits = model(X_tensor)
        predictions = torch.argmax(logits, dim=1).cpu().numpy()

    metrics = calculate_metrics(y_test, predictions)

    return metrics, y_test, predictions


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Device: {DEVICE}")
    print("Evaluating existing Step 76 checkpoints on the test splits.")
    print("No model retraining or test-set tuning will be performed.\n")

    results = []

    for subject_num in SUBJECTS:
        paths = get_subject_paths(subject_num)
        subject_id = paths["subject_id"]

        print(f"\n{'=' * 60}")
        print(f"Subject {subject_id}")
        print(f"{'=' * 60}")

        features = np.load(paths["features"])
        labels, label_column = load_labels(paths["labels"])
        test_indices = np.load(paths["splits"]["test"]).astype(np.int64)

        if features.shape[0] != len(labels):
            raise ValueError(
                f"Subject {subject_id}: feature rows ({features.shape[0]}) "
                f"do not match label rows ({len(labels)})."
            )

        if features.ndim != 3:
            raise ValueError(
                f"Subject {subject_id}: expected features with shape "
                f"(trials, channels, bands), got {features.shape}."
            )

        if features.shape[1] * features.shape[2] != NUM_FEATURES:
            raise ValueError(
                f"Subject {subject_id}: feature shape {features.shape} "
                f"does not flatten to {NUM_FEATURES} features."
            )

        if len(test_indices) == 0:
            raise ValueError(f"Subject {subject_id}: test split is empty.")

        if np.any(test_indices < 0) or np.any(test_indices >= len(labels)):
            raise IndexError(
                f"Subject {subject_id}: test split contains invalid indices."
            )

        print(f"Features: {features.shape}")
        print(f"Labels column: {label_column}")
        print(f"Test trials: {len(test_indices)}")

        for model_name in ["MLP", "CNN1D"]:
            checkpoint_path = (
                paths["checkpoint_dir"]
                / model_name
                / "best_model.pt"
            )

            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Missing Step 76 checkpoint: {checkpoint_path}"
                )

            model, feature_mean, feature_std = load_trained_model(
                model_name,
                checkpoint_path,
            )

            metrics, y_true, y_pred = evaluate_model(
                model=model,
                model_name=model_name,
                features=features,
                labels=labels,
                test_indices=test_indices,
                feature_mean=feature_mean,
                feature_std=feature_std,
            )

            row = {
                "subject": subject_id,
                "model": model_name,
                "n_test_trials": len(test_indices),
                **metrics,
                "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
            }

            results.append(row)

            print(
                f"{model_name}: "
                f"Accuracy={metrics['accuracy']:.4f} | "
                f"Balanced Accuracy={metrics['balanced_accuracy']:.4f} | "
                f"Macro-F1={metrics['macro_f1']:.4f}"
            )

            # Save per-subject predictions for transparent reporting.
            prediction_dir = OUTPUT_ROOT / f"sub-{subject_id}"
            prediction_dir.mkdir(parents=True, exist_ok=True)

            prediction_df = pd.DataFrame({
                "test_trial_index": test_indices,
                "true_label": y_true,
                "predicted_label": y_pred,
            })

            prediction_df.to_csv(
                prediction_dir / f"{model_name.lower()}_predictions.csv",
                index=False,
            )

            del model
            if DEVICE.type == "mps":
                torch.mps.empty_cache()

    # Per-subject metrics.
    summary_df = pd.DataFrame(results)
    summary_path = OUTPUT_ROOT / "test_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # Aggregate subject-level means and standard deviations.
    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
    ]

    aggregate_df = (
        summary_df.groupby("model")[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Flatten multi-level column names for a clean CSV.
    aggregate_df.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in aggregate_df.columns
    ]

    aggregate_path = OUTPUT_ROOT / "test_aggregate.csv"
    aggregate_df.to_csv(aggregate_path, index=False)

    # Save a compact JSON record of the evaluation configuration.
    configuration = {
        "step": 77,
        "device": str(DEVICE),
        "subjects_evaluated": len(SUBJECTS),
        "models": ["MLP", "CNN1D"],
        "num_classes": NUM_CLASSES,
        "features_per_trial": NUM_FEATURES,
        "evaluation_split": "test",
        "training_performed": False,
        "test_set_used_for_tuning": False,
        "note": (
            "Conventional-model test results were examined earlier in this "
            "project, so the test set is not a pristine untouched benchmark."
        ),
    }

    with open(OUTPUT_ROOT / "evaluation_config.json", "w") as file:
        json.dump(configuration, file, indent=4)

    print(f"\n{'=' * 60}")
    print("STEP 77 COMPLETED")
    print(f"{'=' * 60}")

    print(f"\nPer-subject results:\n{summary_df.to_string(index=False)}")
    print(f"\nAggregate results:\n{aggregate_df.to_string(index=False)}")

    print(f"\nSaved per-subject results: {summary_path}")
    print(f"Saved aggregate results:   {aggregate_path}")
    print(f"Saved predictions under:   {OUTPUT_ROOT}")
    print("\nNo models were retrained.")


if __name__ == "__main__":
    main()
