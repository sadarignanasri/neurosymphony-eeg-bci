"""
NEURO-SYMPHONY
EXPERIMENT 27 — CHANNEL-DOMAIN FFT MAGNITUDE + TRANSFORMER

IMPORTANT:
The approved CSV contains one trial-level value for each of the
64 EEG channels. It does NOT contain temporal EEG samples.

Therefore:
    FFT is applied across the 64 channel values.

This is CHANNEL-DOMAIN FFT.

It is NOT temporal EEG frequency analysis.

True temporal FFT / PSD / delta-theta-alpha-beta-gamma analysis
requires the original temporal EEG waveform.

Pipeline:

CSV
    ↓
64 EEG channel values
    ↓
rFFT across channel dimension
    ↓
33 complex FFT bins
    ↓
magnitude
    ↓
training-only normalization
    ↓
Transformer
    ↓
8-class classification

Input to Transformer:
    [batch, 33, 1]

33 = rFFT bins from 64 input values
1  = magnitude feature per FFT bin
"""

from pathlib import Path
import json
import time
import random

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ds004514_eeg_approved.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "eeg"
    / "fft_magnitude_transformer"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Dataset
NUM_CLASSES = 8
NUM_CHANNELS = 64

# rFFT of 64 real values gives:
# 64 // 2 + 1 = 33 bins
NUM_FFT_BINS = NUM_CHANNELS // 2 + 1

EEG_CHANNELS = (
    [f"A{i}" for i in range(1, 33)]
    + [f"B{i}" for i in range(1, 33)]
)

CLASS_NAMES = [
    "animal_tactile_imagery_task",
    "animal_visual_imagery_task",
    "animal_auditory_imagery_task",
    "animal_silent_naming_task",
    "tool_tactile_imagery_task",
    "tool_visual_imagery_task",
    "tool_auditory_imagery_task",
    "tool_silent_naming_task",
]


# Transformer
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
FF_DIM = 128
DROPOUT = 0.20

# Training
BATCH_SIZE = 32
MAX_EPOCHS = 50
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 8

RANDOM_SEED = 42


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()


# =============================================================================
# DEVICE
# =============================================================================

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")

        print("=" * 80)
        print("GPU DETECTED")
        print("=" * 80)
        print("CUDA available:", torch.cuda.is_available())
        print("GPU:", torch.cuda.get_device_name(0))
        print()

        return device

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")

        print("=" * 80)
        print("APPLE MPS DEVICE DETECTED")
        print("=" * 80)
        print("Device:", device)
        print()

        return device

    device = torch.device("cpu")

    print("=" * 80)
    print("CPU DEVICE")
    print("=" * 80)
    print()

    return device


DEVICE = get_device()


# =============================================================================
# DATASET VALIDATION
# =============================================================================

def validate_dataset(df: pd.DataFrame):

    print("=" * 80)
    print("DATASET VALIDATION")
    print("=" * 80)

    required_columns = (
        ["subject_id", "trial_id"]
        + EEG_CHANNELS
        + ["label_name", "label"]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print("Dataset shape:", df.shape)

    # -------------------------------------------------------------------------
    # EEG columns
    # -------------------------------------------------------------------------

    if len(EEG_CHANNELS) != 64:
        raise ValueError("Expected exactly 64 EEG channels.")

    print("EEG channels:", len(EEG_CHANNELS))

    eeg_values = df[EEG_CHANNELS].to_numpy(dtype=np.float64)

    if np.isnan(eeg_values).any():
        raise ValueError("NaN values detected in EEG data.")

    if np.isinf(eeg_values).any():
        raise ValueError("Infinite values detected in EEG data.")

    print("✓ No missing EEG values")
    print("✓ No infinite EEG values")

    # -------------------------------------------------------------------------
    # Subjects
    # -------------------------------------------------------------------------

    subjects = sorted(df["subject_id"].astype(str).unique())

    print("Subjects:", len(subjects))

    if len(subjects) != 12:
        raise ValueError(
            f"Expected 12 subjects, found {len(subjects)}."
        )

    # -------------------------------------------------------------------------
    # Classes
    # -------------------------------------------------------------------------

    labels = sorted(df["label"].unique())

    print("Classes:", len(labels))
    print("Labels:", labels)

    if labels != list(range(1, 9)):
        raise ValueError(
            f"Expected labels [1..8], found {labels}"
        )

    # -------------------------------------------------------------------------
    # Duplicate subject/trial
    # -------------------------------------------------------------------------

    duplicate_count = df.duplicated(
        subset=["subject_id", "trial_id"]
    ).sum()

    if duplicate_count != 0:
        raise ValueError(
            f"Found {duplicate_count} duplicate subject/trial rows."
        )

    print("✓ No duplicate subject/trial keys")

    # -------------------------------------------------------------------------
    # Subject counts
    # -------------------------------------------------------------------------

    subject_counts = (
        df.groupby("subject_id")
        .size()
        .sort_index()
    )

    print()
    print("Trials per subject:")
    print(subject_counts.to_string())

    if not (subject_counts == 720).all():
        raise ValueError(
            "Each subject must contain exactly 720 trials."
        )

    # -------------------------------------------------------------------------
    # Class counts
    # -------------------------------------------------------------------------

    class_counts = (
        df.groupby("label")
        .size()
        .sort_index()
    )

    print()
    print("Trials per class:")
    print(class_counts.to_string())

    if not (class_counts == 1080).all():
        raise ValueError(
            "Each class must contain exactly 1080 trials."
        )

    print()
    print("✓ DATASET VALIDATION PASSED")
    print()


# =============================================================================
# FFT FEATURE EXTRACTION
# =============================================================================

def extract_fft_magnitude(df: pd.DataFrame):

    print("=" * 80)
    print("FFT FEATURE EXTRACTION")
    print("=" * 80)

    X = df[EEG_CHANNELS].to_numpy(
        dtype=np.float64
    )

    print("Original EEG feature shape:")
    print(" ", X.shape)

    print()
    print("Applying rFFT across the 64 EEG channel values...")

    # IMPORTANT:
    # Keep the complex FFT representation first.
    #
    # Shape:
    #   [trials, 64]
    #
    # becomes:
    #
    #   [trials, 33]
    #
    # because rFFT(64) -> 33 bins.

    fft_complex = np.fft.rfft(
        X,
        axis=1
    )

    print("Complex FFT shape:")
    print(" ", fft_complex.shape)

    # Magnitude from complex FFT.
    fft_magnitude = np.abs(
        fft_complex
    ).astype(np.float32)

    print()
    print("FFT magnitude shape:")
    print(" ", fft_magnitude.shape)

    expected_shape = (
        len(df),
        NUM_FFT_BINS
    )

    if fft_magnitude.shape != expected_shape:
        raise ValueError(
            f"Unexpected FFT feature shape. "
            f"Expected {expected_shape}, "
            f"got {fft_magnitude.shape}"
        )

    print()
    print("Expected FFT bins:", NUM_FFT_BINS)
    print("Actual FFT bins:", fft_magnitude.shape[1])

    print()
    print("IMPORTANT:")
    print(
        "These 33 bins are CHANNEL-DOMAIN FFT bins."
    )
    print(
        "They are NOT frequency bands in Hz."
    )

    print()
    print("✓ FFT MAGNITUDE EXTRACTION PASSED")
    print()

    return fft_complex, fft_magnitude


# =============================================================================
# SINUSOIDAL POSITIONAL ENCODING
# =============================================================================

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model: int,
        max_len: int,
    ):
        super().__init__()

        position = torch.arange(
            0,
            max_len,
            dtype=torch.float32
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float32
            )
            * (-np.log(10000.0) / d_model)
        )

        pe = torch.zeros(
            max_len,
            d_model
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        pe = pe.unsqueeze(0)

        self.register_buffer(
            "pe",
            pe
        )

    def forward(self, x):

        sequence_length = x.size(1)

        return x + self.pe[:, :sequence_length]


# =============================================================================
# TRANSFORMER MODEL
# =============================================================================

class FFTMagnitudeTransformer(nn.Module):

    def __init__(
        self,
        num_fft_bins: int = NUM_FFT_BINS,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
        ff_dim: int = FF_DIM,
        dropout: float = DROPOUT,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()

        # Each FFT bin contains one magnitude value.
        #
        # [batch, 33, 1]
        #
        # Project:
        #
        # [batch, 33, 1]
        #       ↓
        # [batch, 33, 64]

        self.input_projection = nn.Linear(
            1,
            d_model
        )

        self.position_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=num_fft_bins
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        self.norm = nn.LayerNorm(
            d_model
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.classifier = nn.Linear(
            d_model,
            num_classes
        )

    def forward(self, x):

        # x:
        # [batch, 33, 1]

        x = self.input_projection(x)

        # [batch, 33, 64]

        x = self.position_encoding(x)

        x = self.transformer(x)

        # Mean pooling over FFT-bin tokens

        x = x.mean(dim=1)

        # [batch, 64]

        x = self.norm(x)

        x = self.dropout(x)

        logits = self.classifier(x)

        # [batch, 8]

        return logits


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_features(
    fft_magnitude,
    train_indices,
    val_indices,
    test_indices,
):
    """
    Normalize using TRAINING data only.

    This prevents information leakage from validation/test subjects.
    """

    train_features = fft_magnitude[train_indices]
    val_features = fft_magnitude[val_indices]
    test_features = fft_magnitude[test_indices]

    mean = train_features.mean(
        axis=0,
        keepdims=True
    )

    std = train_features.std(
        axis=0,
        keepdims=True
    )

    std[std < 1e-8] = 1.0

    train_norm = (
        train_features - mean
    ) / std

    val_norm = (
        val_features - mean
    ) / std

    test_norm = (
        test_features - mean
    ) / std

    return (
        train_norm.astype(np.float32),
        val_norm.astype(np.float32),
        test_norm.astype(np.float32),
        mean.astype(np.float32),
        std.astype(np.float32),
    )


# =============================================================================
# DATA LOADERS
# =============================================================================

def make_loader(
    features,
    labels,
    shuffle,
):
    """
    Convert:

        [samples, 33]

    into:

        [samples, 33, 1]

    because every FFT token contains one magnitude feature.
    """

    X = torch.tensor(
        features,
        dtype=torch.float32
    ).unsqueeze(-1)

    y = torch.tensor(
        labels,
        dtype=torch.long
    )

    dataset = TensorDataset(
        X,
        y
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    return loader


# =============================================================================
# MODEL EVALUATION
# =============================================================================

def evaluate_model(
    model,
    loader,
):
    model.eval()

    all_predictions = []
    all_labels = []

    total_loss = 0.0
    total_samples = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(
                DEVICE
            )

            y_batch = y_batch.to(
                DEVICE
            )

            logits = model(
                X_batch
            )

            loss = criterion(
                logits,
                y_batch
            )

            predictions = torch.argmax(
                logits,
                dim=1
            )

            batch_size = y_batch.size(0)

            total_loss += (
                loss.item()
                * batch_size
            )

            total_samples += batch_size

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                y_batch.cpu().numpy()
            )

    all_predictions = np.asarray(
        all_predictions
    )

    all_labels = np.asarray(
        all_labels
    )

    average_loss = (
        total_loss / total_samples
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    balanced_accuracy = balanced_accuracy_score(
        all_labels,
        all_predictions
    )

    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "macro_f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "labels": all_labels,
        "predictions": all_predictions,
    }


# =============================================================================
# TRAINING
# =============================================================================

def train_model(
    train_loader,
    val_loader,
):
    model = FFTMagnitudeTransformer().to(
        DEVICE
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_accuracy = -np.inf
    best_val_loss = np.inf
    best_state = None

    epochs_without_improvement = 0

    history = []

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(
                DEVICE
            )

            y_batch = y_batch.to(
                DEVICE
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                X_batch
            )

            loss = criterion(
                logits,
                y_batch
            )

            loss.backward()

            optimizer.step()

            batch_size = y_batch.size(0)

            running_loss += (
                loss.item()
                * batch_size
            )

            predictions = torch.argmax(
                logits,
                dim=1
            )

            correct += (
                predictions == y_batch
            ).sum().item()

            total += batch_size

        train_loss = (
            running_loss / total
        )

        train_accuracy = (
            correct / total
        )

        val_metrics = evaluate_model(
            model,
            val_loader
        )

        val_loss = val_metrics["loss"]
        val_accuracy = val_metrics["accuracy"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_macro_f1": val_metrics["macro_f1"],
            }
        )

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy * 100:.3f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy * 100:.3f}% | "
            f"Val F1: {val_metrics['macro_f1'] * 100:.3f}%"
        )

        # ---------------------------------------------------------------------
        # Early stopping
        # ---------------------------------------------------------------------

        improved = (
            val_accuracy > best_val_accuracy
            or (
                np.isclose(
                    val_accuracy,
                    best_val_accuracy
                )
                and val_loss < best_val_loss
            )
        )

        if improved:

            best_val_accuracy = val_accuracy
            best_val_loss = val_loss

            best_state = {
                key: value.detach()
                .cpu()
                .clone()
                for key, value
                in model.state_dict().items()
            }

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print(
                f"Early stopping at epoch {epoch}."
            )

            break

    if best_state is None:
        raise RuntimeError(
            "No best model state was saved."
        )

    model.load_state_dict(
        best_state
    )

    return model, history


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def main():

    start_time = time.time()

    print()
    print("=" * 80)
    print("NEURO-SYMPHONY — EXPERIMENT 27")
    print("CHANNEL-DOMAIN FFT MAGNITUDE + TRANSFORMER")
    print("=" * 80)
    print()

    print("Project root:")
    print(PROJECT_ROOT)

    print()
    print("Input CSV:")
    print(DATA_PATH)

    print()
    print("Output directory:")
    print(RESULTS_DIR)

    print()

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Approved CSV not found:\n{DATA_PATH}"
        )

    # =========================================================================
    # LOAD CSV
    # =========================================================================

    df = pd.read_csv(
        DATA_PATH
    )

    validate_dataset(df)

    # =========================================================================
    # FFT
    # =========================================================================

    fft_complex, fft_magnitude = extract_fft_magnitude(
        df
    )

    # Save a compact feature CSV for verification.
    feature_columns = [
        f"fft_mag_bin_{i}"
        for i in range(NUM_FFT_BINS)
    ]

    feature_df = pd.DataFrame(
        fft_magnitude,
        columns=feature_columns
    )

    feature_df.insert(
        0,
        "label",
        df["label"].to_numpy()
    )

    feature_df.insert(
        0,
        "trial_id",
        df["trial_id"].to_numpy()
    )

    feature_df.insert(
        0,
        "subject_id",
        df["subject_id"].to_numpy()
    )

    feature_path = (
        RESULTS_DIR
        / "fft_magnitude_features.csv"
    )

    feature_df.to_csv(
        feature_path,
        index=False
    )

    print(
        f"FFT feature file saved:\n{feature_path}"
    )

    print()

    # =========================================================================
    # LABELS
    # =========================================================================

    labels = (
        df["label"].to_numpy(
            dtype=np.int64
        )
        - 1
    )

    # =========================================================================
    # SUBJECT-WISE 12-FOLD EVALUATION
    # =========================================================================

    subjects = sorted(
        df["subject_id"]
        .astype(str)
        .unique()
    )

    fold_results = []
    all_history = []
    all_predictions = []

    combined_confusion = np.zeros(
        (
            NUM_CLASSES,
            NUM_CLASSES
        ),
        dtype=np.int64
    )

    for fold_idx, test_subject in enumerate(
        subjects
    ):

        # Next subject becomes validation subject.
        validation_subject = subjects[
            (fold_idx + 1) % len(subjects)
        ]

        train_subjects = [
            subject
            for subject in subjects
            if subject not in {
                test_subject,
                validation_subject,
            }
        ]

        print()
        print("=" * 80)
        print(
            f"FOLD {fold_idx + 1}/{len(subjects)}"
        )
        print("=" * 80)

        print(
            "Test subject:",
            test_subject
        )

        print(
            "Validation subject:",
            validation_subject
        )

        print(
            "Train subjects:",
            train_subjects
        )

        # ---------------------------------------------------------------------
        # Indices
        # ---------------------------------------------------------------------

        subject_array = (
            df["subject_id"]
            .astype(str)
            .to_numpy()
        )

        train_mask = np.isin(
            subject_array,
            train_subjects
        )

        val_mask = (
            subject_array
            == validation_subject
        )

        test_mask = (
            subject_array
            == test_subject
        )

        train_indices = np.where(
            train_mask
        )[0]

        val_indices = np.where(
            val_mask
        )[0]

        test_indices = np.where(
            test_mask
        )[0]

        print()
        print(
            "Train trials:",
            len(train_indices)
        )

        print(
            "Validation trials:",
            len(val_indices)
        )

        print(
            "Test trials:",
            len(test_indices)
        )

        # ---------------------------------------------------------------------
        # Training-only normalization
        # ---------------------------------------------------------------------

        (
            X_train,
            X_val,
            X_test,
            train_mean,
            train_std,
        ) = prepare_features(
            fft_magnitude,
            train_indices,
            val_indices,
            test_indices,
        )

        y_train = labels[
            train_indices
        ]

        y_val = labels[
            val_indices
        ]

        y_test = labels[
            test_indices
        ]

        # ---------------------------------------------------------------------
        # Check Transformer input
        # ---------------------------------------------------------------------

        print()
        print("Transformer input shape:")
        print(
            "Train:",
            (
                X_train.shape[0],
                NUM_FFT_BINS,
                1,
            )
        )

        print(
            "Validation:",
            (
                X_val.shape[0],
                NUM_FFT_BINS,
                1,
            )
        )

        print(
            "Test:",
            (
                X_test.shape[0],
                NUM_FFT_BINS,
                1,
            )
        )

        # ---------------------------------------------------------------------
        # DataLoaders
        # ---------------------------------------------------------------------

        train_loader = make_loader(
            X_train,
            y_train,
            shuffle=True,
        )

        val_loader = make_loader(
            X_val,
            y_val,
            shuffle=False,
        )

        test_loader = make_loader(
            X_test,
            y_test,
            shuffle=False,
        )

        # ---------------------------------------------------------------------
        # Train
        # ---------------------------------------------------------------------

        print()
        print("Training Transformer...")
        print()

        fold_start = time.time()

        model, history = train_model(
            train_loader,
            val_loader,
        )

        fold_time = (
            time.time()
            - fold_start
        )

        # Add fold number
        for row in history:
            row["fold"] = fold_idx + 1

        all_history.extend(
            history
        )

        # ---------------------------------------------------------------------
        # Final metrics
        # ---------------------------------------------------------------------

        train_metrics = evaluate_model(
            model,
            train_loader
        )

        val_metrics = evaluate_model(
            model,
            val_loader
        )

        test_metrics = evaluate_model(
            model,
            test_loader
        )

        print()
        print("-" * 80)
        print("FOLD RESULTS")
        print("-" * 80)

        print(
            f"Train Accuracy: "
            f"{train_metrics['accuracy'] * 100:.3f}%"
        )

        print(
            f"Validation Accuracy: "
            f"{val_metrics['accuracy'] * 100:.3f}%"
        )

        print(
            f"Test Accuracy: "
            f"{test_metrics['accuracy'] * 100:.3f}%"
        )

        print(
            f"Test Precision: "
            f"{test_metrics['precision'] * 100:.3f}%"
        )

        print(
            f"Test Recall: "
            f"{test_metrics['recall'] * 100:.3f}%"
        )

        print(
            f"Test Macro F1: "
            f"{test_metrics['macro_f1'] * 100:.3f}%"
        )

        print(
            f"Test Balanced Accuracy: "
            f"{test_metrics['balanced_accuracy'] * 100:.3f}%"
        )

        print(
            f"Fold runtime: "
            f"{fold_time:.2f} seconds"
        )

        # ---------------------------------------------------------------------
        # Confusion matrix
        # ---------------------------------------------------------------------

        cm = confusion_matrix(
            y_test,
            test_metrics["predictions"],
            labels=np.arange(
                NUM_CLASSES
            ),
        )

        combined_confusion += cm

        # ---------------------------------------------------------------------
        # Predictions
        # ---------------------------------------------------------------------

        prediction_rows = pd.DataFrame(
            {
                "fold": fold_idx + 1,
                "subject_id": df.iloc[
                    test_indices
                ]["subject_id"].to_numpy(),
                "trial_id": df.iloc[
                    test_indices
                ]["trial_id"].to_numpy(),
                "true_label": y_test + 1,
                "predicted_label": (
                    test_metrics["predictions"]
                    + 1
                ),
            }
        )

        all_predictions.append(
            prediction_rows
        )

        # ---------------------------------------------------------------------
        # Fold result
        # ---------------------------------------------------------------------

        fold_results.append(
            {
                "fold": fold_idx + 1,
                "test_subject": test_subject,
                "validation_subject": validation_subject,
                "train_trials": len(train_indices),
                "validation_trials": len(val_indices),
                "test_trials": len(test_indices),
                "train_accuracy": train_metrics["accuracy"],
                "validation_accuracy": val_metrics["accuracy"],
                "test_accuracy": test_metrics["accuracy"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_balanced_accuracy": test_metrics[
                    "balanced_accuracy"
                ],
                "epochs_used": len(history),
                "runtime_seconds": fold_time,
            }
        )

        # ---------------------------------------------------------------------
        # Save fold model
        # ---------------------------------------------------------------------

        model_path = (
            RESULTS_DIR
            / f"fold_{fold_idx + 1:02d}_model.pt"
        )

        torch.save(
            model.state_dict(),
            model_path
        )

    # =========================================================================
    # RESULTS
    # =========================================================================

    results_df = pd.DataFrame(
        fold_results
    )

    results_path = (
        RESULTS_DIR
        / "fft_magnitude_transformer_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    history_df = pd.DataFrame(
        all_history
    )

    history_path = (
        RESULTS_DIR
        / "training_history.csv"
    )

    history_df.to_csv(
        history_path,
        index=False
    )

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True
    )

    predictions_path = (
        RESULTS_DIR
        / "fft_magnitude_transformer_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False
    )

    # =========================================================================
    # CONFUSION MATRIX
    # =========================================================================

    confusion_df = pd.DataFrame(
        combined_confusion,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    )

    confusion_path = (
        RESULTS_DIR
        / "confusion_matrix.csv"
    )

    confusion_df.to_csv(
        confusion_path
    )

    # =========================================================================
    # OVERALL CLASSIFICATION REPORT
    # =========================================================================

    true_labels = (
        predictions_df["true_label"]
        .to_numpy()
        - 1
    )

    predicted_labels = (
        predictions_df["predicted_label"]
        .to_numpy()
        - 1
    )

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=np.arange(
            NUM_CLASSES
        ),
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    report_path = (
        RESULTS_DIR
        / "classification_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(report)

    # =========================================================================
    # SUMMARY
    # =========================================================================

    accuracy_mean = (
        results_df["test_accuracy"]
        .mean()
    )

    accuracy_std = (
        results_df["test_accuracy"]
        .std(ddof=1)
    )

    precision_mean = (
        results_df["test_precision"]
        .mean()
    )

    recall_mean = (
        results_df["test_recall"]
        .mean()
    )

    f1_mean = (
        results_df["test_macro_f1"]
        .mean()
    )

    f1_std = (
        results_df["test_macro_f1"]
        .std(ddof=1)
    )

    balanced_mean = (
        results_df["test_balanced_accuracy"]
        .mean()
    )

    chance = 1.0 / NUM_CLASSES

    improvement_pp = (
        accuracy_mean - chance
    ) * 100

    total_runtime = (
        time.time()
        - start_time
    )

    # =========================================================================
    # METADATA
    # =========================================================================

    metadata = {
        "experiment": "FFT Magnitude + Transformer",
        "dataset": "DS004514",
        "input_csv": str(DATA_PATH),
        "dataset_shape": list(df.shape),
        "num_subjects": len(subjects),
        "num_classes": NUM_CLASSES,
        "num_eeg_channels": NUM_CHANNELS,
        "fft_type": "rFFT across channel dimension",
        "fft_complex_shape": list(
            fft_complex.shape
        ),
        "fft_magnitude_shape": list(
            fft_magnitude.shape
        ),
        "transformer_input": [
            "batch",
            NUM_FFT_BINS,
            1,
        ],
        "d_model": D_MODEL,
        "attention_heads": N_HEADS,
        "transformer_layers": N_LAYERS,
        "feedforward_dimension": FF_DIM,
        "dropout": DROPOUT,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "early_stopping_patience": PATIENCE,
        "seed": RANDOM_SEED,
        "device": str(DEVICE),
        "test_accuracy_mean": float(
            accuracy_mean
        ),
        "test_accuracy_std": float(
            accuracy_std
        ),
        "test_precision_mean": float(
            precision_mean
        ),
        "test_recall_mean": float(
            recall_mean
        ),
        "test_macro_f1_mean": float(
            f1_mean
        ),
        "test_macro_f1_std": float(
            f1_std
        ),
        "test_balanced_accuracy_mean": float(
            balanced_mean
        ),
        "chance_accuracy": chance,
        "improvement_over_chance_percentage_points": float(
            improvement_pp
        ),
        "total_runtime_seconds": float(
            total_runtime
        ),
    }

    metadata_path = (
        RESULTS_DIR
        / "metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2
        )

    # =========================================================================
    # FINAL OUTPUT
    # =========================================================================

    print()
    print("=" * 80)
    print("EXPERIMENT 27 COMPLETE")
    print("=" * 80)

    print()
    print(
        f"Test Accuracy: "
        f"{accuracy_mean * 100:.3f}% "
        f"± {accuracy_std * 100:.3f}%"
    )

    print(
        f"Test Precision: "
        f"{precision_mean * 100:.3f}%"
    )

    print(
        f"Test Recall: "
        f"{recall_mean * 100:.3f}%"
    )

    print(
        f"Test Macro F1: "
        f"{f1_mean * 100:.3f}% "
        f"± {f1_std * 100:.3f}%"
    )

    print(
        f"Test Balanced Accuracy: "
        f"{balanced_mean * 100:.3f}%"
    )

    print(
        f"Chance Level: "
        f"{chance * 100:.3f}%"
    )

    print(
        f"Improvement over chance: "
        f"{improvement_pp:+.3f} percentage points"
    )

    print()
    print("Files generated:")
    print(
        f"  {feature_path}"
    )
    print(
        f"  {results_path}"
    )
    print(
        f"  {history_path}"
    )
    print(
        f"  {predictions_path}"
    )
    print(
        f"  {confusion_path}"
    )
    print(
        f"  {report_path}"
    )
    print(
        f"  {metadata_path}"
    )

    print()
    print(
        f"Total runtime: "
        f"{total_runtime / 60:.2f} minutes"
    )

    print()
    print("=" * 80)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()