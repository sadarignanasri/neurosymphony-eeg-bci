"""
================================================================================
EXPERIMENT 5: PLAIN 1D CNN
================================================================================

Neuro-Symphony
EEG Semantic Decoding - Controlled Experiment Pipeline

IMPORTANT REPRESENTATION NOTE
------------------------------
The current approved EEG CSV contains:

    1 row = 1 complete trial
    64 columns = 64 EEG channel values

It does NOT contain temporal EEG samples from the original BDF files.

Therefore this experiment treats the 64 EEG channels as a 1D
CHANNEL-DOMAIN representation:

    [batch, 1, 64]

This is NOT a temporal EEG CNN.

The purpose of this experiment is to determine whether a plain CNN can learn
useful channel-domain representations from the current approved CSV.

CONTROLLED PROTOCOL
-------------------
12 subjects total.

For every fold:

    10 subjects -> training
    1 subject    -> validation
    1 subject    -> testing

The test subject rotates through all 12 subjects.

SCALING
-------
RobustScaler is fitted ONLY on the 10 training subjects.

LABEL HANDLING
--------------
The dataset contains labels:

    1, 2, 3, 4, 5, 6, 7, 8

PyTorch CrossEntropyLoss requires zero-based class indices:

    0, 1, 2, 3, 4, 5, 6, 7

Therefore:

    dataset label 1 -> PyTorch target 0
    dataset label 2 -> PyTorch target 1
    ...
    dataset label 8 -> PyTorch target 7

Predictions are converted back to dataset labels 1..8 before saving.

ARCHITECTURE
------------
Input:
    [B, 1, 64]

Conv Block 1:
    Conv1d(1 -> 32, kernel_size=3)
    ReLU
    MaxPool1d(2)

Conv Block 2:
    Conv1d(32 -> 64, kernel_size=3)
    ReLU
    MaxPool1d(2)

Conv Block 3:
    Conv1d(64 -> 128, kernel_size=3)
    ReLU

Global Average Pooling:
    AdaptiveAvgPool1d(1)

Classifier:
    Dropout(0.3)
    Linear(128 -> 8)

TRAINING
--------
Optimizer:
    AdamW

Loss:
    CrossEntropyLoss

Batch size:
    64

Maximum epochs:
    50

Learning rate:
    1e-3

Weight decay:
    1e-4

Early stopping patience:
    8

OUTPUTS
-------
results/eeg/cnn/

    cnn_results.csv
    cnn_predictions.csv
    cnn_summary.csv
    cnn_classification_reports.txt
    cnn_confusion_matrix.csv
    cnn_training_history.csv
    cnn_split_summary.csv
    cnn_experiment_metadata.csv

================================================================================
"""

from __future__ import annotations

# ==============================================================================
# CPU RUNTIME STABILITY
# ==============================================================================

import os

# These must be set before importing torch.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


# ==============================================================================
# IMPORTS
# ==============================================================================

from pathlib import Path
import copy
import random
import time

import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

import torch
import torch.nn as nn
import torch.optim as optim


# ==============================================================================
# CONFIGURATION
# ==============================================================================

SEED = 42

DATA_PATH = Path(
    "data/processed/ds004514_eeg_approved.csv"
)

OUTPUT_DIR = Path(
    "results/eeg/cnn"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------------------------
# Dataset columns
# ------------------------------------------------------------------------------

SUBJECT_COLUMN = "subject_id"
TRIAL_COLUMN = "trial_id"
LABEL_COLUMN = "label"
LABEL_NAME_COLUMN = "label_name"


# ------------------------------------------------------------------------------
# EEG channels
# ------------------------------------------------------------------------------

EEG_CHANNELS = (
    [f"A{i}" for i in range(1, 33)]
    +
    [f"B{i}" for i in range(1, 33)]
)

NUM_CHANNELS = 64
NUM_CLASSES = 8


# ------------------------------------------------------------------------------
# Training configuration
# ------------------------------------------------------------------------------

BATCH_SIZE = 64
MAX_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 8


# ------------------------------------------------------------------------------
# Dataset labels
# ------------------------------------------------------------------------------

EXPECTED_LABELS = list(range(1, 9))


# ------------------------------------------------------------------------------
# Chance level
# ------------------------------------------------------------------------------

CHANCE_ACCURACY = (
    100.0 / NUM_CLASSES
)


# ==============================================================================
# REPRODUCIBILITY
# ==============================================================================

def set_seed(seed: int = SEED) -> None:

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


set_seed()


# ==============================================================================
# PYTORCH CPU CONFIGURATION
# ==============================================================================

torch.set_num_threads(1)

try:

    torch.set_num_interop_threads(1)

except RuntimeError:

    pass


try:

    torch.backends.mkldnn.enabled = False

except Exception:

    pass


# ==============================================================================
# DEVICE
# ==============================================================================

# CPU is intentionally used for runtime stability.

DEVICE = torch.device(
    "cpu"
)


# ==============================================================================
# PRINTING UTILITIES
# ==============================================================================

def print_separator(
    character: str = "=",
    width: int = 80,
) -> None:

    print(
        character * width
    )


def print_header(
    title: str
) -> None:

    print_separator()

    print(title)

    print_separator()


# ==============================================================================
# LABEL CONVERSION
# ==============================================================================

def label_to_torch(
    labels: np.ndarray
) -> np.ndarray:
    """
    Convert original dataset labels 1..8 to PyTorch labels 0..7.
    """

    labels = np.asarray(
        labels,
        dtype=np.int64
    )

    invalid_mask = (
        ~np.isin(
            labels,
            EXPECTED_LABELS
        )
    )

    if np.any(invalid_mask):

        invalid_values = np.unique(
            labels[invalid_mask]
        )

        raise ValueError(
            "Unexpected labels found: "
            f"{invalid_values}. "
            f"Expected: {EXPECTED_LABELS}"
        )

    return labels - 1


def torch_to_dataset_labels(
    labels: np.ndarray
) -> np.ndarray:
    """
    Convert PyTorch labels 0..7 back to dataset labels 1..8.
    """

    labels = np.asarray(
        labels,
        dtype=np.int64
    )

    if np.any(labels < 0):

        raise ValueError(
            "Negative PyTorch class index detected."
        )

    if np.any(
        labels >= NUM_CLASSES
    ):

        raise ValueError(
            "PyTorch class index >= number of classes detected."
        )

    return labels + 1


# ==============================================================================
# LOAD DATASET
# ==============================================================================

def load_dataset() -> pd.DataFrame:

    print_header(
        "LOADING DATASET"
    )

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            "Dataset not found:\n"
            f"{DATA_PATH.resolve()}"
        )

    df = pd.read_csv(
        DATA_PATH
    )

    print(
        f"Dataset shape: {df.shape}"
    )

    # --------------------------------------------------------------------------
    # Required columns
    # --------------------------------------------------------------------------

    required_columns = (
        [
            SUBJECT_COLUMN,
            TRIAL_COLUMN,
            LABEL_COLUMN,
            LABEL_NAME_COLUMN,
        ]
        +
        EEG_CHANNELS
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns:\n"
            +
            "\n".join(
                missing_columns
            )
        )

    subjects = sorted(
        df[
            SUBJECT_COLUMN
        ]
        .astype(str)
        .unique()
    )

    labels = sorted(
        df[
            LABEL_COLUMN
        ]
        .unique()
    )

    print(
        f"Number of subjects: {len(subjects)}"
    )

    print(
        f"Number of EEG channels: "
        f"{len(EEG_CHANNELS)}"
    )

    print(
        f"Number of classes: {len(labels)}"
    )

    print("\nSubjects:")

    print(
        subjects
    )

    print("\nLabels:")

    print(
        labels
    )

    return df


# ==============================================================================
# DATASET VALIDATION
# ==============================================================================

def validate_dataset(
    df: pd.DataFrame
) -> None:

    print_header(
        "DATASET VALIDATION"
    )

    subjects = sorted(
        df[
            SUBJECT_COLUMN
        ]
        .astype(str)
        .unique()
    )

    labels = sorted(
        df[
            LABEL_COLUMN
        ]
        .unique()
    )

    # --------------------------------------------------------------------------
    # Number of subjects
    # --------------------------------------------------------------------------

    if len(subjects) != 12:

        raise ValueError(
            f"Expected 12 subjects, "
            f"found {len(subjects)}."
        )

    # --------------------------------------------------------------------------
    # Number of classes
    # --------------------------------------------------------------------------

    if len(labels) != NUM_CLASSES:

        raise ValueError(
            f"Expected {NUM_CLASSES} classes, "
            f"found {len(labels)}."
        )

    # --------------------------------------------------------------------------
    # Label values
    # --------------------------------------------------------------------------

    if labels != EXPECTED_LABELS:

        raise ValueError(
            f"Expected labels "
            f"{EXPECTED_LABELS}, "
            f"found {labels}."
        )

    # --------------------------------------------------------------------------
    # Number of EEG channels
    # --------------------------------------------------------------------------

    if len(EEG_CHANNELS) != NUM_CHANNELS:

        raise ValueError(
            f"Expected {NUM_CHANNELS} EEG channels, "
            f"found {len(EEG_CHANNELS)}."
        )

    # ==========================================================================
    # Trials per subject
    # ==========================================================================

    subject_counts = (
        df
        .groupby(SUBJECT_COLUMN)
        .size()
        .sort_index()
    )

    print(
        "\nTrials per subject:"
    )

    print(
        subject_counts
    )

    if not np.all(
        subject_counts.values == 720
    ):

        raise ValueError(
            "Every subject must contain "
            "exactly 720 trials."
        )

    # ==========================================================================
    # Trials per class
    # ==========================================================================

    class_counts = (
        df
        .groupby(LABEL_COLUMN)
        .size()
        .sort_index()
    )

    print(
        "\nTrials per class:"
    )

    print(
        class_counts
    )

    if not np.all(
        class_counts.values == 1080
    ):

        raise ValueError(
            "Every class must contain "
            "exactly 1080 trials."
        )

    # ==========================================================================
    # Subject × class
    # ==========================================================================

    subject_class_counts = pd.crosstab(
        df[SUBJECT_COLUMN],
        df[LABEL_COLUMN]
    )

    print(
        "\nSubject × class distribution:"
    )

    print(
        subject_class_counts
    )

    expected_distribution = np.full(
        (12, 8),
        90,
        dtype=int
    )

    if not np.array_equal(
        subject_class_counts.values,
        expected_distribution
    ):

        raise ValueError(
            "Expected exactly 90 trials "
            "per subject/class combination."
        )

    # ==========================================================================
    # Duplicate subject/trial
    # ==========================================================================

    duplicate_count = (
        df
        .duplicated(
            subset=[
                SUBJECT_COLUMN,
                TRIAL_COLUMN,
            ]
        )
        .sum()
    )

    print(
        "\nDuplicate subject/trial combinations: "
        f"{duplicate_count}"
    )

    if duplicate_count != 0:

        raise ValueError(
            "Duplicate subject/trial combinations detected."
        )

    # ==========================================================================
    # Missing EEG
    # ==========================================================================

    eeg_values = df[
        EEG_CHANNELS
    ]

    missing_count = (
        eeg_values
        .isna()
        .sum()
        .sum()
    )

    print(
        f"Missing EEG values: "
        f"{missing_count}"
    )

    if missing_count != 0:

        raise ValueError(
            "Missing EEG values detected."
        )

    # ==========================================================================
    # Infinite EEG
    # ==========================================================================

    infinite_count = np.isinf(
        eeg_values.to_numpy(
            dtype=np.float64
        )
    ).sum()

    print(
        f"Infinite EEG values: "
        f"{infinite_count}"
    )

    if infinite_count != 0:

        raise ValueError(
            "Infinite EEG values detected."
        )

    # ==========================================================================
    # Numeric EEG channels
    # ==========================================================================

    for channel in EEG_CHANNELS:

        if not pd.api.types.is_numeric_dtype(
            df[channel]
        ):

            raise TypeError(
                f"EEG channel {channel} "
                "is not numeric."
            )

    print(
        "\nDataset validation passed."
    )


# ==============================================================================
# PLAIN 1D CNN
# ==============================================================================

class PlainCNN(
    nn.Module
):
    """
    Plain 1D CNN operating over the 64 EEG channel positions.

    Input:
        [batch, 1, 64]

    Output:
        [batch, 8]
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
    ):

        super().__init__()

        self.features = nn.Sequential(

            # ------------------------------------------------------------------
            # CNN Block 1
            # ------------------------------------------------------------------

            nn.Conv1d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(),

            nn.MaxPool1d(
                kernel_size=2
            ),

            # ------------------------------------------------------------------
            # CNN Block 2
            # ------------------------------------------------------------------

            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(),

            nn.MaxPool1d(
                kernel_size=2
            ),

            # ------------------------------------------------------------------
            # CNN Block 3
            # ------------------------------------------------------------------

            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(),

            # ------------------------------------------------------------------
            # Global average pooling
            # ------------------------------------------------------------------

            nn.AdaptiveAvgPool1d(
                1
            ),
        )

        self.classifier = nn.Sequential(

            nn.Dropout(
                p=0.30
            ),

            nn.Linear(
                128,
                num_classes
            ),
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        x = self.features(
            x
        )

        # Shape:
        # [batch, 128, 1]

        x = x.squeeze(
            -1
        )

        # Shape:
        # [batch, 128]

        x = self.classifier(
            x
        )

        # Shape:
        # [batch, 8]

        return x


# ==============================================================================
# CNN SANITY CHECK
# ==============================================================================

def model_sanity_check() -> None:

    print_header(
        "CNN MODEL SANITY CHECK"
    )

    model = PlainCNN(
        num_classes=NUM_CLASSES
    ).to(
        DEVICE
    )

    test_input = torch.zeros(
        2,
        1,
        NUM_CHANNELS,
        dtype=torch.float32,
        device=DEVICE,
    )

    print(
        f"Input shape: "
        f"{tuple(test_input.shape)}"
    )

    with torch.no_grad():

        output = model(
            test_input
        )

    print(
        f"Output shape: "
        f"{tuple(output.shape)}"
    )

    expected_shape = (
        2,
        NUM_CLASSES
    )

    if tuple(
        output.shape
    ) != expected_shape:

        raise RuntimeError(
            "CNN sanity check failed. "
            f"Expected {expected_shape}, "
            f"got {tuple(output.shape)}."
        )

    print(
        "CNN model sanity check passed."
    )


# ==============================================================================
# TRAIN ONE EPOCH
# ==============================================================================

def train_one_epoch(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
) -> tuple[float, float]:

    model.train()

    # --------------------------------------------------------------------------
    # Shuffle
    # --------------------------------------------------------------------------

    indices = np.random.permutation(
        len(X_train)
    )

    total_loss = 0.0

    correct = 0

    total = 0

    # --------------------------------------------------------------------------
    # Manual mini-batches
    # --------------------------------------------------------------------------

    for start in range(
        0,
        len(indices),
        BATCH_SIZE,
    ):

        batch_indices = indices[
            start:
            start + BATCH_SIZE
        ]

        X_batch = torch.from_numpy(
            X_train[
                batch_indices
            ]
        ).float().to(
            DEVICE
        )

        y_batch = torch.from_numpy(
            y_train[
                batch_indices
            ]
        ).long().to(
            DEVICE
        )

        # ----------------------------------------------------------------------
        # Forward
        # ----------------------------------------------------------------------

        optimizer.zero_grad()

        outputs = model(
            X_batch
        )

        # ----------------------------------------------------------------------
        # Loss
        # ----------------------------------------------------------------------

        loss = criterion(
            outputs,
            y_batch
        )

        # ----------------------------------------------------------------------
        # Backward
        # ----------------------------------------------------------------------

        loss.backward()

        optimizer.step()

        # ----------------------------------------------------------------------
        # Metrics
        # ----------------------------------------------------------------------

        batch_size = len(
            batch_indices
        )

        total_loss += (
            loss.item()
            *
            batch_size
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == y_batch
        ).sum().item()

        total += batch_size

    epoch_loss = (
        total_loss / total
    )

    epoch_accuracy = (
        correct / total
    ) * 100.0

    return (
        epoch_loss,
        epoch_accuracy,
    )


# ==============================================================================
# EVALUATE MODEL
# ==============================================================================

def evaluate_model(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    criterion: nn.Module,
) -> tuple[
    float,
    float,
    np.ndarray,
]:

    model.eval()

    total_loss = 0.0

    total = 0

    all_predictions = []

    with torch.no_grad():

        for start in range(
            0,
            len(X),
            BATCH_SIZE,
        ):

            end = min(
                start + BATCH_SIZE,
                len(X),
            )

            X_batch = torch.from_numpy(
                X[start:end]
            ).float().to(
                DEVICE
            )

            y_batch = torch.from_numpy(
                y[start:end]
            ).long().to(
                DEVICE
            )

            outputs = model(
                X_batch
            )

            loss = criterion(
                outputs,
                y_batch
            )

            batch_size = (
                end - start
            )

            total_loss += (
                loss.item()
                *
                batch_size
            )

            total += batch_size

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    average_loss = (
        total_loss / total
    )

    all_predictions = np.asarray(
        all_predictions,
        dtype=np.int64
    )

    accuracy = (
        np.mean(
            all_predictions == y
        )
        * 100.0
    )

    return (
        average_loss,
        accuracy,
        all_predictions,
    )


# ==============================================================================
# TRAIN MODEL
# ==============================================================================

def train_model(
    X_train: np.ndarray,
    y_train_original: np.ndarray,
    X_val: np.ndarray,
    y_val_original: np.ndarray,
) -> tuple[
    nn.Module,
    list[dict],
]:

    # --------------------------------------------------------------------------
    # IMPORTANT LABEL CONVERSION
    #
    # Original:
    #     1..8
    #
    # PyTorch:
    #     0..7
    # --------------------------------------------------------------------------

    y_train = label_to_torch(
        y_train_original
    )

    y_val = label_to_torch(
        y_val_original
    )

    # --------------------------------------------------------------------------
    # Float32
    # --------------------------------------------------------------------------

    X_train = np.asarray(
        X_train,
        dtype=np.float32
    )

    X_val = np.asarray(
        X_val,
        dtype=np.float32
    )

    # --------------------------------------------------------------------------
    # Reshape:
    #
    # [samples, 64]
    #
    # ->
    #
    # [samples, 1, 64]
    # --------------------------------------------------------------------------

    X_train = X_train.reshape(
        -1,
        1,
        NUM_CHANNELS
    )

    X_val = X_val.reshape(
        -1,
        1,
        NUM_CHANNELS
    )

    # --------------------------------------------------------------------------
    # Model
    # --------------------------------------------------------------------------

    model = PlainCNN(
        num_classes=NUM_CLASSES
    ).to(
        DEVICE
    )

    # --------------------------------------------------------------------------
    # Loss
    # --------------------------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------------------------

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------------------------
    # History
    # --------------------------------------------------------------------------

    history = []

    # --------------------------------------------------------------------------
    # Early stopping state
    # --------------------------------------------------------------------------

    best_val_accuracy = -np.inf

    best_epoch = 0

    best_state = copy.deepcopy(
        model.state_dict()
    )

    epochs_without_improvement = 0

    # ==========================================================================
    # EPOCH LOOP
    # ==========================================================================

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        train_loss, train_accuracy = (
            train_one_epoch(
                model=model,
                X_train=X_train,
                y_train=y_train,
                criterion=criterion,
                optimizer=optimizer,
            )
        )

        val_loss, val_accuracy, _ = (
            evaluate_model(
                model=model,
                X=X_val,
                y=y_val,
                criterion=criterion,
            )
        )

        # ----------------------------------------------------------------------
        # Store history
        # ----------------------------------------------------------------------

        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": val_loss,
            "validation_accuracy": val_accuracy,
        }

        history.append(
            history_row
        )

        # ----------------------------------------------------------------------
        # Print
        # ----------------------------------------------------------------------

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.3f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.3f}%"
        )

        # ----------------------------------------------------------------------
        # Improvement
        # ----------------------------------------------------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = (
                val_accuracy
            )

            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        # ----------------------------------------------------------------------
        # Early stopping
        # ----------------------------------------------------------------------

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print(
                f"Early stopping at epoch "
                f"{epoch}. "
                f"Best validation accuracy: "
                f"{best_val_accuracy:.3f}% "
                f"(epoch {best_epoch})"
            )

            break

    # ==========================================================================
    # Restore best model
    # ==========================================================================

    model.load_state_dict(
        best_state
    )

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.3f}% "
        f"(epoch {best_epoch})"
    )

    return (
        model,
        history,
    )


# ==============================================================================
# MAIN
# ==============================================================================

def main():

    experiment_start = time.time()

    # ==========================================================================
    # HEADER
    # ==========================================================================

    print_header(
        "EXPERIMENT 5: PLAIN 1D CNN"
    )

    print(
        f"\nDevice: {DEVICE}"
    )

    print(
        "Runtime mode: stable CPU configuration"
    )

    print(
        f"PyTorch threads: "
        f"{torch.get_num_threads()}"
    )

    print(
        f"PyTorch version: "
        f"{torch.__version__}"
    )

    try:

        print(
            f"MKLDNN enabled: "
            f"{torch.backends.mkldnn.enabled}"
        )

    except Exception:

        print(
            "MKLDNN status: unavailable"
        )

    # ==========================================================================
    # LOAD DATA
    # ==========================================================================

    df = load_dataset()

    # ==========================================================================
    # VALIDATE DATA
    # ==========================================================================

    validate_dataset(
        df
    )

    # ==========================================================================
    # FEATURE MATRIX
    # ==========================================================================

    X = df[
        EEG_CHANNELS
    ].to_numpy(
        dtype=np.float64
    )

    y = df[
        LABEL_COLUMN
    ].to_numpy(
        dtype=np.int64
    )

    subjects = (
        df[
            SUBJECT_COLUMN
        ]
        .astype(str)
        .to_numpy()
    )

    print(
        "\nFeature matrix:"
    )

    print(
        f"X shape: {X.shape}"
    )

    print(
        f"y shape: {y.shape}"
    )

    # ==========================================================================
    # MODEL SANITY CHECK
    # ==========================================================================

    model_sanity_check()

    # ==========================================================================
    # CONTROLLED LOSO
    # ==========================================================================

    print_header(
        "CONTROLLED LOSO EXPERIMENT"
    )

    print(
        "\nProtocol: "
        "10 training subjects + "
        "1 validation subject + "
        "1 test subject"
    )

    print(
        "Test subject rotates across all 12 subjects."
    )

    unique_subjects = sorted(
        np.unique(
            subjects
        )
    )

    if len(
        unique_subjects
    ) != 12:

        raise ValueError(
            "Expected exactly 12 subjects."
        )

    # ==========================================================================
    # RESULT STORAGE
    # ==========================================================================

    fold_results = []

    all_predictions = []

    all_histories = []

    split_summaries = []

    classification_reports = []

    global_confusion_matrix = np.zeros(
        (
            NUM_CLASSES,
            NUM_CLASSES
        ),
        dtype=np.int64
    )

    # ==========================================================================
    # 12 FOLDS
    # ==========================================================================

    for fold_idx, test_subject in enumerate(
        unique_subjects,
        start=1,
    ):

        fold_start = time.time()

        print("\n")

        print_separator(
            "-"
        )

        print(
            f"FOLD {fold_idx}/12"
        )

        print_separator(
            "-"
        )

        # ----------------------------------------------------------------------
        # Remaining subjects
        # ----------------------------------------------------------------------

        remaining_subjects = [
            subject
            for subject in unique_subjects
            if subject != test_subject
        ]

        # ----------------------------------------------------------------------
        # Deterministic validation subject
        # ----------------------------------------------------------------------

        validation_index = (
            (fold_idx - 1)
            %
            len(remaining_subjects)
        )

        validation_subject = (
            remaining_subjects[
                validation_index
            ]
        )

        # ----------------------------------------------------------------------
        # Training subjects
        # ----------------------------------------------------------------------

        train_subjects = [
            subject
            for subject in remaining_subjects
            if subject != validation_subject
        ]

        # ----------------------------------------------------------------------
        # Masks
        # ----------------------------------------------------------------------

        train_mask = np.isin(
            subjects,
            train_subjects
        )

        validation_mask = (
            subjects
            ==
            validation_subject
        )

        test_mask = (
            subjects
            ==
            test_subject
        )

        # ----------------------------------------------------------------------
        # Split
        # ----------------------------------------------------------------------

        X_train_raw = X[
            train_mask
        ]

        y_train = y[
            train_mask
        ]

        X_val_raw = X[
            validation_mask
        ]

        y_val = y[
            validation_mask
        ]

        X_test_raw = X[
            test_mask
        ]

        y_test = y[
            test_mask
        ]

        # ----------------------------------------------------------------------
        # Print split
        # ----------------------------------------------------------------------

        print(
            f"Train subjects: "
            f"{train_subjects}"
        )

        print(
            f"Validation subject: "
            f"{validation_subject}"
        )

        print(
            f"Test subject: "
            f"{test_subject}"
        )

        print(
            f"Train samples: "
            f"{len(X_train_raw)}"
        )

        print(
            f"Validation samples: "
            f"{len(X_val_raw)}"
        )

        print(
            f"Test samples: "
            f"{len(X_test_raw)}"
        )

        # ----------------------------------------------------------------------
        # Class counts
        # ----------------------------------------------------------------------

        train_class_counts = np.bincount(
            y_train,
            minlength=NUM_CLASSES + 1
        )[1:]

        val_class_counts = np.bincount(
            y_val,
            minlength=NUM_CLASSES + 1
        )[1:]

        test_class_counts = np.bincount(
            y_test,
            minlength=NUM_CLASSES + 1
        )[1:]

        print(
            "Train class counts: "
            f"{train_class_counts.tolist()}"
        )

        print(
            "Val class counts: "
            f"{val_class_counts.tolist()}"
        )

        print(
            "Test class counts: "
            f"{test_class_counts.tolist()}"
        )

        # ----------------------------------------------------------------------
        # Verify split sizes
        # ----------------------------------------------------------------------

        if len(
            train_subjects
        ) != 10:

            raise RuntimeError(
                "Training split must contain "
                "exactly 10 subjects."
            )

        if len(
            X_train_raw
        ) != 7200:

            raise RuntimeError(
                "Training split must contain "
                "exactly 7200 trials."
            )

        if len(
            X_val_raw
        ) != 720:

            raise RuntimeError(
                "Validation split must contain "
                "exactly 720 trials."
            )

        if len(
            X_test_raw
        ) != 720:

            raise RuntimeError(
                "Test split must contain "
                "exactly 720 trials."
            )

        # ==========================================================================
        # ROBUST SCALER
        # ==========================================================================

        scaler = RobustScaler()

        # IMPORTANT:
        # Fit ONLY on training subjects.

        scaler.fit(
            X_train_raw
        )

        X_train = scaler.transform(
            X_train_raw
        ).astype(
            np.float32
        )

        X_val = scaler.transform(
            X_val_raw
        ).astype(
            np.float32
        )

        X_test = scaler.transform(
            X_test_raw
        ).astype(
            np.float32
        )

        # ==========================================================================
        # TRAIN MODEL
        # ==========================================================================

        model, history = train_model(
            X_train=X_train,
            y_train_original=y_train,
            X_val=X_val,
            y_val_original=y_val,
        )

        # ==========================================================================
        # STORE TRAINING HISTORY
        # ==========================================================================

        for history_row in history:

            all_histories.append(
                {
                    "fold": fold_idx,
                    "test_subject": test_subject,
                    "validation_subject": validation_subject,
                    "epoch": history_row[
                        "epoch"
                    ],
                    "train_loss": history_row[
                        "train_loss"
                    ],
                    "train_accuracy": history_row[
                        "train_accuracy"
                    ],
                    "validation_loss": history_row[
                        "validation_loss"
                    ],
                    "validation_accuracy": history_row[
                        "validation_accuracy"
                    ],
                }
            )

        # ==========================================================================
        # CONVERT LABELS FOR FINAL EVALUATION
        # ==========================================================================

        y_train_torch = label_to_torch(
            y_train
        )

        y_val_torch = label_to_torch(
            y_val
        )

        y_test_torch = label_to_torch(
            y_test
        )

        # --------------------------------------------------------------------------
        # Reshape feature arrays
        # --------------------------------------------------------------------------

        X_train_model = X_train.reshape(
            -1,
            1,
            NUM_CHANNELS
        )

        X_val_model = X_val.reshape(
            -1,
            1,
            NUM_CHANNELS
        )

        X_test_model = X_test.reshape(
            -1,
            1,
            NUM_CHANNELS
        )

        # --------------------------------------------------------------------------
        # Loss
        # --------------------------------------------------------------------------

        criterion = nn.CrossEntropyLoss()

        # ==========================================================================
        # TRAIN METRICS
        # ==========================================================================

        (
            train_loss,
            train_accuracy,
            train_predictions_torch,
        ) = evaluate_model(
            model=model,
            X=X_train_model,
            y=y_train_torch,
            criterion=criterion,
        )

        # ==========================================================================
        # VALIDATION METRICS
        # ==========================================================================

        (
            validation_loss,
            validation_accuracy,
            validation_predictions_torch,
        ) = evaluate_model(
            model=model,
            X=X_val_model,
            y=y_val_torch,
            criterion=criterion,
        )

        # ==========================================================================
        # TEST METRICS
        # ==========================================================================

        (
            test_loss,
            test_accuracy,
            test_predictions_torch,
        ) = evaluate_model(
            model=model,
            X=X_test_model,
            y=y_test_torch,
            criterion=criterion,
        )

        # ==========================================================================
        # CONVERT PREDICTIONS BACK TO ORIGINAL LABELS
        # ==========================================================================

        train_predictions = (
            torch_to_dataset_labels(
                train_predictions_torch
            )
        )

        validation_predictions = (
            torch_to_dataset_labels(
                validation_predictions_torch
            )
        )

        test_predictions = (
            torch_to_dataset_labels(
                test_predictions_torch
            )
        )

        # ==========================================================================
        # TEST MACRO F1
        # ==========================================================================

        test_macro_f1 = (
            f1_score(
                y_test,
                test_predictions,
                labels=EXPECTED_LABELS,
                average="macro",
                zero_division=0,
            )
            * 100.0
        )

        # ==========================================================================
        # TEST BALANCED ACCURACY
        # ==========================================================================

        test_balanced_accuracy = (
            balanced_accuracy_score(
                y_test,
                test_predictions
            )
            * 100.0
        )

        # ==========================================================================
        # CONFUSION MATRIX
        # ==========================================================================

        fold_confusion_matrix = (
            confusion_matrix(
                y_test,
                test_predictions,
                labels=EXPECTED_LABELS,
            )
        )

        global_confusion_matrix += (
            fold_confusion_matrix
        )

        # ==========================================================================
        # CLASSIFICATION REPORT
        # ==========================================================================

        report = classification_report(
            y_test,
            test_predictions,
            labels=EXPECTED_LABELS,
            target_names=[
                str(label)
                for label in EXPECTED_LABELS
            ],
            zero_division=0,
        )

        classification_reports.append(
            {
                "fold": fold_idx,
                "validation_subject": validation_subject,
                "test_subject": test_subject,
                "report": report,
            }
        )

        # ==========================================================================
        # PREDICTIONS
        # ==========================================================================

        for trial_index in range(
            len(y_test)
        ):

            all_predictions.append(
                {
                    "fold": fold_idx,
                    "test_subject": test_subject,
                    "validation_subject": validation_subject,
                    "trial_index": trial_index,
                    "true_label": int(
                        y_test[
                            trial_index
                        ]
                    ),
                    "predicted_label": int(
                        test_predictions[
                            trial_index
                        ]
                    ),
                }
            )

        # ==========================================================================
        # BEST EPOCH
        # ==========================================================================

        if history:

            best_history_entry = max(
                history,
                key=lambda item:
                item[
                    "validation_accuracy"
                ],
            )

            best_validation_accuracy = (
                best_history_entry[
                    "validation_accuracy"
                ]
            )

            best_epoch = (
                best_history_entry[
                    "epoch"
                ]
            )

        else:

            best_validation_accuracy = np.nan

            best_epoch = np.nan

        # ==========================================================================
        # FOLD RUNTIME
        # ==========================================================================

        fold_runtime = (
            time.time()
            -
            fold_start
        )

        # ==========================================================================
        # STORE FOLD RESULT
        # ==========================================================================

        fold_results.append(
            {
                "fold": fold_idx,

                "train_subject_count": len(
                    train_subjects
                ),

                "validation_subject": (
                    validation_subject
                ),

                "test_subject": (
                    test_subject
                ),

                "train_samples": len(
                    X_train
                ),

                "validation_samples": len(
                    X_val
                ),

                "test_samples": len(
                    X_test
                ),

                "train_accuracy": (
                    train_accuracy
                ),

                "validation_accuracy": (
                    validation_accuracy
                ),

                "test_accuracy": (
                    test_accuracy
                ),

                "test_macro_f1": (
                    test_macro_f1
                ),

                "test_balanced_accuracy": (
                    test_balanced_accuracy
                ),

                "train_loss": (
                    train_loss
                ),

                "validation_loss": (
                    validation_loss
                ),

                "test_loss": (
                    test_loss
                ),

                "best_validation_accuracy": (
                    best_validation_accuracy
                ),

                "best_epoch": (
                    best_epoch
                ),

                "runtime_minutes": (
                    fold_runtime / 60.0
                ),
            }
        )

        # ==========================================================================
        # SPLIT SUMMARY
        # ==========================================================================

        split_summaries.append(
            {
                "fold": fold_idx,

                "train_subjects": ",".join(
                    train_subjects
                ),

                "validation_subject": (
                    validation_subject
                ),

                "test_subject": (
                    test_subject
                ),

                "train_samples": len(
                    X_train
                ),

                "validation_samples": len(
                    X_val
                ),

                "test_samples": len(
                    X_test
                ),

                "train_class_counts": ",".join(
                    map(
                        str,
                        train_class_counts
                    )
                ),

                "validation_class_counts": ",".join(
                    map(
                        str,
                        val_class_counts
                    )
                ),

                "test_class_counts": ",".join(
                    map(
                        str,
                        test_class_counts
                    )
                ),
            }
        )

        # ==========================================================================
        # PRINT FOLD RESULT
        # ==========================================================================

        print(
            "\nFold result:"
        )

        print(
            f"Train accuracy: "
            f"{train_accuracy:.3f}%"
        )

        print(
            f"Validation accuracy: "
            f"{validation_accuracy:.3f}%"
        )

        print(
            f"Test accuracy: "
            f"{test_accuracy:.3f}%"
        )

        print(
            f"Test Macro F1: "
            f"{test_macro_f1:.3f}%"
        )

        print(
            f"Test balanced accuracy: "
            f"{test_balanced_accuracy:.3f}%"
        )

        print(
            f"Best epoch: "
            f"{best_epoch}"
        )

        print(
            f"Best validation accuracy: "
            f"{best_validation_accuracy:.3f}%"
        )

        print(
            f"Fold runtime: "
            f"{fold_runtime / 60.0:.2f} minutes"
        )

    # ==============================================================================
    # DATAFRAMES
    # ==============================================================================

    results_df = pd.DataFrame(
        fold_results
    )

    predictions_df = pd.DataFrame(
        all_predictions
    )

    history_df = pd.DataFrame(
        all_histories
    )

    split_df = pd.DataFrame(
        split_summaries
    )

    # ==============================================================================
    # OVERALL METRICS
    # ==============================================================================

    mean_train_accuracy = (
        results_df[
            "train_accuracy"
        ].mean()
    )

    mean_validation_accuracy = (
        results_df[
            "validation_accuracy"
        ].mean()
    )

    mean_test_accuracy = (
        results_df[
            "test_accuracy"
        ].mean()
    )

    std_test_accuracy = (
        results_df[
            "test_accuracy"
        ].std(
            ddof=1
        )
    )

    mean_macro_f1 = (
        results_df[
            "test_macro_f1"
        ].mean()
    )

    std_macro_f1 = (
        results_df[
            "test_macro_f1"
        ].std(
            ddof=1
        )
    )

    mean_balanced_accuracy = (
        results_df[
            "test_balanced_accuracy"
        ].mean()
    )

    improvement_over_chance = (
        mean_test_accuracy
        -
        CHANCE_ACCURACY
    )

    total_runtime = (
        time.time()
        -
        experiment_start
    )

    # ==============================================================================
    # FINAL RESULTS
    # ==============================================================================

    print("\n")

    print_header(
        "FINAL EXPERIMENT 5 RESULTS"
    )

    print(
        f"Mean train accuracy: "
        f"{mean_train_accuracy:.3f}%"
    )

    print(
        f"Mean validation accuracy: "
        f"{mean_validation_accuracy:.3f}%"
    )

    print(
        f"Mean test accuracy: "
        f"{mean_test_accuracy:.3f}% "
        f"± {std_test_accuracy:.3f}%"
    )

    print(
        f"Mean Macro F1: "
        f"{mean_macro_f1:.3f}% "
        f"± {std_macro_f1:.3f}%"
    )

    print(
        f"Mean balanced accuracy: "
        f"{mean_balanced_accuracy:.3f}%"
    )

    print(
        f"Chance accuracy: "
        f"{CHANCE_ACCURACY:.3f}%"
    )

    print(
        f"Improvement over chance: "
        f"{improvement_over_chance:+.3f} "
        f"percentage points"
    )

    print(
        f"Total runtime: "
        f"{total_runtime / 60.0:.2f} minutes"
    )

    # ==============================================================================
    # SAVE FOLD RESULTS
    # ==============================================================================

    results_path = (
        OUTPUT_DIR
        /
        "cnn_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    # ==============================================================================
    # SAVE PREDICTIONS
    # ==============================================================================

    predictions_path = (
        OUTPUT_DIR
        /
        "cnn_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False
    )

    # ==============================================================================
    # SAVE SUMMARY
    # ==============================================================================

    summary_df = pd.DataFrame(
        [
            {
                "experiment": (
                    "Plain 1D CNN"
                ),

                "representation": (
                    "Channel-domain"
                ),

                "protocol": (
                    "10 train + "
                    "1 validation + "
                    "1 test"
                ),

                "subjects": (
                    len(unique_subjects)
                ),

                "classes": (
                    NUM_CLASSES
                ),

                "channels": (
                    NUM_CHANNELS
                ),

                "mean_train_accuracy": (
                    mean_train_accuracy
                ),

                "mean_validation_accuracy": (
                    mean_validation_accuracy
                ),

                "mean_test_accuracy": (
                    mean_test_accuracy
                ),

                "std_test_accuracy": (
                    std_test_accuracy
                ),

                "mean_macro_f1": (
                    mean_macro_f1
                ),

                "std_macro_f1": (
                    std_macro_f1
                ),

                "mean_balanced_accuracy": (
                    mean_balanced_accuracy
                ),

                "chance_accuracy": (
                    CHANCE_ACCURACY
                ),

                "improvement_over_chance_pp": (
                    improvement_over_chance
                ),

                "total_runtime_minutes": (
                    total_runtime / 60.0
                ),
            }
        ]
    )

    summary_path = (
        OUTPUT_DIR
        /
        "cnn_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    # ==============================================================================
    # SAVE TRAINING HISTORY
    # ==============================================================================

    history_path = (
        OUTPUT_DIR
        /
        "cnn_training_history.csv"
    )

    history_df.to_csv(
        history_path,
        index=False
    )

    # ==============================================================================
    # SAVE SPLIT SUMMARY
    # ==============================================================================

    split_path = (
        OUTPUT_DIR
        /
        "cnn_split_summary.csv"
    )

    split_df.to_csv(
        split_path,
        index=False
    )

    # ==============================================================================
    # SAVE CONFUSION MATRIX
    # ==============================================================================

    confusion_df = pd.DataFrame(
        global_confusion_matrix,

        index=[
            f"true_{label}"
            for label in EXPECTED_LABELS
        ],

        columns=[
            f"pred_{label}"
            for label in EXPECTED_LABELS
        ],
    )

    confusion_path = (
        OUTPUT_DIR
        /
        "cnn_confusion_matrix.csv"
    )

    confusion_df.to_csv(
        confusion_path
    )

    # ==============================================================================
    # SAVE CLASSIFICATION REPORTS
    # ==============================================================================

    reports_path = (
        OUTPUT_DIR
        /
        "cnn_classification_reports.txt"
    )

    with open(
        reports_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "EXPERIMENT 5: PLAIN 1D CNN\n"
        )

        file.write(
            "=" * 80
            +
            "\n\n"
        )

        file.write(
            "Dataset labels: 1..8\n"
        )

        file.write(
            "PyTorch labels: 0..7\n"
        )

        file.write(
            "Protocol: "
            "10 train + 1 validation + 1 test\n"
        )

        file.write(
            f"Mean test accuracy: "
            f"{mean_test_accuracy:.3f}% "
            f"+/- "
            f"{std_test_accuracy:.3f}%\n"
        )

        file.write(
            f"Mean Macro F1: "
            f"{mean_macro_f1:.3f}% "
            f"+/- "
            f"{std_macro_f1:.3f}%\n"
        )

        file.write(
            f"Mean balanced accuracy: "
            f"{mean_balanced_accuracy:.3f}%\n"
        )

        file.write(
            "\n"
        )

        for item in classification_reports:

            file.write(
                "\n"
                +
                "-" * 80
                +
                "\n"
            )

            file.write(
                f"Fold {item['fold']}\n"
            )

            file.write(
                f"Validation subject: "
                f"{item['validation_subject']}\n"
            )

            file.write(
                f"Test subject: "
                f"{item['test_subject']}\n\n"
            )

            file.write(
                item["report"]
            )

            file.write(
                "\n"
            )

    # ==============================================================================
    # SAVE METADATA
    # ==============================================================================

    metadata = {
        "experiment": (
            "Experiment 5 - Plain 1D CNN"
        ),

        "dataset": str(
            DATA_PATH
        ),

        "representation": (
            "Channel-domain"
        ),

        "temporal_signal_available": (
            False
        ),

        "num_subjects": (
            len(unique_subjects)
        ),

        "num_classes": (
            NUM_CLASSES
        ),

        "num_channels": (
            NUM_CHANNELS
        ),

        "samples": (
            len(df)
        ),

        "protocol": (
            "10 train + "
            "1 validation + "
            "1 test"
        ),

        "scaler": (
            "RobustScaler"
        ),

        "scaler_fit_scope": (
            "Training subjects only"
        ),

        "model": (
            "Plain 1D CNN"
        ),

        "conv1": (
            "Conv1d(1,32,kernel_size=3,padding=1)"
        ),

        "conv2": (
            "Conv1d(32,64,kernel_size=3,padding=1)"
        ),

        "conv3": (
            "Conv1d(64,128,kernel_size=3,padding=1)"
        ),

        "pooling": (
            "MaxPool1d + AdaptiveAvgPool1d"
        ),

        "dropout": (
            0.30
        ),

        "optimizer": (
            "AdamW"
        ),

        "learning_rate": (
            LEARNING_RATE
        ),

        "weight_decay": (
            WEIGHT_DECAY
        ),

        "batch_size": (
            BATCH_SIZE
        ),

        "max_epochs": (
            MAX_EPOCHS
        ),

        "early_stopping_patience": (
            PATIENCE
        ),

        "loss": (
            "CrossEntropyLoss"
        ),

        "device": (
            str(DEVICE)
        ),

        "pytorch_version": (
            torch.__version__
        ),

        "chance_accuracy": (
            CHANCE_ACCURACY
        ),

        "mean_train_accuracy": (
            mean_train_accuracy
        ),

        "mean_validation_accuracy": (
            mean_validation_accuracy
        ),

        "mean_test_accuracy": (
            mean_test_accuracy
        ),

        "std_test_accuracy": (
            std_test_accuracy
        ),

        "mean_macro_f1": (
            mean_macro_f1
        ),

        "std_macro_f1": (
            std_macro_f1
        ),

        "mean_balanced_accuracy": (
            mean_balanced_accuracy
        ),

        "improvement_over_chance_pp": (
            improvement_over_chance
        ),

        "runtime_minutes": (
            total_runtime / 60.0
        ),

        "label_conversion": (
            "Dataset 1..8 -> PyTorch 0..7"
        ),
    }

    metadata_df = pd.DataFrame(
        [
            {
                "parameter": key,
                "value": value,
            }
            for key, value in metadata.items()
        ]
    )

    metadata_path = (
        OUTPUT_DIR
        /
        "cnn_experiment_metadata.csv"
    )

    metadata_df.to_csv(
        metadata_path,
        index=False
    )

    # ==============================================================================
    # OUTPUT PATHS
    # ==============================================================================

    print("\n")

    print_header(
        "OUTPUT FILES"
    )

    print(
        f"Results:\n"
        f"{results_path}"
    )

    print(
        f"\nPredictions:\n"
        f"{predictions_path}"
    )

    print(
        f"\nSummary:\n"
        f"{summary_path}"
    )

    print(
        f"\nTraining history:\n"
        f"{history_path}"
    )

    print(
        f"\nSplit summary:\n"
        f"{split_path}"
    )

    print(
        f"\nConfusion matrix:\n"
        f"{confusion_path}"
    )

    print(
        f"\nClassification reports:\n"
        f"{reports_path}"
    )

    print(
        f"\nMetadata:\n"
        f"{metadata_path}"
    )

    # ==============================================================================
    # FINAL STATUS
    # ==============================================================================

    print("\n")

    print_separator()

    print(
        "EXPERIMENT 5 COMPLETED SUCCESSFULLY"
    )

    print_separator()

    print(
        f"\nFinal test accuracy: "
        f"{mean_test_accuracy:.3f}% "
        f"± "
        f"{std_test_accuracy:.3f}%"
    )

    print(
        f"Final Macro F1: "
        f"{mean_macro_f1:.3f}% "
        f"± "
        f"{std_macro_f1:.3f}%"
    )

    print(
        f"Chance: "
        f"{CHANCE_ACCURACY:.3f}%"
    )

    print(
        f"Difference from chance: "
        f"{improvement_over_chance:+.3f} pp"
    )

    print_separator()


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":

    main()