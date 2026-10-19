
"""
NEURO-SYMPHONY
Experiment 1 — EEGNet-Inspired Channel-Domain CNN

Dataset:
    data/processed/ds004514_eeg_approved.csv

Important:
    The current CSV contains one row per trial with 64 channel values.
    It does NOT contain the original temporal EEG samples.

Therefore this experiment is an EEGNet-inspired CHANNEL-DOMAIN CNN.
It must NOT be interpreted as conventional temporal EEGNet.

Evaluation:
    Strict subject-independent 10/1/1 split

    10 subjects -> training
    1 subject   -> validation
    1 subject   -> testing

Across 12 folds:
    Every subject becomes the test subject exactly once.

Metrics:
    Train accuracy
    Validation accuracy
    Test accuracy
    Macro F1
    Balanced accuracy
    Classification report
    Confusion matrix

No test data is used during training or validation.
"""

from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import RobustScaler


# =============================================================================
# CONFIGURATION
# =============================================================================

SEED = 42

DATA_PATH = Path("data/processed/ds004514_eeg_approved.csv")

RESULTS_DIR = Path("results/eeg/eegnet")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 8
NUM_CHANNELS = 64

EPOCHS = 50
BATCH_SIZE = 64

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

PATIENCE = 8

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# DATASET VALIDATION
# =============================================================================

def validate_dataset(df):
    print("=" * 80)
    print("DATASET VALIDATION")
    print("=" * 80)

    eeg_columns = [f"A{i}" for i in range(1, 33)] + \
                  [f"B{i}" for i in range(1, 33)]

    required_columns = (
        ["subject_id", "trial_id", "label_name", "label"]
        + eeg_columns
    )

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if len(eeg_columns) != NUM_CHANNELS:
        raise ValueError(
            f"Expected {NUM_CHANNELS} EEG channels, "
            f"found {len(eeg_columns)}"
        )

    if df[eeg_columns].isna().any().any():
        raise ValueError("Missing values detected in EEG channels.")

    if not np.isfinite(
        df[eeg_columns].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError("Infinite values detected in EEG channels.")

    subjects = sorted(df["subject_id"].unique())

    if len(subjects) != 12:
        raise ValueError(
            f"Expected 12 subjects, found {len(subjects)}"
        )

    labels = sorted(df["label"].unique())

    if labels != list(range(1, NUM_CLASSES + 1)):
        raise ValueError(
            f"Expected labels 1-{NUM_CLASSES}, found {labels}"
        )

    subject_counts = df.groupby("subject_id").size()

    if not (subject_counts == 720).all():
        raise ValueError(
            "Each subject must contain exactly 720 trials."
        )

    class_counts = df.groupby("label").size()

    if not (class_counts == 1080).all():
        raise ValueError(
            "Each class must contain exactly 1080 trials."
        )

    subject_class_counts = (
        df.groupby(["subject_id", "label"])
        .size()
        .unique()
    )

    if len(subject_class_counts) != 1 or subject_class_counts[0] != 90:
        raise ValueError(
            "Each subject/class combination must contain 90 trials."
        )

    duplicate_trials = df.duplicated(
        subset=["subject_id", "trial_id"]
    ).sum()

    if duplicate_trials != 0:
        raise ValueError(
            f"Found {duplicate_trials} duplicate subject/trial pairs."
        )

    print(f"Dataset shape: {df.shape}")
    print(f"EEG channels: {len(eeg_columns)}")
    print(f"Subjects: {len(subjects)}")
    print(f"Classes: {len(labels)}")
    print("✓ DATASET VALIDATION PASSED")

    return eeg_columns, subjects


# =============================================================================
# EEGNET-INSPIRED CHANNEL-DOMAIN MODEL
# =============================================================================

class EEGNetChannelDomain(nn.Module):
    """
    EEGNet-inspired architecture for the current trial-level CSV.

    Input:
        [batch, 64]

    Reshaped to:
        [batch, 1, 64]

    The architecture uses:
        - temporal-like 1D convolution across channel positions
        - depthwise channel processing
        - separable convolution
        - average pooling
        - dropout
        - classification head

    IMPORTANT:
        The 64 positions are EEG CHANNELS, not temporal samples.
    """

    def __init__(self, num_channels=64, num_classes=8):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                padding=2,
                bias=False,
            ),
            nn.BatchNorm1d(16),
            nn.ELU(),
        )

        self.depthwise = nn.Sequential(
            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1,
                groups=16,
                bias=False,
            ),
            nn.BatchNorm1d(32),
            nn.ELU(),
        )

        self.pool1 = nn.AvgPool1d(
            kernel_size=2,
            stride=2
        )

        self.dropout1 = nn.Dropout(0.25)

        self.separable = nn.Sequential(
            nn.Conv1d(
                in_channels=32,
                out_channels=32,
                kernel_size=5,
                padding=2,
                groups=32,
                bias=False,
            ),
            nn.BatchNorm1d(32),
            nn.ELU(),

            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm1d(64),
            nn.ELU(),
        )

        self.pool2 = nn.AvgPool1d(
            kernel_size=2,
            stride=2
        )

        self.dropout2 = nn.Dropout(0.25)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Linear(
            64,
            num_classes
        )

    def forward(self, x):

        # x shape:
        # [batch, 64]

        x = x.unsqueeze(1)

        # [batch, 1, 64]
        x = self.block1(x)

        # [batch, 16, 64]
        x = self.depthwise(x)

        # [batch, 32, 64]
        x = self.pool1(x)

        # [batch, 32, 32]
        x = self.dropout1(x)

        x = self.separable(x)

        # [batch, 64, 32]
        x = self.pool2(x)

        # [batch, 64, 16]
        x = self.dropout2(x)

        x = self.global_pool(x)

        # [batch, 64, 1]
        x = x.squeeze(-1)

        # [batch, 64]
        logits = self.classifier(x)

        return logits


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_data(df, eeg_columns, train_subjects, val_subject, test_subject):

    train_df = df[df["subject_id"].isin(train_subjects)].copy()

    val_df = df[df["subject_id"] == val_subject].copy()

    test_df = df[df["subject_id"] == test_subject].copy()

    # -------------------------------------------------------------------------
    # Extract features
    # -------------------------------------------------------------------------

    X_train = train_df[eeg_columns].to_numpy(dtype=np.float32)
    X_val = val_df[eeg_columns].to_numpy(dtype=np.float32)
    X_test = test_df[eeg_columns].to_numpy(dtype=np.float32)

    # -------------------------------------------------------------------------
    # Convert labels from 1-8 to 0-7 for PyTorch
    # -------------------------------------------------------------------------

    y_train = (
        train_df["label"].to_numpy(dtype=np.int64) - 1
    )

    y_val = (
        val_df["label"].to_numpy(dtype=np.int64) - 1
    )

    y_test = (
        test_df["label"].to_numpy(dtype=np.int64) - 1
    )

    # -------------------------------------------------------------------------
    # IMPORTANT:
    # RobustScaler is fitted ONLY on training subjects.
    # -------------------------------------------------------------------------

    scaler = RobustScaler()

    X_train = scaler.fit_transform(X_train).astype(np.float32)

    X_val = scaler.transform(X_val).astype(np.float32)

    X_test = scaler.transform(X_test).astype(np.float32)

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    )


# =============================================================================
# TENSOR DATASET
# =============================================================================

class EEGDataset(torch.utils.data.Dataset):

    def __init__(self, X, y):
        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y,
            dtype=torch.long
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


# =============================================================================
# TRAINING
# =============================================================================

def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
):

    best_val_loss = float("inf")
    best_state = None

    patience_counter = 0

    history = []

    for epoch in range(1, EPOCHS + 1):

        # ---------------------------------------------------------------------
        # TRAIN
        # ---------------------------------------------------------------------

        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()

            logits = model(X_batch)

            loss = criterion(
                logits,
                y_batch
            )

            loss.backward()

            optimizer.step()

            train_loss += (
                loss.item() * X_batch.size(0)
            )

            predictions = torch.argmax(
                logits,
                dim=1
            )

            train_correct += (
                predictions == y_batch
            ).sum().item()

            train_total += y_batch.size(0)

        train_loss /= train_total

        train_accuracy = (
            train_correct / train_total
        )

        # ---------------------------------------------------------------------
        # VALIDATION
        # ---------------------------------------------------------------------

        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for X_batch, y_batch in val_loader:

                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                logits = model(X_batch)

                loss = criterion(
                    logits,
                    y_batch
                )

                val_loss += (
                    loss.item() * X_batch.size(0)
                )

                predictions = torch.argmax(
                    logits,
                    dim=1
                )

                val_correct += (
                    predictions == y_batch
                ).sum().item()

                val_total += y_batch.size(0)

        val_loss /= val_total

        val_accuracy = (
            val_correct / val_total
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        })

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy * 100:.3f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy * 100:.3f}%"
        )

        # ---------------------------------------------------------------------
        # EARLY STOPPING
        # ---------------------------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            patience_counter = 0

        else:

            patience_counter += 1

        if patience_counter >= PATIENCE:

            print(
                f"Early stopping at epoch {epoch}"
            )

            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return pd.DataFrame(history)


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(model, X, y):

    dataset = EEGDataset(X, y)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(DEVICE)

            logits = model(X_batch)

            predictions = torch.argmax(
                logits,
                dim=1
            ).cpu().numpy()

            all_predictions.extend(
                predictions.tolist()
            )

            all_targets.extend(
                y_batch.numpy().tolist()
            )

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    accuracy = accuracy_score(
        all_targets,
        all_predictions
    )

    macro_f1 = f1_score(
        all_targets,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    balanced_accuracy = balanced_accuracy_score(
        all_targets,
        all_predictions,
    )

    return (
        accuracy,
        macro_f1,
        balanced_accuracy,
        all_targets,
        all_predictions,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    set_seed()

    print("=" * 80)
    print("NEURO-SYMPHONY — EXPERIMENT 1: EEGNET")
    print("=" * 80)

    print(f"Device: {DEVICE}")
    print(f"PyTorch version: {torch.__version__}")

    if torch.backends.mps.is_available():
        print("✓ Apple MPS available")

    print()
    print("IMPORTANT:")
    print(
        "This is an EEGNet-inspired CHANNEL-DOMAIN CNN "
        "because the current CSV contains one value per EEG channel "
        "rather than temporal EEG samples."
    )
    print()

    # -------------------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------------------

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    eeg_columns, subjects = validate_dataset(df)

    # -------------------------------------------------------------------------
    # Save class distribution
    # -------------------------------------------------------------------------

    class_distribution = (
        df.groupby("label")
        .size()
        .reset_index(name="trial_count")
    )

    class_distribution.to_csv(
        RESULTS_DIR / "class_distribution.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # LOSO / 10-1-1 evaluation
    # -------------------------------------------------------------------------

    fold_results = []
    all_predictions = []
    all_histories = []

    start_time = time.time()

    for fold_index, test_subject in enumerate(subjects, start=1):

        # -------------------------------------------------------------
        # Deterministic validation subject
        # -------------------------------------------------------------

        remaining_subjects = [
            subject
            for subject in subjects
            if subject != test_subject
        ]

        # Rotate validation subject among the remaining subjects.
        val_index = (fold_index - 1) % len(remaining_subjects)

        val_subject = remaining_subjects[val_index]

        train_subjects = [
            subject
            for subject in remaining_subjects
            if subject != val_subject
        ]

        print()
        print("=" * 80)
        print(f"FOLD {fold_index}/12")
        print("=" * 80)

        print(
            f"Train subjects ({len(train_subjects)}): "
            f"{train_subjects}"
        )

        print(
            f"Validation subject: {val_subject}"
        )

        print(
            f"Test subject: {test_subject}"
        )

        # -------------------------------------------------------------
        # Prepare data
        # -------------------------------------------------------------

        (
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
        ) = prepare_data(
            df,
            eeg_columns,
            train_subjects,
            val_subject,
            test_subject,
        )

        print()
        print(
            f"Train trials: {len(X_train)}"
        )

        print(
            f"Validation trials: {len(X_val)}"
        )

        print(
            f"Test trials: {len(X_test)}"
        )

        # -------------------------------------------------------------
        # Class distribution check
        # -------------------------------------------------------------

        print()
        print("Class counts:")

        print(
            "Train:",
            np.bincount(
                y_train,
                minlength=NUM_CLASSES
            )
        )

        print(
            "Validation:",
            np.bincount(
                y_val,
                minlength=NUM_CLASSES
            )
        )

        print(
            "Test:",
            np.bincount(
                y_test,
                minlength=NUM_CLASSES
            )
        )

        # -------------------------------------------------------------
        # DataLoaders
        # -------------------------------------------------------------

        train_dataset = EEGDataset(
            X_train,
            y_train
        )

        val_dataset = EEGDataset(
            X_val,
            y_val
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )

        # -------------------------------------------------------------
        # Model
        # -------------------------------------------------------------

        model = EEGNetChannelDomain(
            num_channels=NUM_CHANNELS,
            num_classes=NUM_CLASSES,
        ).to(DEVICE)

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        # -------------------------------------------------------------
        # Train
        # -------------------------------------------------------------

        history = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
        )

        history["fold"] = fold_index
        history["test_subject"] = test_subject
        history["validation_subject"] = val_subject

        all_histories.append(history)

        # -------------------------------------------------------------
        # Evaluate TRAIN
        # -------------------------------------------------------------

        (
            train_accuracy,
            train_f1,
            train_balanced,
            _,
            _,
        ) = evaluate_model(
            model,
            X_train,
            y_train,
        )

        # -------------------------------------------------------------
        # Evaluate VALIDATION
        # -------------------------------------------------------------

        (
            val_accuracy,
            val_f1,
            val_balanced,
            _,
            _,
        ) = evaluate_model(
            model,
            X_val,
            y_val,
        )

        # -------------------------------------------------------------
        # Evaluate TEST
        # -------------------------------------------------------------

        (
            test_accuracy,
            test_f1,
            test_balanced,
            test_targets,
            test_predictions,
        ) = evaluate_model(
            model,
            X_test,
            y_test,
        )

        # -------------------------------------------------------------
        # Store predictions
        # -------------------------------------------------------------

        test_prediction_df = df[
            df["subject_id"] == test_subject
        ][
            [
                "subject_id",
                "trial_id",
                "label_name",
                "label",
            ]
        ].copy()

        test_prediction_df["true_label_zero_based"] = (
            test_targets
        )

        test_prediction_df["predicted_label_zero_based"] = (
            test_predictions
        )

        test_prediction_df["predicted_label"] = (
            test_predictions + 1
        )

        test_prediction_df["fold"] = fold_index

        test_prediction_df["validation_subject"] = (
            val_subject
        )

        all_predictions.append(
            test_prediction_df
        )

        # -------------------------------------------------------------
        # Fold output
        # -------------------------------------------------------------

        print()
        print(
            f"Train accuracy: "
            f"{train_accuracy * 100:.3f}%"
        )

        print(
            f"Validation accuracy: "
            f"{val_accuracy * 100:.3f}%"
        )

        print(
            f"Test accuracy: "
            f"{test_accuracy * 100:.3f}%"
        )

        print(
            f"Macro F1: "
            f"{test_f1 * 100:.3f}%"
        )

        print(
            f"Balanced accuracy: "
            f"{test_balanced * 100:.3f}%"
        )

        fold_results.append({
            "fold": fold_index,
            "train_subject_count": len(train_subjects),
            "validation_subject": val_subject,
            "test_subject": test_subject,
            "train_trials": len(X_train),
            "validation_trials": len(X_val),
            "test_trials": len(X_test),
            "train_accuracy": train_accuracy,
            "validation_accuracy": val_accuracy,
            "test_accuracy": test_accuracy,
            "macro_f1": test_f1,
            "balanced_accuracy": test_balanced,
        })

        # -------------------------------------------------------------
        # Save fold model
        # -------------------------------------------------------------

        torch.save(
            model.state_dict(),
            RESULTS_DIR / f"fold_{fold_index:02d}_model.pt",
        )

    # =========================================================================
    # FINAL RESULTS
    # =========================================================================

    results_df = pd.DataFrame(fold_results)

    history_df = pd.concat(
        all_histories,
        ignore_index=True
    )

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True
    )

    # -------------------------------------------------------------------------
    # Overall statistics
    # -------------------------------------------------------------------------

    mean_train_accuracy = (
        results_df["train_accuracy"].mean()
    )

    mean_validation_accuracy = (
        results_df["validation_accuracy"].mean()
    )

    mean_test_accuracy = (
        results_df["test_accuracy"].mean()
    )

    std_test_accuracy = (
        results_df["test_accuracy"].std()
    )

    mean_macro_f1 = (
        results_df["macro_f1"].mean()
    )

    std_macro_f1 = (
        results_df["macro_f1"].std()
    )

    mean_balanced_accuracy = (
        results_df["balanced_accuracy"].mean()
    )

    chance = 1.0 / NUM_CLASSES

    test_improvement = (
        mean_test_accuracy - chance
    )

    total_runtime = (
        time.time() - start_time
    )

    # -------------------------------------------------------------------------
    # Overall classification report
    # -------------------------------------------------------------------------

    overall_true = (
        predictions_df["true_label_zero_based"]
        .to_numpy()
    )

    overall_pred = (
        predictions_df["predicted_label_zero_based"]
        .to_numpy()
    )

    report = classification_report(
        overall_true,
        overall_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=[
            str(i)
            for i in range(1, NUM_CLASSES + 1)
        ],
        zero_division=0,
    )

    # -------------------------------------------------------------------------
    # Overall confusion matrix
    # -------------------------------------------------------------------------

    cm = confusion_matrix(
        overall_true,
        overall_pred,
        labels=list(range(NUM_CLASSES)),
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            f"true_{i}"
            for i in range(1, NUM_CLASSES + 1)
        ],
        columns=[
            f"pred_{i}"
            for i in range(1, NUM_CLASSES + 1)
        ],
    )

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------

    results_df.to_csv(
        RESULTS_DIR / "eegnet_results.csv",
        index=False,
    )

    predictions_df.to_csv(
        RESULTS_DIR / "eegnet_predictions.csv",
        index=False,
    )

    history_df.to_csv(
        RESULTS_DIR / "eegnet_training_history.csv",
        index=False,
    )

    cm_df.to_csv(
        RESULTS_DIR / "eegnet_confusion_matrix.csv"
    )

    with open(
        RESULTS_DIR / "eegnet_classification_report.txt",
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "NEURO-SYMPHONY — EXPERIMENT 1 EEGNET\n"
        )

        f.write(
            "=" * 80 + "\n\n"
        )

        f.write(
            "Current CSV representation:\n"
        )

        f.write(
            "One trial = 64 channel values; "
            "no temporal samples.\n"
        )

        f.write(
            "Therefore this experiment is "
            "channel-domain EEGNet-inspired CNN.\n\n"
        )

        f.write(
            f"Mean train accuracy: "
            f"{mean_train_accuracy * 100:.3f}%\n"
        )

        f.write(
            f"Mean validation accuracy: "
            f"{mean_validation_accuracy * 100:.3f}%\n"
        )

        f.write(
            f"Mean test accuracy: "
            f"{mean_test_accuracy * 100:.3f}%\n"
        )

        f.write(
            f"Test accuracy std: "
            f"{std_test_accuracy * 100:.3f}%\n"
        )

        f.write(
            f"Mean Macro F1: "
            f"{mean_macro_f1 * 100:.3f}%\n"
        )

        f.write(
            f"Macro F1 std: "
            f"{std_macro_f1 * 100:.3f}%\n"
        )

        f.write(
            f"Mean balanced accuracy: "
            f"{mean_balanced_accuracy * 100:.3f}%\n"
        )

        f.write(
            f"Chance level: "
            f"{chance * 100:.3f}%\n"
        )

        f.write(
            f"Difference from chance: "
            f"{test_improvement * 100:+.3f} percentage points\n"
        )

        f.write(
            f"Runtime: "
            f"{total_runtime / 60:.2f} minutes\n\n"
        )

        f.write(
            "Overall classification report:\n\n"
        )

        f.write(report)

    # -------------------------------------------------------------------------
    # Save experiment summary
    # -------------------------------------------------------------------------

    summary = pd.DataFrame([{
        "experiment": "EEGNet-inspired channel-domain CNN",
        "dataset": str(DATA_PATH),
        "subjects": len(subjects),
        "classes": NUM_CLASSES,
        "channels": NUM_CHANNELS,
        "train_subjects_per_fold": 10,
        "validation_subjects_per_fold": 1,
        "test_subjects_per_fold": 1,
        "folds": len(subjects),
        "epochs_max": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "patience": PATIENCE,
        "mean_train_accuracy": mean_train_accuracy,
        "mean_validation_accuracy": mean_validation_accuracy,
        "mean_test_accuracy": mean_test_accuracy,
        "std_test_accuracy": std_test_accuracy,
        "mean_macro_f1": mean_macro_f1,
        "std_macro_f1": std_macro_f1,
        "mean_balanced_accuracy": mean_balanced_accuracy,
        "chance_accuracy": chance,
        "difference_from_chance": test_improvement,
        "runtime_minutes": total_runtime / 60,
    }])

    summary.to_csv(
        RESULTS_DIR / "eegnet_summary.csv",
        index=False,
    )

    # =========================================================================
    # FINAL TERMINAL OUTPUT
    # =========================================================================

    print()
    print("=" * 80)
    print("EXPERIMENT 1 — EEGNET COMPLETE")
    print("=" * 80)

    print(
        f"Mean train accuracy: "
        f"{mean_train_accuracy * 100:.3f}%"
    )

    print(
        f"Mean validation accuracy: "
        f"{mean_validation_accuracy * 100:.3f}%"
    )

    print(
        f"Mean test accuracy: "
        f"{mean_test_accuracy * 100:.3f}% "
        f"± {std_test_accuracy * 100:.3f}%"
    )

    print(
        f"Macro F1: "
        f"{mean_macro_f1 * 100:.3f}% "
        f"± {std_macro_f1 * 100:.3f}%"
    )

    print(
        f"Balanced accuracy: "
        f"{mean_balanced_accuracy * 100:.3f}%"
    )

    print(
        f"Chance level: "
        f"{chance * 100:.3f}%"
    )

    print(
        f"Difference from chance: "
        f"{test_improvement * 100:+.3f} percentage points"
    )

    print(
        f"Runtime: "
        f"{total_runtime / 60:.2f} minutes"
    )

    print()
    print("OUTPUT FILES")
    print("-" * 80)

    print(
        f"Results: "
        f"{RESULTS_DIR / 'eegnet_results.csv'}"
    )

    print(
        f"Predictions: "
        f"{RESULTS_DIR / 'eegnet_predictions.csv'}"
    )

    print(
        f"Training history: "
        f"{RESULTS_DIR / 'eegnet_training_history.csv'}"
    )

    print(
        f"Classification report: "
        f"{RESULTS_DIR / 'eegnet_classification_report.txt'}"
    )

    print(
        f"Confusion matrix: "
        f"{RESULTS_DIR / 'eegnet_confusion_matrix.csv'}"
    )

    print(
        f"Summary: "
        f"{RESULTS_DIR / 'eegnet_summary.csv'}"
    )

    print()
    print("Done.")


if __name__ == "__main__":
    main()