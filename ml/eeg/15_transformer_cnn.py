"""
NEURO-SYMPHONY
EEG CNN + Transformer

Representation:
    One trial -> 64 EEG channel values

Architecture:
    Channel values
        ↓
    1D CNN feature extraction
        ↓
    Transformer encoder
        ↓
    Global average pooling
        ↓
    Classifier

Evaluation:
    12-fold LOSO
    Inner validation subject
"""

from pathlib import Path
import time
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import (
    DataLoader,
    TensorDataset
)

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
)


SEED = 42

DATA_FILE = Path(
    "data/processed/ds004514_eeg_approved.csv"
)

OUTPUT_DIR = Path(
    "results/eeg/transformer"
)

RESULTS_FILE = OUTPUT_DIR / (
    "transformer_cnn_results.csv"
)

PREDICTIONS_FILE = OUTPUT_DIR / (
    "transformer_cnn_predictions.csv"
)

N_CHANNELS = 64
N_CLASSES = 8

D_MODEL = 64
N_HEADS = 8
N_LAYERS = 2
DIM_FEEDFORWARD = 128

DROPOUT = 0.2

BATCH_SIZE = 64
MAX_EPOCHS = 50

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

PATIENCE = 8


def set_seed(seed=SEED):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def validate_dataset(df):

    required = [
        "subject_id",
        "trial_id",
        "label_name",
        "label",
    ]

    for column in required:

        if column not in df.columns:
            raise ValueError(
                f"Missing column: {column}"
            )

    channel_columns = [
        column
        for column in df.columns
        if column.startswith("A")
        or column.startswith("B")
    ]

    if len(channel_columns) != 64:

        raise ValueError(
            f"Expected 64 channels, "
            f"found {len(channel_columns)}"
        )

    if df[channel_columns].isna().any().any():

        raise ValueError(
            "Missing EEG values"
        )

    if np.isinf(
        df[channel_columns].to_numpy()
    ).any():

        raise ValueError(
            "Infinite EEG values"
        )

    subjects = sorted(
        df["subject_id"].unique()
    )

    classes = sorted(
        df["label"].unique()
    )

    if len(subjects) != 12:
        raise ValueError(
            "Expected 12 subjects"
        )

    if len(classes) != 8:
        raise ValueError(
            "Expected 8 classes"
        )

    if df.duplicated(
        subset=[
            "subject_id",
            "trial_id"
        ]
    ).any():

        raise ValueError(
            "Duplicate subject/trial keys"
        )

    print(
        f"Dataset shape: {df.shape}"
    )

    print(
        f"Channels: {len(channel_columns)}"
    )

    print(
        f"Subjects: {len(subjects)}"
    )

    print(
        f"Classes: {len(classes)}"
    )

    print(
        "✓ DATASET VALIDATION PASSED"
    )

    return channel_columns, subjects


class CNNTransformer(nn.Module):

    def __init__(
        self,
        n_channels=64,
        d_model=64,
        n_heads=8,
        n_layers=2,
        dim_feedforward=128,
        dropout=0.2,
        n_classes=8,
    ):

        super().__init__()

        # ----------------------------------------------------
        # CNN feature extractor
        # ----------------------------------------------------

        self.cnn = nn.Sequential(

            nn.Conv1d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm1d(32),

            nn.GELU(),

            nn.Conv1d(
                32,
                64,
                kernel_size=5,
                padding=2
            ),

            nn.BatchNorm1d(64),

            nn.GELU(),

            nn.Conv1d(
                64,
                d_model,
                kernel_size=3,
                padding=1
            ),

            nn.GELU(),

        )

        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                n_channels,
                d_model
            )
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        self.norm = nn.LayerNorm(
            d_model
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                d_model,
                d_model
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                d_model,
                n_classes
            )

        )

        nn.init.normal_(
            self.position_embedding,
            mean=0.0,
            std=0.02
        )

    def forward(self, x):

        # [batch, 64]

        x = x.unsqueeze(1)

        # [batch, 1, 64]

        x = self.cnn(x)

        # [batch, d_model, 64]

        x = x.transpose(1, 2)

        # [batch, 64, d_model]

        x = (
            x +
            self.position_embedding
        )

        x = self.transformer(x)

        # Global average pooling.
        x = x.mean(dim=1)

        x = self.norm(x)

        return self.classifier(x)


def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    device
):

    train_dataset = TensorDataset(
        torch.tensor(
            X_train,
            dtype=torch.float32
        ),
        torch.tensor(
            y_train,
            dtype=torch.long
        )
    )

    val_dataset = TensorDataset(
        torch.tensor(
            X_val,
            dtype=torch.float32
        ),
        torch.tensor(
            y_val,
            dtype=torch.long
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    best_loss = float("inf")
    best_state = None

    no_improvement = 0
    epochs_used = 0

    for epoch in range(MAX_EPOCHS):

        model.train()

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            logits = model(
                X_batch
            )

            loss = criterion(
                logits,
                y_batch
            )

            loss.backward()

            optimizer.step()

        model.eval()

        losses = []

        with torch.no_grad():

            for X_batch, y_batch in val_loader:

                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(
                    X_batch
                )

                loss = criterion(
                    logits,
                    y_batch
                )

                losses.append(
                    loss.item()
                )

        val_loss = np.mean(
            losses
        )

        epochs_used = epoch + 1

        if val_loss < best_loss:

            best_loss = val_loss

            best_state = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }

            no_improvement = 0

        else:

            no_improvement += 1

        if no_improvement >= PATIENCE:

            break

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    return model, epochs_used


def predict(
    model,
    X,
    device
):

    dataset = TensorDataset(
        torch.tensor(
            X,
            dtype=torch.float32
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    predictions = []

    model.eval()

    with torch.no_grad():

        for (X_batch,) in loader:

            X_batch = X_batch.to(device)

            logits = model(
                X_batch
            )

            predictions.extend(
                torch.argmax(
                    logits,
                    dim=1
                ).cpu().numpy()
            )

    return np.asarray(
        predictions
    )


def main():

    set_seed()

    device = get_device()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 80)
    print(
        "NEURO-SYMPHONY — "
        "CNN + TRANSFORMER"
    )
    print("=" * 80)

    print(
        f"Device: {device}"
    )

    print(
        f"Input: {DATA_FILE.resolve()}"
    )

    df = pd.read_csv(
        DATA_FILE
    )

    channel_columns, subjects = (
        validate_dataset(df)
    )

    X = df[
        channel_columns
    ].to_numpy(
        dtype=np.float32
    )

    y = (
        df["label"].to_numpy(
            dtype=np.int64
        ) - 1
    )

    subject_array = (
        df["subject_id"].to_numpy()
    )

    results = []
    predictions_all = []

    total_start = time.time()

    for fold_idx, test_subject in enumerate(
        subjects,
        start=1
    ):

        fold_start = time.time()

        print()
        print("=" * 80)
        print(
            f"FOLD {fold_idx}/{len(subjects)}"
        )
        print(
            f"Test subject: {test_subject}"
        )
        print("=" * 80)

        training_subjects = [
            subject
            for subject in subjects
            if subject != test_subject
        ]

        validation_subject = (
            training_subjects[-1]
        )

        train_mask = (
            (subject_array != test_subject)
            &
            (
                subject_array
                != validation_subject
            )
        )

        val_mask = (
            subject_array
            == validation_subject
        )

        test_mask = (
            subject_array
            == test_subject
        )

        X_train = X[
            train_mask
        ]

        y_train = y[
            train_mask
        ]

        X_val = X[
            val_mask
        ]

        y_val = y[
            val_mask
        ]

        X_test = X[
            test_mask
        ]

        y_test = y[
            test_mask
        ]

        scaler = StandardScaler()

        X_train = scaler.fit_transform(
            X_train
        ).astype(np.float32)

        X_val = scaler.transform(
            X_val
        ).astype(np.float32)

        X_test = scaler.transform(
            X_test
        ).astype(np.float32)

        print(
            f"Training samples: "
            f"{len(X_train)}"
        )

        print(
            f"Validation samples: "
            f"{len(X_val)}"
        )

        print(
            f"Test samples: "
            f"{len(X_test)}"
        )

        print(
            f"Validation subject: "
            f"{validation_subject}"
        )

        set_seed()

        model = CNNTransformer()

        model, epochs_used = train_model(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            device
        )

        predictions = predict(
            model,
            X_test,
            device
        )

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        macro_f1 = f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )

        balanced = balanced_accuracy_score(
            y_test,
            predictions
        )

        elapsed = (
            time.time()
            - fold_start
        )

        print(
            f"Fold accuracy: "
            f"{accuracy * 100:.3f}%"
        )

        print(
            f"Fold Macro F1: "
            f"{macro_f1 * 100:.3f}%"
        )

        print(
            f"Fold balanced accuracy: "
            f"{balanced * 100:.3f}%"
        )

        print(
            f"Epochs used: "
            f"{epochs_used}"
        )

        print(
            f"Time: "
            f"{elapsed:.1f}s"
        )

        results.append(
            {
                "fold": fold_idx,
                "test_subject": test_subject,
                "validation_subject":
                    validation_subject,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "balanced_accuracy":
                    balanced,
                "epochs_used":
                    epochs_used,
                "time_seconds":
                    elapsed
            }
        )

        for true_label, predicted_label in zip(
            y_test,
            predictions
        ):

            predictions_all.append(
                {
                    "subject_id":
                        test_subject,
                    "true_label":
                        int(true_label + 1),
                    "predicted_label":
                        int(
                            predicted_label + 1
                        )
                }
            )

    results_df = pd.DataFrame(
        results
    )

    predictions_df = pd.DataFrame(
        predictions_all
    )

    results_df.to_csv(
        RESULTS_FILE,
        index=False
    )

    predictions_df.to_csv(
        PREDICTIONS_FILE,
        index=False
    )

    accuracy_mean = (
        results_df["accuracy"].mean()
    )

    accuracy_std = (
        results_df["accuracy"].std(
            ddof=0
        )
    )

    f1_mean = (
        results_df["macro_f1"].mean()
    )

    f1_std = (
        results_df["macro_f1"].std(
            ddof=0
        )
    )

    balanced_mean = (
        results_df[
            "balanced_accuracy"
        ].mean()
    )

    balanced_std = (
        results_df[
            "balanced_accuracy"
        ].std(
            ddof=0
        )
    )

    chance = 1 / N_CLASSES

    total_time = (
        time.time()
        - total_start
    )

    print()
    print("=" * 80)
    print(
        "CNN + TRANSFORMER "
        "EXPERIMENT COMPLETE"
    )
    print("=" * 80)

    print(
        f"Accuracy: "
        f"{accuracy_mean * 100:.3f}% "
        f"± "
        f"{accuracy_std * 100:.3f}%"
    )

    print(
        f"Macro F1: "
        f"{f1_mean * 100:.3f}% "
        f"± "
        f"{f1_std * 100:.3f}%"
    )

    print(
        f"Balanced Accuracy: "
        f"{balanced_mean * 100:.3f}% "
        f"± "
        f"{balanced_std * 100:.3f}%"
    )

    print(
        f"Chance level: "
        f"{chance * 100:.3f}%"
    )

    print(
        f"Improvement over chance: "
        f"{(accuracy_mean - chance) * 100:+.3f} pp"
    )

    print(
        f"Runtime: "
        f"{total_time / 60:.2f} minutes"
    )

    print()
    print(
        f"Results: "
        f"{RESULTS_FILE.resolve()}"
    )

    print(
        f"Predictions: "
        f"{PREDICTIONS_FILE.resolve()}"
    )

    print()
    print("Done.")


if __name__ == "__main__":
    main()