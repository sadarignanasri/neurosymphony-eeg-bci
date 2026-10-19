"""
NEURO-SYMPHONY
EEG Transformer — CLS Token Architecture

Purpose:
    Test whether a learnable CLS token provides a better
    global representation than mean pooling.

Input:
    data/processed/ds004514_eeg_approved.csv

Representation:
    One trial -> 64 EEG channel tokens

Evaluation:
    12-fold Leave-One-Subject-Out (LOSO)

Important:
    This CSV contains trial-level channel values, not
    the original temporal EEG waveform.
"""

from pathlib import Path
import time
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

DATA_FILE = Path(
    "data/processed/ds004514_eeg_approved.csv"
)

OUTPUT_DIR = Path(
    "results/eeg/transformer"
)

RESULTS_FILE = OUTPUT_DIR / "transformer_cls_results.csv"
PREDICTIONS_FILE = OUTPUT_DIR / "transformer_cls_predictions.csv"

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


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DEVICE
# ============================================================

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_dataset(df):
    print("=" * 80)
    print("DATASET VALIDATION")
    print("=" * 80)

    required_columns = [
        "subject_id",
        "trial_id",
        "label_name",
        "label",
    ]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    channel_columns = [
        col for col in df.columns
        if col.startswith("A") or col.startswith("B")
    ]

    if len(channel_columns) != N_CHANNELS:
        raise ValueError(
            f"Expected {N_CHANNELS} EEG channels, "
            f"found {len(channel_columns)}"
        )

    print(f"Dataset shape: {df.shape}")
    print("✓ Required columns present")
    print(f"✓ {len(channel_columns)} EEG channels detected")

    if df[channel_columns].isna().any().any():
        raise ValueError("Missing EEG values detected")

    print("✓ No missing EEG values")

    if np.isinf(df[channel_columns].to_numpy()).any():
        raise ValueError("Infinite EEG values detected")

    print("✓ No infinite EEG values")

    subjects = sorted(df["subject_id"].unique())

    if len(subjects) != 12:
        raise ValueError(
            f"Expected 12 subjects, found {len(subjects)}"
        )

    print(f"✓ Subjects: {len(subjects)}")

    classes = sorted(df["label"].unique())

    if len(classes) != N_CLASSES:
        raise ValueError(
            f"Expected {N_CLASSES} classes, found {len(classes)}"
        )

    print(f"✓ Classes: {len(classes)}")
    print(f"✓ Labels: {classes}")

    duplicate_count = df.duplicated(
        subset=["subject_id", "trial_id"]
    ).sum()

    if duplicate_count != 0:
        raise ValueError(
            f"Found {duplicate_count} duplicate subject/trial keys"
        )

    print("✓ No duplicate subject/trial keys")
    print()
    print("✓ DATASET VALIDATION PASSED")
    print()

    return channel_columns, subjects


# ============================================================
# TRANSFORMER MODEL
# ============================================================

class CLSTransformer(nn.Module):

    def __init__(
        self,
        n_channels=N_CHANNELS,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        n_classes=N_CLASSES,
    ):
        super().__init__()

        # Convert each scalar EEG channel value into
        # a D_MODEL-dimensional token.
        self.channel_projection = nn.Linear(
            1,
            d_model
        )

        # Learnable CLS token.
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, d_model)
        )

        # 64 EEG channels + 1 CLS token.
        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                n_channels + 1,
                d_model
            )
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        self.norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):

        nn.init.normal_(
            self.cls_token,
            mean=0.0,
            std=0.02
        )

        nn.init.normal_(
            self.position_embedding,
            mean=0.0,
            std=0.02
        )

    def forward(self, x):

        # x:
        # [batch, 64]

        x = x.unsqueeze(-1)

        # [batch, 64, 1]
        x = self.channel_projection(x)

        # [batch, 64, d_model]

        batch_size = x.size(0)

        cls_tokens = self.cls_token.expand(
            batch_size,
            -1,
            -1
        )

        # [batch, 1, d_model]

        x = torch.cat(
            [cls_tokens, x],
            dim=1
        )

        # [batch, 65, d_model]

        x = x + self.position_embedding

        x = self.transformer(x)

        # Take CLS representation.
        cls_output = x[:, 0]

        cls_output = self.norm(cls_output)

        logits = self.classifier(cls_output)

        return logits


# ============================================================
# TRAINING
# ============================================================

def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    device,
):

    train_dataset = TensorDataset(
        torch.tensor(
            X_train,
            dtype=torch.float32
        ),
        torch.tensor(
            y_train,
            dtype=torch.long
        ),
    )

    val_dataset = TensorDataset(
        torch.tensor(
            X_val,
            dtype=torch.float32
        ),
        torch.tensor(
            y_val,
            dtype=torch.long
        ),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    epochs_used = 0

    for epoch in range(MAX_EPOCHS):

        model.train()

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            logits = model(X_batch)

            loss = criterion(
                logits,
                y_batch
            )

            loss.backward()

            optimizer.step()

        # ------------------------------
        # Validation
        # ------------------------------

        model.eval()

        val_losses = []

        with torch.no_grad():

            for X_batch, y_batch in val_loader:

                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(X_batch)

                loss = criterion(
                    logits,
                    y_batch
                )

                val_losses.append(
                    loss.item()
                )

        val_loss = np.mean(val_losses)

        epochs_used = epoch + 1

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, epochs_used


# ============================================================
# PREDICTION
# ============================================================

def predict(model, X, device):

    dataset = TensorDataset(
        torch.tensor(
            X,
            dtype=torch.float32
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    predictions = []

    model.eval()

    with torch.no_grad():

        for (X_batch,) in loader:

            X_batch = X_batch.to(device)

            logits = model(X_batch)

            preds = torch.argmax(
                logits,
                dim=1
            )

            predictions.extend(
                preds.cpu().numpy()
            )

    return np.asarray(predictions)


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    set_seed()

    device = get_device()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 80)
    print("NEURO-SYMPHONY — EEG TRANSFORMER CLS ARCHITECTURE")
    print("=" * 80)
    print()
    print(f"Device: {device}")
    print()
    print(f"Input file:")
    print(DATA_FILE.resolve())
    print()
    print(f"Output directory:")
    print(OUTPUT_DIR.resolve())
    print()

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    channel_columns, subjects = validate_dataset(df)

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    print("=" * 80)
    print("CLS TRANSFORMER CONFIGURATION")
    print("=" * 80)

    print(f"Channels: {N_CHANNELS}")
    print(f"d_model: {D_MODEL}")
    print(f"Attention heads: {N_HEADS}")
    print(f"Transformer layers: {N_LAYERS}")
    print(f"Feedforward dimension: {DIM_FEEDFORWARD}")
    print(f"Dropout: {DROPOUT}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Maximum epochs: {MAX_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Early stopping patience: {PATIENCE}")

    print()
    print("Validation: 12-fold Leave-One-Subject-Out")
    print()
    print("Representation:")
    print("One trial → [CLS] + 64 EEG channel tokens")
    print()
    print("Classification:")
    print("CLS token → classifier → 8 classes")
    print()
    print("IMPORTANT:")
    print(
        "This CSV contains trial-level channel values, "
        "not the original EEG temporal waveform."
    )
    print()

    # --------------------------------------------------------
    # Prepare arrays
    # --------------------------------------------------------

    X = df[channel_columns].to_numpy(
        dtype=np.float32
    )

    # Convert labels 1-8 → 0-7.
    y = (
        df["label"].to_numpy(
            dtype=np.int64
        ) - 1
    )

    subject_array = df["subject_id"].to_numpy()

    # --------------------------------------------------------
    # LOSO
    # --------------------------------------------------------

    all_results = []
    all_predictions = []

    total_start = time.time()

    for fold_idx, test_subject in enumerate(
        subjects,
        start=1
    ):

        fold_start = time.time()

        print("=" * 80)
        print(
            f"FOLD {fold_idx}/{len(subjects)}"
        )
        print(
            f"Test subject: {test_subject}"
        )
        print("=" * 80)

        train_mask = (
            subject_array != test_subject
        )

        test_mask = (
            subject_array == test_subject
        )

        X_train_full = X[train_mask]
        y_train_full = y[train_mask]

        X_test = X[test_mask]
        y_test = y[test_mask]

        # ----------------------------------------------------
        # Inner validation split
        #
        # Use the last subject in the training set
        # as validation.
        # ----------------------------------------------------

        training_subjects = [
            s for s in subjects
            if s != test_subject
        ]

        validation_subject = training_subjects[-1]

        inner_train_mask = (
            subject_array != test_subject
        ) & (
            subject_array != validation_subject
        )

        validation_mask = (
            subject_array == validation_subject
        )

        X_train = X[inner_train_mask]
        y_train = y[inner_train_mask]

        X_val = X[validation_mask]
        y_val = y[validation_mask]

        # ----------------------------------------------------
        # Standardization
        #
        # Fit ONLY on inner training subjects.
        # ----------------------------------------------------

        scaler = StandardScaler()

        X_train = scaler.fit_transform(
            X_train
        ).astype(np.float32)

        X_val = scaler.transform(
            X_val
        ).astype(np.float32)

        X_test_scaled = scaler.transform(
            X_test
        ).astype(np.float32)

        print(
            f"Training samples: {len(X_train)}"
        )

        print(
            f"Validation samples: {len(X_val)}"
        )

        print(
            f"Test samples: {len(X_test_scaled)}"
        )

        print(
            f"Validation subject: {validation_subject}"
        )

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        set_seed(SEED)

        model = CLSTransformer()

        model, epochs_used = train_model(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            device,
        )

        # ----------------------------------------------------
        # Test
        # ----------------------------------------------------

        predictions = predict(
            model,
            X_test_scaled,
            device,
        )

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        macro_f1 = f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        )

        balanced_acc = balanced_accuracy_score(
            y_test,
            predictions
        )

        elapsed = time.time() - fold_start

        print()
        print(
            f"Fold accuracy: {accuracy * 100:.3f}%"
        )

        print(
            f"Fold Macro F1: {macro_f1 * 100:.3f}%"
        )

        print(
            f"Fold balanced accuracy: "
            f"{balanced_acc * 100:.3f}%"
        )

        print(
            f"Epochs used: {epochs_used}"
        )

        print(
            f"Time: {elapsed:.1f}s"
        )

        all_results.append(
            {
                "fold": fold_idx,
                "test_subject": test_subject,
                "validation_subject": validation_subject,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "balanced_accuracy": balanced_acc,
                "epochs_used": epochs_used,
                "time_seconds": elapsed,
            }
        )

        for i, prediction in enumerate(
            predictions
        ):

            all_predictions.append(
                {
                    "subject_id": test_subject,
                    "true_label": int(y_test[i] + 1),
                    "predicted_label": int(
                        prediction + 1
                    ),
                }
            )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        all_results
    )

    predictions_df = pd.DataFrame(
        all_predictions
    )

    results_df.to_csv(
        RESULTS_FILE,
        index=False
    )

    predictions_df.to_csv(
        PREDICTIONS_FILE,
        index=False
    )

    accuracy_mean = results_df[
        "accuracy"
    ].mean()

    accuracy_std = results_df[
        "accuracy"
    ].std(ddof=0)

    f1_mean = results_df[
        "macro_f1"
    ].mean()

    f1_std = results_df[
        "macro_f1"
    ].std(ddof=0)

    balanced_mean = results_df[
        "balanced_accuracy"
    ].mean()

    balanced_std = results_df[
        "balanced_accuracy"
    ].std(ddof=0)

    chance = 1 / N_CLASSES

    total_time = (
        time.time() - total_start
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("CLS TRANSFORMER EXPERIMENT COMPLETE")
    print("=" * 80)
    print()

    print(
        f"Accuracy: "
        f"{accuracy_mean * 100:.3f}% "
        f"± {accuracy_std * 100:.3f}%"
    )

    print(
        f"Macro F1: "
        f"{f1_mean * 100:.3f}% "
        f"± {f1_std * 100:.3f}%"
    )

    print(
        f"Balanced Accuracy: "
        f"{balanced_mean * 100:.3f}% "
        f"± {balanced_std * 100:.3f}%"
    )

    print()

    print(
        f"Chance level: "
        f"{chance * 100:.3f}%"
    )

    print(
        f"Improvement over chance: "
        f"{(accuracy_mean - chance) * 100:+.3f} pp"
    )

    print()

    print(
        f"Total runtime: "
        f"{total_time / 60:.2f} minutes"
    )

    print()
    print("Results:")
    print(RESULTS_FILE.resolve())

    print()
    print("Predictions:")
    print(PREDICTIONS_FILE.resolve())

    print()
    print("=" * 80)
    print("25–30% TARGET")
    print("=" * 80)

    if accuracy_mean >= 0.30:
        print("30% target reached.")

    elif accuracy_mean >= 0.25:
        print("25% target reached.")

    else:
        print(
            "25–30% target not reached with "
            "the CLS Transformer."
        )

    print()
    print("Done.")


if __name__ == "__main__":
    main()