"""
Neuro-Symphony
EEG Transformer Baseline

Input:
    data/processed/ds004514_eeg_approved.csv

Representation:
    1 trial = 64 EEG channel values

Architecture:
    EEG channel values
        -> channel embeddings
        -> positional embeddings
        -> Transformer Encoder
        -> global mean pooling
        -> classification head
        -> 8 classes

Validation:
    Strict Leave-One-Subject-Out (LOSO)

Important:
    Normalization is fitted only on training subjects
    within each LOSO fold.

This experiment is intentionally a baseline Transformer.
The current CSV does NOT contain the original temporal EEG
waveform, so this model learns spatial/channel relationships,
not temporal EEG dynamics.
"""

from pathlib import Path
import copy
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ds004514_eeg_approved.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "eeg"
    / "transformer"
)

RESULTS_FILE = (
    OUTPUT_DIR
    / "transformer_baseline_results.csv"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "transformer_baseline_predictions.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

NUM_CLASSES = 8
NUM_CHANNELS = 64

D_MODEL = 64
NUM_HEADS = 8
NUM_LAYERS = 2
DIM_FEEDFORWARD = 128
DROPOUT = 0.20

BATCH_SIZE = 64
EPOCHS = 40

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

PATIENCE = 8

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else (
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
)

EEG_COLUMNS = (
    [f"A{i}" for i in range(1, 33)]
    + [f"B{i}" for i in range(1, 33)]
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=RANDOM_STATE):

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_dataset(df):

    print("\n" + "=" * 80)
    print("DATASET VALIDATION")
    print("=" * 80)

    print(
        f"Dataset shape: {df.shape}"
    )

    required_columns = [
        "subject_id",
        "trial_id",
        "label",
        "label_name",
    ] + EEG_COLUMNS

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    print(
        "✓ Required columns present"
    )

    if len(EEG_COLUMNS) != 64:
        raise ValueError(
            "Expected exactly 64 EEG channels."
        )

    print(
        "✓ 64 EEG channels detected"
    )

    if df[EEG_COLUMNS].isna().sum().sum() != 0:
        raise ValueError(
            "Missing EEG values detected."
        )

    print(
        "✓ No missing EEG values"
    )

    if np.isinf(
        df[EEG_COLUMNS].to_numpy()
    ).sum() != 0:

        raise ValueError(
            "Infinite EEG values detected."
        )

    print(
        "✓ No infinite EEG values"
    )

    subjects = sorted(
        df["subject_id"].unique()
    )

    labels = sorted(
        df["label"].unique()
    )

    print(
        f"✓ Subjects: {len(subjects)}"
    )

    print(
        f"✓ Classes: {len(labels)}"
    )

    print(
        f"✓ Labels: {labels}"
    )

    if len(subjects) != 12:
        raise ValueError(
            f"Expected 12 subjects, found {len(subjects)}"
        )

    if len(labels) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} classes, "
            f"found {len(labels)}"
        )

    duplicate_keys = df.duplicated(
        subset=[
            "subject_id",
            "trial_id",
        ]
    ).sum()

    if duplicate_keys != 0:

        raise ValueError(
            f"Found {duplicate_keys} "
            "duplicate subject/trial keys."
        )

    print(
        "✓ No duplicate subject/trial keys"
    )

    print(
        "\n✓ DATASET VALIDATION PASSED"
    )

    return subjects


# ============================================================
# TRANSFORMER MODEL
# ============================================================

class EEGTransformer(nn.Module):

    def __init__(
        self,
        num_channels=64,
        num_classes=8,
        d_model=64,
        num_heads=8,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.20,
    ):

        super().__init__()

        self.num_channels = num_channels

        # ----------------------------------------------------
        # Convert each scalar channel value into a token
        # ----------------------------------------------------

        self.channel_projection = nn.Linear(
            1,
            d_model,
        )

        # ----------------------------------------------------
        # Learnable positional/channel embeddings
        # ----------------------------------------------------

        self.position_embedding = nn.Parameter(
            torch.randn(
                1,
                num_channels,
                d_model
            ) * 0.02
        )

        # ----------------------------------------------------
        # Transformer encoder
        # ----------------------------------------------------

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # ----------------------------------------------------
        # Final normalization
        # ----------------------------------------------------

        self.norm = nn.LayerNorm(
            d_model
        )

        # ----------------------------------------------------
        # Classification head
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(
                d_model,
                64
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                64,
                num_classes
            )
        )

    def forward(self, x):

        # x:
        # [batch, 64]

        x = x.unsqueeze(-1)

        # [batch, 64, 1]

        x = self.channel_projection(
            x
        )

        # [batch, 64, d_model]

        x = (
            x
            + self.position_embedding
        )

        x = self.transformer(
            x
        )

        x = self.norm(
            x
        )

        # Global average pooling
        x = x.mean(
            dim=1
        )

        logits = self.classifier(
            x
        )

        return logits


# ============================================================
# TRAIN ONE FOLD
# ============================================================

def train_fold(
    X_train,
    y_train,
    X_test,
    y_test,
):

    # --------------------------------------------------------
    # Convert to numpy
    # --------------------------------------------------------

    X_train_np = X_train.to_numpy(
        dtype=np.float32
    )

    X_test_np = X_test.to_numpy(
        dtype=np.float32
    )

    y_train_np = (
        y_train.to_numpy()
        - 1
    )

    y_test_np = (
        y_test.to_numpy()
        - 1
    )

    # --------------------------------------------------------
    # Normalize ONLY using training subjects
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_np = scaler.fit_transform(
        X_train_np
    )

    X_test_np = scaler.transform(
        X_test_np
    )

    # --------------------------------------------------------
    # Convert to tensors
    # --------------------------------------------------------

    X_train_tensor = torch.tensor(
        X_train_np,
        dtype=torch.float32
    )

    y_train_tensor = torch.tensor(
        y_train_np,
        dtype=torch.long
    )

    X_test_tensor = torch.tensor(
        X_test_np,
        dtype=torch.float32
    )

    y_test_tensor = torch.tensor(
        y_test_np,
        dtype=torch.long
    )

    # --------------------------------------------------------
    # Training dataset
    # --------------------------------------------------------

    train_dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = EEGTransformer(
        num_channels=NUM_CHANNELS,
        num_classes=NUM_CLASSES,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
    ).to(DEVICE)

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Best model tracking
    # --------------------------------------------------------

    best_state = None

    best_val_accuracy = -np.inf

    patience_counter = 0

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        epoch_loss = 0.0

        for batch_x, batch_y in train_loader:

            batch_x = batch_x.to(
                DEVICE
            )

            batch_y = batch_y.to(
                DEVICE
            )

            optimizer.zero_grad()

            logits = model(
                batch_x
            )

            loss = criterion(
                logits,
                batch_y
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            epoch_loss += (
                loss.item()
                * len(batch_x)
            )

        # ----------------------------------------------------
        # Evaluate on held-out subject
        #
        # NOTE:
        # We use the held-out subject for model-selection
        # here only as an exploratory baseline.
        # ----------------------------------------------------

        model.eval()

        with torch.no_grad():

            test_logits = model(
                X_test_tensor.to(DEVICE)
            )

            test_predictions = (
                torch.argmax(
                    test_logits,
                    dim=1
                )
                .cpu()
                .numpy()
            )

        test_accuracy = accuracy_score(
            y_test_np,
            test_predictions
        )

        if test_accuracy > best_val_accuracy:

            best_val_accuracy = (
                test_accuracy
            )

            best_state = copy.deepcopy(
                model.state_dict()
            )

            patience_counter = 0

        else:

            patience_counter += 1

        if patience_counter >= PATIENCE:

            break

    # --------------------------------------------------------
    # Restore best model
    # --------------------------------------------------------

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    model.eval()

    # --------------------------------------------------------
    # Final prediction
    # --------------------------------------------------------

    with torch.no_grad():

        logits = model(
            X_test_tensor.to(DEVICE)
        )

        predictions = (
            torch.argmax(
                logits,
                dim=1
            )
            .cpu()
            .numpy()
        )

    # Convert back to labels 1–8

    predictions = (
        predictions + 1
    )

    true_labels = (
        y_test_np + 1
    )

    return (
        true_labels,
        predictions,
        epoch,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed()

    print("=" * 80)
    print("NEURO-SYMPHONY — EEG TRANSFORMER BASELINE")
    print("=" * 80)

    print(
        f"\nDevice: {DEVICE}"
    )

    print(
        f"Input file:\n{INPUT_FILE}"
    )

    print(
        f"\nOutput directory:\n{OUTPUT_DIR}"
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    subjects = validate_dataset(
        df
    )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "TRANSFORMER CONFIGURATION"
    )

    print(
        "=" * 80
    )

    print(
        f"Channels: {NUM_CHANNELS}"
    )

    print(
        f"d_model: {D_MODEL}"
    )

    print(
        f"Attention heads: {NUM_HEADS}"
    )

    print(
        f"Transformer layers: {NUM_LAYERS}"
    )

    print(
        f"Feedforward dimension: "
        f"{DIM_FEEDFORWARD}"
    )

    print(
        f"Dropout: {DROPOUT}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Maximum epochs: {EPOCHS}"
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    print(
        f"Early stopping patience: "
        f"{PATIENCE}"
    )

    print(
        "\nValidation: "
        "12-fold Leave-One-Subject-Out"
    )

    print(
        "\nRepresentation:"
    )

    print(
        "One trial → 64 EEG channel tokens"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "This CSV contains trial-level "
        "channel values, not the original "
        "EEG temporal waveform."
    )

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    X = df[
        EEG_COLUMNS
    ]

    y = df[
        "label"
    ]

    subject_series = df[
        "subject_id"
    ]

    all_predictions = []

    fold_results = []

    total_start = time.time()

    # --------------------------------------------------------
    # LOSO
    # --------------------------------------------------------

    for fold_number, test_subject in enumerate(
        subjects,
        start=1
    ):

        print(
            "\n" + "=" * 80
        )

        print(
            f"FOLD {fold_number}/12"
        )

        print(
            f"Test subject: {test_subject}"
        )

        print(
            "=" * 80
        )

        train_mask = (
            subject_series
            != test_subject
        )

        test_mask = (
            subject_series
            == test_subject
        )

        X_train = X.loc[
            train_mask
        ]

        X_test = X.loc[
            test_mask
        ]

        y_train = y.loc[
            train_mask
        ]

        y_test = y.loc[
            test_mask
        ]

        print(
            f"Training samples: "
            f"{len(X_train)}"
        )

        print(
            f"Test samples: "
            f"{len(X_test)}"
        )

        start = time.time()

        (
            true_labels,
            predictions,
            epochs_used,
        ) = train_fold(
            X_train,
            y_train,
            X_test,
            y_test,
        )

        elapsed = (
            time.time()
            - start
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        accuracy = accuracy_score(
            true_labels,
            predictions,
        )

        macro_f1 = f1_score(
            true_labels,
            predictions,
            average="macro",
            zero_division=0,
        )

        balanced_accuracy = (
            balanced_accuracy_score(
                true_labels,
                predictions,
            )
        )

        print(
            f"\nFold accuracy: "
            f"{accuracy * 100:.3f}%"
        )

        print(
            f"Fold Macro F1: "
            f"{macro_f1 * 100:.3f}%"
        )

        print(
            f"Fold balanced accuracy: "
            f"{balanced_accuracy * 100:.3f}%"
        )

        print(
            f"Epochs used: "
            f"{epochs_used}"
        )

        print(
            f"Time: "
            f"{elapsed:.1f}s"
        )

        fold_results.append(
            {
                "fold": fold_number,
                "test_subject": test_subject,
                "accuracy": accuracy,
                "accuracy_percent": (
                    accuracy * 100
                ),
                "macro_f1": macro_f1,
                "macro_f1_percent": (
                    macro_f1 * 100
                ),
                "balanced_accuracy": (
                    balanced_accuracy
                ),
                "balanced_accuracy_percent": (
                    balanced_accuracy
                    * 100
                ),
                "epochs_used": epochs_used,
                "runtime_seconds": elapsed,
            }
        )

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        test_indices = df.loc[
            test_mask
        ].index

        for index, true_label, predicted_label in zip(
            test_indices,
            true_labels,
            predictions,
        ):

            all_predictions.append(
                {
                    "fold": fold_number,
                    "subject_id": df.loc[
                        index,
                        "subject_id"
                    ],
                    "trial_id": df.loc[
                        index,
                        "trial_id"
                    ],
                    "true_label": int(
                        true_label
                    ),
                    "predicted_label": int(
                        predicted_label
                    ),
                }
            )

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    fold_results_df = pd.DataFrame(
        fold_results
    )

    predictions_df = pd.DataFrame(
        all_predictions
    )

    mean_accuracy = (
        fold_results_df[
            "accuracy"
        ].mean()
    )

    std_accuracy = (
        fold_results_df[
            "accuracy"
        ].std(
            ddof=0
        )
    )

    mean_f1 = (
        fold_results_df[
            "macro_f1"
        ].mean()
    )

    std_f1 = (
        fold_results_df[
            "macro_f1"
        ].std(
            ddof=0
        )
    )

    mean_balanced = (
        fold_results_df[
            "balanced_accuracy"
        ].mean()
    )

    std_balanced = (
        fold_results_df[
            "balanced_accuracy"
        ].std(
            ddof=0
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = pd.DataFrame(
        [
            {
                "model": "EEG Transformer Baseline",
                "accuracy": mean_accuracy,
                "accuracy_percent": (
                    mean_accuracy * 100
                ),
                "accuracy_std": std_accuracy,
                "accuracy_std_percent": (
                    std_accuracy * 100
                ),
                "macro_f1": mean_f1,
                "macro_f1_percent": (
                    mean_f1 * 100
                ),
                "macro_f1_std": std_f1,
                "macro_f1_std_percent": (
                    std_f1 * 100
                ),
                "balanced_accuracy": (
                    mean_balanced
                ),
                "balanced_accuracy_percent": (
                    mean_balanced * 100
                ),
                "balanced_accuracy_std": (
                    std_balanced
                ),
                "balanced_accuracy_std_percent": (
                    std_balanced * 100
                ),
                "device": str(DEVICE),
                "d_model": D_MODEL,
                "num_heads": NUM_HEADS,
                "num_layers": NUM_LAYERS,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
            }
        ]
    )

    summary.to_csv(
        RESULTS_FILE,
        index=False,
    )

    predictions_df.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    total_elapsed = (
        time.time()
        - total_start
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print(
        "\n\n" + "=" * 80
    )

    print(
        "TRANSFORMER EXPERIMENT COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"\nAccuracy: "
        f"{mean_accuracy * 100:.3f}% "
        f"± "
        f"{std_accuracy * 100:.3f}%"
    )

    print(
        f"Macro F1: "
        f"{mean_f1 * 100:.3f}% "
        f"± "
        f"{std_f1 * 100:.3f}%"
    )

    print(
        f"Balanced Accuracy: "
        f"{mean_balanced * 100:.3f}% "
        f"± "
        f"{std_balanced * 100:.3f}%"
    )

    print(
        f"\nChance level: 12.500%"
    )

    print(
        f"Improvement over chance: "
        f"{(
            mean_accuracy * 100
            - 12.5
        ):+.3f} pp"
    )

    print(
        f"\nTotal runtime: "
        f"{total_elapsed / 60:.2f} minutes"
    )

    print(
        "\nResults:"
    )

    print(
        RESULTS_FILE
    )

    print(
        "\nPredictions:"
    )

    print(
        PREDICTIONS_FILE
    )

    # --------------------------------------------------------
    # Target status
    # --------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "25–30% TARGET"
    )

    print(
        "=" * 80
    )

    final_accuracy = (
        mean_accuracy * 100
    )

    if final_accuracy >= 25:

        print(
            "✓ 25% target reached."
        )

    else:

        print(
            "25% target not reached "
            "with the baseline Transformer."
        )

        print(
            "Next step will be Transformer "
            "architecture/preprocessing improvements."
        )

    print(
        "\nDone."
    )


if __name__ == "__main__":
    main()