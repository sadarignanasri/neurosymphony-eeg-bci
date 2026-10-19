
"""
Experiment 6: CNN + Transformer for EEG Classification
======================================================

Neuro-Symphony
DS004514 EEG semantic decoding

IMPORTANT:
-----------
The current approved EEG CSV contains ONE spatial value per EEG channel
for each trial. It does NOT contain temporal EEG samples.

Therefore, this model operates on the 64-channel spatial representation.

Architecture:
    64 EEG channels
        ↓
    CNN feature extraction
        ↓
    Channel/token representation
        ↓
    Transformer Encoder
        ↓
    Attention pooling
        ↓
    Classification head
        ↓
    8 classes

Evaluation:
-----------
Strict subject-independent 10 / 1 / 1 protocol:

    10 subjects -> training
    1 subject   -> validation
    1 subject   -> testing

The test subject rotates through all 12 subjects.

Important:
-----------
- Scaling is fitted ONLY on training subjects.
- Validation is used for early stopping.
- Test data is never used for model selection.
- Labels in the CSV are 1-8.
- PyTorch internally uses 0-7.
- CPU mode is intentionally used for stability.
- No DataLoader.
- No MKLDNN.
- No MPS.
"""

from pathlib import Path
import time
import random
import warnings

import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

DATA_PATH = Path(
    "data/processed/ds004514_eeg_approved.csv"
)

OUTPUT_DIR = Path(
    "results/eeg/cnn_transformer"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

N_CLASSES = 8
N_CHANNELS = 64

MAX_EPOCHS = 50
BATCH_SIZE = 64

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

PATIENCE = 8
MIN_DELTA = 1e-4

DEVICE = torch.device("cpu")

# CPU stability settings
torch.set_num_threads(1)

try:
    torch.backends.mkldnn.enabled = False
except Exception:
    pass

warnings.filterwarnings(
    "ignore",
    message=".*nested tensor.*"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed()


# ============================================================
# PRINT ENVIRONMENT
# ============================================================

print("=" * 80)
print("EXPERIMENT 6 — CNN + TRANSFORMER")
print("=" * 80)

print(f"Device          : {DEVICE}")
print(f"PyTorch version : {torch.__version__}")
print(f"CPU threads     : {torch.get_num_threads()}")
print(f"MKLDNN enabled  : {torch.backends.mkldnn.enabled}")


# ============================================================
# LOAD DATASET
# ============================================================

print("\n" + "=" * 80)
print("1. LOADING DATASET")
print("=" * 80)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")

required_columns = (
    ["subject_id", "trial_id"]
    + [f"A{i}" for i in range(1, 33)]
    + [f"B{i}" for i in range(1, 33)]
    + ["label_name", "label"]
)

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# CHANNEL LIST
# ============================================================

CHANNELS = (
    [f"A{i}" for i in range(1, 33)]
    + [f"B{i}" for i in range(1, 33)]
)

print(f"EEG channels: {len(CHANNELS)}")
print(f"First channels: {CHANNELS[:5]}")
print(f"Last channels : {CHANNELS[-5:]}")


# ============================================================
# DATASET VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("2. DATASET VALIDATION")
print("=" * 80)

subjects = sorted(
    df["subject_id"].astype(str).unique()
)

labels = sorted(
    df["label"].astype(int).unique()
)

print(f"Subjects: {len(subjects)}")
print(f"Classes : {len(labels)}")
print(f"Labels  : {labels}")

if len(subjects) != 12:
    raise ValueError(
        f"Expected 12 subjects, found {len(subjects)}"
    )

if labels != list(range(1, 9)):
    raise ValueError(
        f"Expected labels 1-8, found {labels}"
    )


# Subject counts
subject_counts = (
    df.groupby("subject_id")
    .size()
    .sort_index()
)

print("\nTrials per subject:")
print(subject_counts)

if not (subject_counts == 720).all():
    raise ValueError(
        "Every subject must contain exactly 720 trials."
    )


# Class counts
class_counts = (
    df.groupby("label")
    .size()
    .sort_index()
)

print("\nTrials per class:")
print(class_counts)

if not (class_counts == 1080).all():
    raise ValueError(
        "Every class must contain exactly 1080 trials."
    )


# Subject-class counts
subject_class_counts = (
    df.groupby(["subject_id", "label"])
    .size()
)

print("\nSubject/class minimum count:")
print(subject_class_counts.min())

if not (subject_class_counts == 90).all():
    raise ValueError(
        "Every subject/class combination must contain exactly 90 trials."
    )


# Duplicate check
duplicate_count = df.duplicated(
    subset=["subject_id", "trial_id"]
).sum()

print(f"\nDuplicate subject/trial rows: {duplicate_count}")

if duplicate_count != 0:
    raise ValueError(
        "Duplicate subject/trial rows detected."
    )


# Missing values
missing_count = df[CHANNELS].isna().sum().sum()

print(f"Missing EEG values: {missing_count}")

if missing_count != 0:
    raise ValueError(
        "Missing EEG values detected."
    )


# Infinite values
infinite_count = np.isinf(
    df[CHANNELS].to_numpy(dtype=np.float64)
).sum()

print(f"Infinite EEG values: {infinite_count}")

if infinite_count != 0:
    raise ValueError(
        "Infinite EEG values detected."
    )


# ============================================================
# FEATURE MATRIX
# ============================================================

X_all = df[CHANNELS].to_numpy(
    dtype=np.float32
)

y_all = df["label"].to_numpy(
    dtype=np.int64
)

subject_array = (
    df["subject_id"]
    .astype(str)
    .to_numpy()
)

print("\nFeature matrix:")
print(f"X shape: {X_all.shape}")
print(f"y shape: {y_all.shape}")


# ============================================================
# MODEL
# ============================================================

class CNNTransformer(nn.Module):
    """
    CNN + Transformer model for the current
    64-channel trial-level EEG representation.

    Input:
        [batch, 64]

    CNN:
        Converts neighboring channel values into
        local spatial representations.

    Transformer:
        Treats CNN feature positions as tokens
        and models relationships between them.

    Output:
        [batch, 8]
    """

    def __init__(
        self,
        n_classes=N_CLASSES
    ):
        super().__init__()

        # ----------------------------------------------------
        # CNN FEATURE EXTRACTION
        # ----------------------------------------------------

        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                padding=2
            ),
            nn.ReLU(),

            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv1d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.AvgPool1d(
                kernel_size=2
            )
        )

        # After pooling:
        #
        # 64 channel positions
        #       ↓
        # 32 positions
        #
        # Each position has 32 features.

        d_model = 32

        # ----------------------------------------------------
        # POSITIONAL EMBEDDING
        # ----------------------------------------------------

        self.positional_embedding = nn.Parameter(
            torch.zeros(
                1,
                32,
                d_model
            )
        )

        nn.init.normal_(
            self.positional_embedding,
            mean=0.0,
            std=0.02
        )

        # ----------------------------------------------------
        # TRANSFORMER
        # ----------------------------------------------------

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=64,
            dropout=0.2,
            activation="gelu",
            batch_first=True,
            norm_first=False
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        # ----------------------------------------------------
        # ATTENTION POOLING
        # ----------------------------------------------------

        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )

        # ----------------------------------------------------
        # CLASSIFICATION HEAD
        # ----------------------------------------------------

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):

        # x:
        # [batch, 64]

        x = x.unsqueeze(1)

        # [batch, 1, 64]

        x = self.cnn(x)

        # [batch, 32, 32]

        x = x.transpose(1, 2)

        # [batch, 32, 32]
        #
        # 32 tokens
        # 32 features/token

        x = x + self.positional_embedding

        x = self.transformer(x)

        # [batch, 32, 32]

        attention_scores = self.attention_pool(x)

        # [batch, 32, 1]

        attention_weights = torch.softmax(
            attention_scores,
            dim=1
        )

        pooled = (
            x * attention_weights
        ).sum(dim=1)

        # [batch, 32]

        logits = self.classifier(
            pooled
        )

        # [batch, 8]

        return logits


# ============================================================
# MODEL SANITY CHECK
# ============================================================

print("\n" + "=" * 80)
print("3. MODEL SANITY CHECK")
print("=" * 80)

model_test = CNNTransformer(
    n_classes=N_CLASSES
).to(DEVICE)

dummy_input = torch.randn(
    2,
    N_CHANNELS,
    dtype=torch.float32,
    device=DEVICE
)

with torch.no_grad():
    dummy_output = model_test(
        dummy_input
    )

print(
    f"Input shape : {tuple(dummy_input.shape)}"
)

print(
    f"Output shape: {tuple(dummy_output.shape)}"
)

if tuple(dummy_output.shape) != (2, N_CLASSES):
    raise RuntimeError(
        "CNN + Transformer output shape is incorrect."
    )

print("Model sanity check PASSED.")

del model_test
del dummy_input
del dummy_output


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    optimizer,
    criterion,
    X_train,
    y_train
):

    model.train()

    indices = np.random.permutation(
        len(X_train)
    )

    total_loss = 0.0
    correct = 0
    total = 0

    for start in range(
        0,
        len(indices),
        BATCH_SIZE
    ):

        batch_idx = indices[
            start:start + BATCH_SIZE
        ]

        xb = torch.tensor(
            X_train[batch_idx],
            dtype=torch.float32,
            device=DEVICE
        )

        # CSV labels are 1-8.
        # PyTorch CrossEntropyLoss expects 0-7.
        yb = torch.tensor(
            y_train[batch_idx] - 1,
            dtype=torch.long,
            device=DEVICE
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(xb)

        loss = criterion(
            logits,
            yb
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += (
            loss.item() * len(batch_idx)
        )

        predictions = torch.argmax(
            logits,
            dim=1
        )

        correct += (
            predictions == yb
        ).sum().item()

        total += len(batch_idx)

    mean_loss = (
        total_loss / total
    )

    accuracy = (
        100.0 * correct / total
    )

    return mean_loss, accuracy


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    criterion,
    X,
    y
):

    model.eval()

    total_loss = 0.0
    total = 0

    all_predictions = []
    all_targets = []

    with torch.no_grad():

        for start in range(
            0,
            len(X),
            BATCH_SIZE
        ):

            end = min(
                start + BATCH_SIZE,
                len(X)
            )

            xb = torch.tensor(
                X[start:end],
                dtype=torch.float32,
                device=DEVICE
            )

            yb = torch.tensor(
                y[start:end] - 1,
                dtype=torch.long,
                device=DEVICE
            )

            logits = model(xb)

            loss = criterion(
                logits,
                yb
            )

            total_loss += (
                loss.item() * (end - start)
            )

            total += (
                end - start
            )

            predictions = torch.argmax(
                logits,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_targets.extend(
                yb.cpu().numpy()
            )

    mean_loss = (
        total_loss / total
    )

    predictions = np.asarray(
        all_predictions
    )

    targets = np.asarray(
        all_targets
    )

    accuracy = (
        accuracy_score(
            targets,
            predictions
        ) * 100.0
    )

    return (
        mean_loss,
        accuracy,
        targets,
        predictions
    )


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_model(
    X_train,
    y_train,
    X_val,
    y_val
):

    model = CNNTransformer(
        n_classes=N_CLASSES
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    best_val_accuracy = -np.inf
    best_epoch = 0

    best_state = None

    patience_counter = 0

    history = []

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                optimizer,
                criterion,
                X_train,
                y_train
            )
        )

        val_loss, val_accuracy, _, _ = (
            evaluate(
                model,
                criterion,
                X_val,
                y_val
            )
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        })

        print(
            f"    Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"Train Loss {train_loss:.4f} | "
            f"Train Acc {train_accuracy:.3f}% | "
            f"Val Loss {val_loss:.4f} | "
            f"Val Acc {val_accuracy:.3f}%"
        )

        if (
            val_accuracy
            > best_val_accuracy + MIN_DELTA
        ):

            best_val_accuracy = (
                val_accuracy
            )

            best_epoch = epoch

            best_state = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }

            patience_counter = 0

        else:

            patience_counter += 1

        if patience_counter >= PATIENCE:

            print(
                f"    Early stopping at epoch {epoch}"
            )

            break

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    history_df = pd.DataFrame(
        history
    )

    return (
        model,
        history_df,
        best_epoch,
        best_val_accuracy
    )


# ============================================================
# LOSO EXPERIMENT
# ============================================================

print("\n" + "=" * 80)
print("4. SUBJECT-INDEPENDENT CNN + TRANSFORMER")
print("=" * 80)

print(
    "Protocol: 10 training subjects + "
    "1 validation subject + 1 test subject"
)

print(
    f"Total folds: {len(subjects)}"
)

print(
    f"Chance accuracy: {100.0 / N_CLASSES:.3f}%"
)


fold_results = []
prediction_records = []
history_records = []
split_records = []

all_test_targets = []
all_test_predictions = []

overall_start = time.time()


# ------------------------------------------------------------
# Rotate test subject through all 12 subjects
# ------------------------------------------------------------

for fold_idx, test_subject in enumerate(
    subjects,
    start=1
):

    fold_start = time.time()

    remaining_subjects = [
        s for s in subjects
        if s != test_subject
    ]

    # Deterministic validation subject.
    #
    # We use the first remaining subject for validation.
    # The remaining 10 subjects are used for training.
    validation_subject = remaining_subjects[
        -1
    ]

    train_subjects = [
        s for s in remaining_subjects
        if s != validation_subject
    ]

    print("\n" + "-" * 80)
    print(
        f"FOLD {fold_idx}/{len(subjects)}"
    )
    print("-" * 80)

    print(
        f"Train subjects ({len(train_subjects)}): "
        f"{train_subjects}"
    )

    print(
        f"Validation subject: "
        f"{validation_subject}"
    )

    print(
        f"Test subject      : "
        f"{test_subject}"
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

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

    X_train = X_all[
        train_mask
    ].copy()

    y_train = y_all[
        train_mask
    ].copy()

    X_val = X_all[
        val_mask
    ].copy()

    y_val = y_all[
        val_mask
    ].copy()

    X_test = X_all[
        test_mask
    ].copy()

    y_test = y_all[
        test_mask
    ].copy()

    print(
        f"\nSplit sizes:"
    )

    print(
        f"  Train: {len(X_train)}"
    )

    print(
        f"  Val  : {len(X_val)}"
    )

    print(
        f"  Test : {len(X_test)}"
    )

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    train_counts = np.bincount(
        y_train,
        minlength=N_CLASSES + 1
    )[1:]

    val_counts = np.bincount(
        y_val,
        minlength=N_CLASSES + 1
    )[1:]

    test_counts = np.bincount(
        y_test,
        minlength=N_CLASSES + 1
    )[1:]

    print(
        f"  Train class counts: "
        f"{train_counts.tolist()}"
    )

    print(
        f"  Val class counts  : "
        f"{val_counts.tolist()}"
    )

    print(
        f"  Test class counts : "
        f"{test_counts.tolist()}"
    )

    # --------------------------------------------------------
    # Save split information
    # --------------------------------------------------------

    split_records.append({
        "fold": fold_idx,
        "train_subjects": ",".join(
            train_subjects
        ),
        "validation_subject": validation_subject,
        "test_subject": test_subject,
        "train_samples": len(X_train),
        "validation_samples": len(X_val),
        "test_samples": len(X_test),
    })

    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------
    #
    # VERY IMPORTANT:
    # RobustScaler is fitted only on training subjects.
    #

    scaler = RobustScaler()

    X_train = scaler.fit_transform(
        X_train
    ).astype(np.float32)

    X_val = scaler.transform(
        X_val
    ).astype(np.float32)

    X_test = scaler.transform(
        X_test
    ).astype(np.float32)

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    set_seed(
        SEED + fold_idx
    )

    model, history_df, best_epoch, best_val = (
        train_model(
            X_train,
            y_train,
            X_val,
            y_val
        )
    )

    # --------------------------------------------------------
    # Training evaluation
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    train_loss, train_accuracy, _, _ = (
        evaluate(
            model,
            criterion,
            X_train,
            y_train
        )
    )

    # --------------------------------------------------------
    # Validation evaluation
    # --------------------------------------------------------

    val_loss, val_accuracy, _, _ = (
        evaluate(
            model,
            criterion,
            X_val,
            y_val
        )
    )

    # --------------------------------------------------------
    # Test evaluation
    # --------------------------------------------------------

    test_loss, test_accuracy, test_targets, test_predictions = (
        evaluate(
            model,
            criterion,
            X_test,
            y_test
        )
    )

    test_macro_f1 = (
        f1_score(
            test_targets,
            test_predictions,
            average="macro",
            zero_division=0
        ) * 100.0
    )

    test_balanced_accuracy = (
        balanced_accuracy_score(
            test_targets,
            test_predictions
        ) * 100.0
    )

    # Convert PyTorch labels 0-7 back to CSV labels 1-8
    test_targets_csv = (
        test_targets + 1
    )

    test_predictions_csv = (
        test_predictions + 1
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        test_targets,
        test_predictions,
        labels=list(range(N_CLASSES))
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            f"true_{i}"
            for i in range(1, N_CLASSES + 1)
        ],
        columns=[
            f"pred_{i}"
            for i in range(1, N_CLASSES + 1)
        ]
    )

    cm_path = (
        OUTPUT_DIR
        / f"fold_{fold_idx:02d}_confusion_matrix.csv"
    )

    cm_df.to_csv(
        cm_path
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        test_targets,
        test_predictions,
        labels=list(range(N_CLASSES)),
        target_names=[
            str(i)
            for i in range(1, N_CLASSES + 1)
        ],
        zero_division=0
    )

    report_path = (
        OUTPUT_DIR
        / f"fold_{fold_idx:02d}_classification_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            f"Fold {fold_idx}\n"
        )

        f.write(
            f"Train subjects: "
            f"{train_subjects}\n"
        )

        f.write(
            f"Validation subject: "
            f"{validation_subject}\n"
        )

        f.write(
            f"Test subject: "
            f"{test_subject}\n\n"
        )

        f.write(report)

    # --------------------------------------------------------
    # Training history
    # --------------------------------------------------------

    history_df = history_df.copy()

    history_df.insert(
        0,
        "fold",
        fold_idx
    )

    history_df.insert(
        1,
        "test_subject",
        test_subject
    )

    history_records.append(
        history_df
    )

    # --------------------------------------------------------
    # Prediction records
    # --------------------------------------------------------

    for i in range(
        len(test_targets_csv)
    ):

        prediction_records.append({
            "fold": fold_idx,
            "test_subject": test_subject,
            "true_label": int(
                test_targets_csv[i]
            ),
            "predicted_label": int(
                test_predictions_csv[i]
            ),
            "correct": int(
                test_targets_csv[i]
                == test_predictions_csv[i]
            )
        })

    all_test_targets.extend(
        test_targets
    )

    all_test_predictions.extend(
        test_predictions
    )

    # --------------------------------------------------------
    # Fold results
    # --------------------------------------------------------

    fold_time = (
        time.time()
        - fold_start
    )

    fold_results.append({
        "fold": fold_idx,
        "test_subject": test_subject,
        "validation_subject": validation_subject,
        "train_subject_count": len(
            train_subjects
        ),
        "train_samples": len(X_train),
        "validation_samples": len(X_val),
        "test_samples": len(X_test),
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_val,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "validation_loss": val_loss,
        "validation_accuracy": val_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "macro_f1": test_macro_f1,
        "balanced_accuracy": test_balanced_accuracy,
        "runtime_minutes": fold_time / 60.0
    })

    print("\nFold result:")
    print(
        f"  Train accuracy     : "
        f"{train_accuracy:.3f}%"
    )

    print(
        f"  Validation accuracy: "
        f"{val_accuracy:.3f}%"
    )

    print(
        f"  Test accuracy      : "
        f"{test_accuracy:.3f}%"
    )

    print(
        f"  Macro F1           : "
        f"{test_macro_f1:.3f}%"
    )

    print(
        f"  Balanced accuracy  : "
        f"{test_balanced_accuracy:.3f}%"
    )

    print(
        f"  Best epoch         : "
        f"{best_epoch}"
    )

    print(
        f"  Fold runtime       : "
        f"{fold_time / 60.0:.2f} min"
    )


# ============================================================
# FINAL RESULTS
# ============================================================

overall_time = (
    time.time()
    - overall_start
)

results_df = pd.DataFrame(
    fold_results
)

predictions_df = pd.DataFrame(
    prediction_records
)

split_df = pd.DataFrame(
    split_records
)

history_all_df = pd.concat(
    history_records,
    ignore_index=True
)


mean_train_accuracy = (
    results_df[
        "train_accuracy"
    ].mean()
)

mean_val_accuracy = (
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
    ].std(ddof=1)
)

mean_macro_f1 = (
    results_df[
        "macro_f1"
    ].mean()
)

std_macro_f1 = (
    results_df[
        "macro_f1"
    ].std(ddof=1)
)

mean_balanced_accuracy = (
    results_df[
        "balanced_accuracy"
    ].mean()
)


# ============================================================
# OVERALL CONFUSION MATRIX
# ============================================================

overall_cm = confusion_matrix(
    all_test_targets,
    all_test_predictions,
    labels=list(range(N_CLASSES))
)

overall_cm_df = pd.DataFrame(
    overall_cm,
    index=[
        f"true_{i}"
        for i in range(1, N_CLASSES + 1)
    ],
    columns=[
        f"pred_{i}"
        for i in range(1, N_CLASSES + 1)
    ]
)


# ============================================================
# OVERALL CLASSIFICATION REPORT
# ============================================================

overall_report = classification_report(
    all_test_targets,
    all_test_predictions,
    labels=list(range(N_CLASSES)),
    target_names=[
        str(i)
        for i in range(1, N_CLASSES + 1)
    ],
    zero_division=0
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_path = (
    OUTPUT_DIR
    / "cnn_transformer_results.csv"
)

predictions_path = (
    OUTPUT_DIR
    / "cnn_transformer_predictions.csv"
)

split_path = (
    OUTPUT_DIR
    / "cnn_transformer_split_summary.csv"
)

history_path = (
    OUTPUT_DIR
    / "cnn_transformer_training_history.csv"
)

cm_path = (
    OUTPUT_DIR
    / "cnn_transformer_confusion_matrix.csv"
)

report_path = (
    OUTPUT_DIR
    / "cnn_transformer_classification_report.txt"
)

metadata_path = (
    OUTPUT_DIR
    / "cnn_transformer_experiment_metadata.csv"
)


results_df.to_csv(
    results_path,
    index=False
)

predictions_df.to_csv(
    predictions_path,
    index=False
)

split_df.to_csv(
    split_path,
    index=False
)

history_all_df.to_csv(
    history_path,
    index=False
)

overall_cm_df.to_csv(
    cm_path
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "CNN + Transformer — Overall Classification Report\n"
    )

    f.write(
        "=" * 60
        + "\n\n"
    )

    f.write(overall_report)


metadata = pd.DataFrame([{
    "experiment": "Experiment 6",
    "model": "CNN + Transformer",
    "dataset": str(DATA_PATH),
    "input_channels": N_CHANNELS,
    "classes": N_CLASSES,
    "protocol": "10 train + 1 validation + 1 test",
    "folds": len(subjects),
    "scaler": "RobustScaler",
    "max_epochs": MAX_EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "patience": PATIENCE,
    "device": str(DEVICE),
    "torch_version": torch.__version__,
    "mean_train_accuracy": mean_train_accuracy,
    "mean_validation_accuracy": mean_val_accuracy,
    "mean_test_accuracy": mean_test_accuracy,
    "std_test_accuracy": std_test_accuracy,
    "mean_macro_f1": mean_macro_f1,
    "std_macro_f1": std_macro_f1,
    "mean_balanced_accuracy": mean_balanced_accuracy,
    "chance_accuracy": 100.0 / N_CLASSES,
    "accuracy_above_chance_pp": (
        mean_test_accuracy
        - (100.0 / N_CLASSES)
    ),
    "runtime_minutes": overall_time / 60.0
}])


metadata.to_csv(
    metadata_path,
    index=False
)


# ============================================================
# FINAL PRINT
# ============================================================

print("\n" + "=" * 80)
print("FINAL RESULTS — EXPERIMENT 6")
print("=" * 80)

print(
    f"Mean train accuracy       : "
    f"{mean_train_accuracy:.3f}%"
)

print(
    f"Mean validation accuracy  : "
    f"{mean_val_accuracy:.3f}%"
)

print(
    f"Mean test accuracy        : "
    f"{mean_test_accuracy:.3f}% "
    f"± {std_test_accuracy:.3f}%"
)

print(
    f"Mean Macro F1             : "
    f"{mean_macro_f1:.3f}% "
    f"± {std_macro_f1:.3f}%"
)

print(
    f"Mean balanced accuracy    : "
    f"{mean_balanced_accuracy:.3f}%"
)

print(
    f"Chance accuracy           : "
    f"{100.0 / N_CLASSES:.3f}%"
)

print(
    f"Accuracy above chance     : "
    f"{mean_test_accuracy - (100.0 / N_CLASSES):.3f} pp"
)

print(
    f"Total runtime             : "
    f"{overall_time / 60.0:.2f} min"
)


print("\nSaved outputs:")

print(
    f"  {results_path}"
)

print(
    f"  {predictions_path}"
)

print(
    f"  {split_path}"
)

print(
    f"  {history_path}"
)

print(
    f"  {cm_path}"
)

print(
    f"  {report_path}"
)

print(
    f"  {metadata_path}"
)

print("\n" + "=" * 80)
print("EXPERIMENT 6 COMPLETED")
print("=" * 80)