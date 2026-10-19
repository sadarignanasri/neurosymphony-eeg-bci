"""
Step 83: Compare alternative CNN architectures for NeuroSymphony EEG.

Models:
1. DeepConvNet
2. ShallowConvNet
3. MultiScaleCNN

Dataset:
data/processed/ds004514_eeg_approved.csv

Protocol:
10 training subjects + 1 validation subject + 1 test subject.
The test subject rotates through all 12 subjects.

Important:
This dataset contains one value per EEG channel per trial, not raw
time-series epochs. These experiments operate on the 64 channel values.
"""

from pathlib import Path
import random
import time
import copy
import json

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

SEED = 42
N_CLASSES = 8
EPOCHS = 60
BATCH_SIZE = 128
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.001
PATIENCE = 10

# Step 83 uses BDF-derived bandpower arrays, not the approved CSV.
OUTPUT_DIR = Path("results/eeg/step83_cnn_architecture_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 70)
print("STEP 83: ALTERNATIVE CNN ARCHITECTURE COMPARISON")
print("=" * 70)
print(f"Device: {DEVICE}")
print("Dataset: original BDF-derived bandpower features")
print(f"Output: {OUTPUT_DIR.resolve()}")

# ---------------------------------------------------------
# LOAD AND VALIDATE BDF-DERIVED BANDPOWER DATA
# ---------------------------------------------------------

from label_utils import load_subject_labels, EXPECTED_LABELS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_ROOT = PROJECT_ROOT / "data" / "features"
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]

expected_channels = [
    *(f"A{i}" for i in range(1, 33)),
    *(f"B{i}" for i in range(1, 33)),
]
channel_columns = [
    f"{band}_{channel}"
    for band in BANDS
    for channel in expected_channels
]

OUTPUT_DIR = PROJECT_ROOT / "results" / "eeg" / "step83_cnn_architecture_comparison_bdf_bandpower"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

parts = []

for subject_number in range(1, 13):
    subject = f"sub-{subject_number:02d}"
    feature_file = (
        FEATURE_ROOT / f"temporal_spectral_{subject}"
        / f"{subject}_band_power.npy"
    )

    if not feature_file.exists():
        raise FileNotFoundError(f"Missing BDF-derived features: {feature_file}")

    features = __import__("numpy").load(feature_file, mmap_mode="r")
    if features.shape != (720, 64, 5):
        raise ValueError(
            f"{subject}: expected bandpower shape (720, 64, 5), "
            f"found {features.shape}"
        )

    labels_zero_based = load_subject_labels(subject)
    if labels_zero_based.shape != (720,):
        raise ValueError(f"{subject}: expected 720 labels")

    # Convert [trials, channels, bands] to [trials, bands, channels],
    # then flatten only for the tabular split/scaling code below.
    flat_features = __import__("numpy").asarray(
        features.transpose(0, 2, 1), dtype="float32"
    ).reshape(720, 320)

    if not __import__("numpy").isfinite(flat_features).all():
        raise ValueError(f"{subject}: bandpower contains NaN or Inf")

    part = pd.DataFrame(flat_features, columns=channel_columns)
    part.insert(0, "label_name", [
        EXPECTED_LABELS[int(label)] for label in labels_zero_based
    ])
    part.insert(0, "label", labels_zero_based + 1)
    part.insert(0, "trial_id", range(1, 721))
    part.insert(0, "subject_id", subject)
    parts.append(part)

df = pd.concat(parts, ignore_index=True)
subjects = sorted(df["subject_id"].unique().tolist())

if len(subjects) != 12 or len(df) != 8640:
    raise ValueError(f"Unexpected combined dataset: {df.shape}")

print(f"BDF-derived bandpower dataset: {df.shape}")
print(f"Subjects: {len(subjects)}")
print(f"Features per trial: {len(channel_columns)} (5 bands x 64 channels)")
print("Labels loaded from Step 31 temporal metadata.")
print("Feature source validated; CSV input is no longer used.")

# ---------------------------------------------------------
# MODEL 1: DEEPCONVNET-STYLE 1D CNN
# ---------------------------------------------------------

class DeepConvNet(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(5, 25, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(25),
            nn.ELU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.35),

            nn.Conv1d(25, 50, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(50),
            nn.ELU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.35),

            nn.Conv1d(50, 100, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(100),
            nn.ELU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.35),

            nn.Conv1d(100, 150, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(150),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(150, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---------------------------------------------------------


# MODEL 2: SHALLOWCONVNET-STYLE CNN
# ---------------------------------------------------------

class ShallowConvNet(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()

        self.conv = nn.Conv1d(
            5, 40, kernel_size=13, padding=6, bias=False
        )
        self.norm = nn.BatchNorm1d(40)
        self.pool = nn.AvgPool1d(kernel_size=4, stride=2)
        self.dropout = nn.Dropout(0.4)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(40, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = x.square()
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        x = self.dropout(x)
        x = self.global_pool(x)
        return self.classifier(x.flatten(1))


# ---------------------------------------------------------
# MODEL 3: MULTI-SCALE CNN
# ---------------------------------------------------------

class MultiScaleCNN(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()

        self.branch3 = nn.Sequential(
            nn.Conv1d(5, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        self.branch7 = nn.Sequential(
            nn.Conv1d(5, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        self.branch15 = nn.Sequential(
            nn.Conv1d(5, 32, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        self.combine = nn.Sequential(
            nn.Conv1d(96, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),

            nn.Conv1d(96, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        a = self.branch3(x)
        b = self.branch7(x)
        c = self.branch15(x)
        x = torch.cat([a, b, c], dim=1)
        return self.classifier(self.combine(x))


MODEL_FACTORIES = {
    "DeepConvNet": DeepConvNet,
    "ShallowConvNet": ShallowConvNet,
    "MultiScaleCNN": MultiScaleCNN,
}

# ---------------------------------------------------------
# TRAINING AND EVALUATION HELPERS
# ---------------------------------------------------------

def make_input(X):
    # [trials, 320] -> [trials, 5 frequency bands, 64 EEG channels]
    array = __import__("numpy").asarray(X, dtype="float32")
    if array.ndim != 2 or array.shape[1] != 320:
        raise ValueError(f"Expected [trials, 320] bandpower features; got {array.shape}")
    return torch.tensor(array.reshape(-1, 5, 64), dtype=torch.float32)


def predict(model, X, batch_size=256):
    model.eval()
    predictions = []

    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = X[start:start + batch_size].to(DEVICE)
            logits = model(batch)
            predictions.extend(
                logits.argmax(dim=1).cpu().numpy().tolist()
            )

    return np.asarray(predictions, dtype=int)


def evaluate(model, X, y):
    predictions = predict(model, X)

    return {
        "accuracy": accuracy_score(y, predictions),
        "balanced_accuracy": balanced_accuracy_score(y, predictions),
        "macro_f1": f1_score(
            y, predictions, average="macro", zero_division=0
        ),
    }, predictions


def train_model(model, X_train, y_train, X_val, y_val):
    model = model.to(DEVICE)

    train_x = make_input(X_train)
    train_y = torch.tensor(
        y_train - 1, dtype=torch.long
    )

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_x = make_input(X_val).to(DEVICE)
    val_y = y_val - 1

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    wait = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_y)
            total_correct += (
                logits.argmax(dim=1) == batch_y
            ).sum().item()
            total_examples += len(batch_y)

        model.eval()
        with torch.no_grad():
            val_logits = model(val_x)
            val_loss = criterion(
                val_logits,
                torch.tensor(val_y, dtype=torch.long, device=DEVICE),
            ).item()
            val_predictions = val_logits.argmax(dim=1).cpu().numpy()

        val_accuracy = accuracy_score(val_y, val_predictions)
        val_f1 = f1_score(
            val_y, val_predictions,
            average="macro", zero_division=0
        )

        history.append({
            "epoch": epoch,
            "train_loss": total_loss / total_examples,
            "train_accuracy": total_correct / total_examples,
            "validation_loss": val_loss,
            "validation_accuracy": val_accuracy,
            "validation_macro_f1": val_f1,
        })

        # Model selection uses validation macro-F1 only.
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"    Epoch {epoch:02d} | "
                f"train_acc={total_correct / total_examples:.3f} | "
                f"val_acc={val_accuracy:.3f} | "
                f"val_macro_f1={val_f1:.3f}"
            )

        if wait >= PATIENCE:
            break

    model.load_state_dict(best_state)

    return model, pd.DataFrame(history), best_epoch, best_val_f1


# ---------------------------------------------------------
# ROTATING SUBJECT-WISE EVALUATION
# ---------------------------------------------------------

results = []
split_rows = []
all_predictions = []

for fold, test_subject in enumerate(subjects):
    remaining = [s for s in subjects if s != test_subject]

    # Rotate validation subject deterministically.
    validation_subject = remaining[fold % len(remaining)]

    train_subjects = [
        s for s in remaining if s != validation_subject
    ]

    train_df = df[df["subject_id"].isin(train_subjects)].copy()
    val_df = df[df["subject_id"] == validation_subject].copy()
    test_df = df[df["subject_id"] == test_subject].copy()

    X_train_raw = train_df[channel_columns].to_numpy(dtype=np.float64)
    X_val_raw = val_df[channel_columns].to_numpy(dtype=np.float64)
    X_test_raw = test_df[channel_columns].to_numpy(dtype=np.float64)

    y_train = train_df["label"].to_numpy(dtype=int)
    y_val = val_df["label"].to_numpy(dtype=int)
    y_test = test_df["label"].to_numpy(dtype=int)

    # Fit scaling on training subjects only.
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_val = scaler.transform(X_val_raw).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)

    if not (
        set(train_subjects).isdisjoint({validation_subject, test_subject})
        and validation_subject != test_subject
    ):
        raise RuntimeError("Subject split overlap detected.")

    split_rows.append({
        "fold": fold + 1,
        "train_subjects": ",".join(train_subjects),
        "validation_subject": validation_subject,
        "test_subject": test_subject,
        "train_samples": len(y_train),
        "validation_samples": len(y_val),
        "test_samples": len(y_test),
    })

    print("\n" + "=" * 70)
    print(
        f"FOLD {fold + 1}/12 | "
        f"Validation: {validation_subject} | Test: {test_subject}"
    )
    print("=" * 70)

    for model_name, factory in MODEL_FACTORIES.items():
        print(f"\nTraining {model_name}...")

        # Keep each fold reproducible.
        torch.manual_seed(SEED + fold)
        start = time.time()

        model = factory(num_classes=N_CLASSES)

        model, history, best_epoch, best_val_f1 = train_model(
            model, X_train, y_train, X_val, y_val
        )

        elapsed = time.time() - start

        test_metrics, test_predictions = evaluate(
            model, make_input(X_test), y_test - 1
        )

        # Predictions from the model are zero-based.
        test_predictions_original = test_predictions + 1

        subject_result = {
            "fold": fold + 1,
            "model": model_name,
            "validation_subject": validation_subject,
            "test_subject": test_subject,
            "train_subject_count": len(train_subjects),
            "train_samples": len(y_train),
            "validation_samples": len(y_val),
            "test_samples": len(y_test),
            "best_epoch": best_epoch,
            "best_validation_macro_f1": best_val_f1,
            "test_accuracy": test_metrics["accuracy"],
            "test_balanced_accuracy": test_metrics["balanced_accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "runtime_seconds": round(elapsed, 2),
        }
        results.append(subject_result)

        model_dir = OUTPUT_DIR / model_name / f"fold_{fold + 1:02d}"
        model_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(history).to_csv(
            model_dir / "training_history.csv", index=False
        )

        torch.save({
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "channel_columns": channel_columns,
            "class_labels": list(range(1, N_CLASSES + 1)),
            "test_subject": test_subject,
            "validation_subject": validation_subject,
            "best_epoch": best_epoch,
        }, model_dir / "best_model.pt")

        report = classification_report(
            y_test,
            test_predictions_original,
            labels=list(range(1, N_CLASSES + 1)),
            output_dict=True,
            zero_division=0,
        )
        pd.DataFrame(report).T.to_csv(
            model_dir / "classification_report.csv"
        )

        cm = confusion_matrix(
            y_test,
            test_predictions_original,
            labels=list(range(1, N_CLASSES + 1)),
        )
        pd.DataFrame(
            cm,
            index=[f"true_{i}" for i in range(1, 9)],
            columns=[f"pred_{i}" for i in range(1, 9)],
        ).to_csv(model_dir / "confusion_matrix.csv")

        prediction_df = test_df[
            ["subject_id", "trial_id", "label", "label_name"]
        ].copy()
        prediction_df["predicted_label"] = test_predictions_original
        prediction_df["correct"] = (
            prediction_df["label"] == prediction_df["predicted_label"]
        )
        prediction_df["model"] = model_name
        prediction_df["fold"] = fold + 1

        prediction_df.to_csv(
            model_dir / "test_predictions.csv", index=False
        )

        all_predictions.append(prediction_df)

        print(
            f"  {model_name}: "
            f"test_acc={test_metrics['accuracy']:.4f}, "
            f"balanced_acc={test_metrics['balanced_accuracy']:.4f}, "
            f"macro_f1={test_metrics['macro_f1']:.4f}, "
            f"best_epoch={best_epoch}, "
            f"runtime={elapsed:.1f}s"
        )

        del model

# ---------------------------------------------------------
# SAVE AGGREGATE RESULTS
# ---------------------------------------------------------

results_df = pd.DataFrame(results)
splits_df = pd.DataFrame(split_rows)

results_df.to_csv(
    OUTPUT_DIR / "step83_subject_results.csv", index=False
)
splits_df.to_csv(
    OUTPUT_DIR / "step83_split_summary.csv", index=False
)

if all_predictions:
    pd.concat(all_predictions, ignore_index=True).to_csv(
        OUTPUT_DIR / "step83_test_predictions.csv", index=False
    )

aggregate = (
    results_df.groupby("model")
    .agg(
        subjects=("test_subject", "nunique"),
        mean_test_accuracy=("test_accuracy", "mean"),
        std_test_accuracy=("test_accuracy", "std"),
        mean_balanced_accuracy=("test_balanced_accuracy", "mean"),
        mean_macro_f1=("test_macro_f1", "mean"),
        std_macro_f1=("test_macro_f1", "std"),
        mean_best_epoch=("best_epoch", "mean"),
        total_runtime_seconds=("runtime_seconds", "sum"),
    )
    .reset_index()
)

aggregate["chance_accuracy_reference"] = 1.0 / N_CLASSES
aggregate["accuracy_above_chance_pp"] = (
    aggregate["mean_test_accuracy"]
    - aggregate["chance_accuracy_reference"]
) * 100

aggregate.to_csv(
    OUTPUT_DIR / "step83_aggregate_results.csv", index=False
)

metadata = {
    "step": 83,
    "dataset": "Original BDF-derived temporal spectral bandpower arrays",
    "input_representation": "5 frequency bands x 64 EEG channels (320 features per trial)",
    "channels": channel_columns,
    "label_values": list(range(1, N_CLASSES + 1)),
    "models": list(MODEL_FACTORIES.keys()),
    "folds": len(subjects),
    "training_subjects_per_fold": 10,
    "validation_subjects_per_fold": 1,
    "test_subjects_per_fold": 1,
    "model_selection_metric": "validation macro-F1",
    "scaler": "RobustScaler fitted on training subjects only",
    "epochs_max": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "patience": PATIENCE,
    "seed": SEED,
    "device": str(DEVICE),
}

with open(OUTPUT_DIR / "step83_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\n" + "=" * 70)
print("STEP 83 COMPLETE")
print("=" * 70)
print(aggregate.to_string(index=False))
print(f"\nResults saved to: {OUTPUT_DIR.resolve()}")
print("Existing experiments were not overwritten.")