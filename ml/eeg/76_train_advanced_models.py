
from pathlib import Path
import copy
import json
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)


ROOT = Path.cwd()
RESULTS_DIR = ROOT / "results/eeg/advanced_models/step76"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
EPOCHS = 50
BATCH_SIZE = 64
PATIENCE = 8
LEARNING_RATE = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu" and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def get_subject_paths(subject_num):
    subject = f"sub-{subject_num:02d}"

    if subject_num == 1:
        feature_path = (
            ROOT / "data/features/temporal_spectral_sub01"
            / "sub-01_band_power.npy"
        )
        metadata_path = (
            ROOT / "data/features/temporal_spectral_sub01"
            / "sub-01_spectral_metadata.csv"
        )
        split_dir = ROOT / "data/splits/sub01"
    else:
        feature_path = (
            ROOT / f"data/features/temporal_spectral_{subject}"
            / f"{subject}_band_power.npy"
        )
        metadata_path = (
            ROOT / f"data/features/temporal_eeg_{subject}"
            / f"{subject}_temporal_eeg_preprocessed_metadata.csv"
        )
        split_dir = ROOT / f"data/splits/{subject}"

    return feature_path, metadata_path, split_dir


def load_subject(subject_num):
    feature_path, metadata_path, split_dir = get_subject_paths(subject_num)

    for path in (feature_path, metadata_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    features = np.load(feature_path).astype(np.float32)
    metadata = pd.read_csv(metadata_path)

    label_column = "label" if subject_num == 1 else "label_id"
    if label_column not in metadata.columns:
        raise ValueError(
            f"{metadata_path} does not contain {label_column}"
        )

    labels = metadata[label_column].to_numpy(dtype=np.int64)

    if features.shape != (720, 64, 5):
        raise ValueError(
            f"Unexpected shape for subject {subject_num}: {features.shape}"
        )

    if len(labels) != len(features):
        raise ValueError("Feature and label counts do not match")

    if not np.isfinite(features).all():
        raise ValueError(f"Non-finite features in subject {subject_num}")

    if set(np.unique(labels)) != set(range(8)):
        raise ValueError(
            f"Expected labels 0-7 for subject {subject_num}; "
            f"found {np.unique(labels)}"
        )

    train_idx = np.load(split_dir / "train_indices.npy")
    val_idx = np.load(split_dir / "val_indices.npy")
    test_idx = np.load(split_dir / "test_indices.npy")

    if (
        np.intersect1d(train_idx, val_idx).size
        or np.intersect1d(train_idx, test_idx).size
        or np.intersect1d(val_idx, test_idx).size
    ):
        raise ValueError(f"Overlapping splits for subject {subject_num}")

    return features, labels, train_idx, val_idx, test_idx


class MLP(nn.Module):
    def __init__(self, input_dim=320, num_classes=8):
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
    def __init__(self, num_classes=8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(20),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 20, 128),
            nn.ReLU(),
            nn.Dropout(0.40),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.classifier(self.features(x))


def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(
                y_true, y_pred, average="macro", zero_division=0
            )
        ),
    }


def fit_model(model, X_train, y_train, X_val, y_val, model_name):
    model = model.to(DEVICE)

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_x = torch.from_numpy(X_val).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-3,
    )
    criterion = nn.CrossEntropyLoss()

    best_f1 = -1.0
    best_state = None
    best_epoch = 0
    patience_count = 0
    history = []
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_y)
            total_examples += len(batch_y)

        model.eval()
        with torch.no_grad():
            val_logits = model(val_x)
            val_pred = val_logits.argmax(dim=1).cpu().numpy()

        metrics = calculate_metrics(y_val, val_pred)
        train_loss = total_loss / max(total_examples, 1)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_accuracy": metrics["accuracy"],
            "val_balanced_accuracy": metrics["balanced_accuracy"],
            "val_macro_f1": metrics["macro_f1"],
        })

        print(
            f"    {model_name} epoch {epoch:02d} | "
            f"loss={train_loss:.4f} | "
            f"val_macro_f1={metrics['macro_f1']:.4f}"
        )

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= PATIENCE:
            print(f"    Early stopping at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError(f"No checkpoint produced for {model_name}")

    model.load_state_dict(best_state)

    elapsed = time.time() - start
    return model, pd.DataFrame(history), best_epoch, elapsed


def main():
    print(f"Device: {DEVICE}")
    print(f"Output: {RESULTS_DIR}")

    summary_rows = []

    for subject_num in range(1, 13):
        subject = f"sub-{subject_num:02d}"
        print(f"\n{'=' * 60}\n{subject}\n{'=' * 60}")

        features, labels, train_idx, val_idx, test_idx = load_subject(
            subject_num
        )

        # Flatten in the existing channel-major, band-minor order.
        X = features.reshape(len(features), -1)

        X_train_raw = X[train_idx]
        X_val_raw = X[val_idx]

        # Standardize using training data only.
        mean = X_train_raw.mean(axis=0, keepdims=True)
        std = X_train_raw.std(axis=0, keepdims=True)
        std[std < 1e-8] = 1.0

        X_train = ((X_train_raw - mean) / std).astype(np.float32)
        X_val = ((X_val_raw - mean) / std).astype(np.float32)

        y_train = labels[train_idx]
        y_val = labels[val_idx]

        for model_name in ("MLP", "CNN1D"):
            print(f"\nTraining {model_name}")

            # Reset seed so each model starts reproducibly.
            torch.manual_seed(SEED)

            if model_name == "MLP":
                model = MLP()
            else:
                model = EEGConv1D()

            trained_model, history, best_epoch, elapsed = fit_model(
                model,
                X_train,
                y_train,
                X_val,
                y_val,
                model_name,
            )

            model_dir = RESULTS_DIR / subject / model_name
            model_dir.mkdir(parents=True, exist_ok=True)

            checkpoint_path = model_dir / "best_model.pt"
            torch.save(
                {
                    "model_state_dict": trained_model.state_dict(),
                    "model_name": model_name,
                    "subject": subject,
                    "input_shape": [64, 5],
                    "flatten_order": "channel-major, band-minor",
                    "feature_mean": mean.astype(np.float32),
                    "feature_std": std.astype(np.float32),
                    "best_epoch": best_epoch,
                    "seed": SEED,
                },
                checkpoint_path,
            )

            history.to_csv(model_dir / "training_history.csv", index=False)

            trained_model.eval()
            with torch.no_grad():
                logits = trained_model(
                    torch.from_numpy(X_val).to(DEVICE)
                )
                predictions = logits.argmax(dim=1).cpu().numpy()

            metrics = calculate_metrics(y_val, predictions)

            row = {
                "subject": subject,
                "model": model_name,
                **metrics,
                "best_epoch": best_epoch,
                "training_seconds": round(elapsed, 2),
                "checkpoint": str(checkpoint_path),
            }
            summary_rows.append(row)

            print(
                f"  BEST {model_name}: "
                f"accuracy={metrics['accuracy']:.4f}, "
                f"balanced_accuracy={metrics['balanced_accuracy']:.4f}, "
                f"macro_f1={metrics['macro_f1']:.4f}, "
                f"epoch={best_epoch}"
            )

    summary = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "validation_summary.csv"
    summary.to_csv(summary_path, index=False)

    aggregate = (
        summary.groupby("model")[
            ["accuracy", "balanced_accuracy", "macro_f1"]
        ]
        .agg(["mean", "std"])
    )

    aggregate_path = RESULTS_DIR / "validation_aggregate.csv"
    aggregate.to_csv(aggregate_path)

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY ACROSS 12 SUBJECTS")
    print("=" * 60)
    print(aggregate.to_string())
    print(f"\nSaved per-subject summary: {summary_path}")
    print(f"Saved aggregate summary: {aggregate_path}")
    print("Test data was not evaluated or used for model selection.")


if __name__ == "__main__":
    main()
