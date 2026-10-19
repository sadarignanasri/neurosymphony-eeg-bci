"""
STEP 84: Train CNN architectures on the same five-band EEG features
used by the Step 35 traditional-model baseline.

Models:
1. DeepConvNet
2. ShallowConvNet
3. MultiScaleCNN

Features: 64 channels x 5 bands = 320 features per trial.
Splits: 504 train, 108 validation, 108 test per subject.
All 12 subjects are pooled, matching Step 35.
Scaler is fitted only on training data.
Existing experiments are preserved.
"""

from pathlib import Path
from label_utils import load_subject_labels
import copy
import json
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

SEED = 42
N_CLASSES = 8
EPOCHS = 60
BATCH_SIZE = 128
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.001
PATIENCE = 10

ROOT = Path(__file__).resolve().parents[2]
FEATURE_ROOT = ROOT / "data/features"
SPLIT_ROOT = ROOT / "data/splits"
OUT = ROOT / "results/eeg/step84_bandpower_cnn_corrected"
OUT.mkdir(parents=True, exist_ok=True)

SUBJECTS = [f"sub-{i:02d}" for i in range(1, 13)]
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
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeepConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(5, 25, 5, padding=2, bias=False),
            nn.BatchNorm1d(25), nn.ELU(), nn.MaxPool1d(2),
            nn.Dropout(0.35),
            nn.Conv1d(25, 50, 5, padding=2, bias=False),
            nn.BatchNorm1d(50), nn.ELU(), nn.MaxPool1d(2),
            nn.Dropout(0.35),
            nn.Conv1d(50, 100, 3, padding=1, bias=False),
            nn.BatchNorm1d(100), nn.ELU(), nn.MaxPool1d(2),
            nn.Dropout(0.35),
            nn.Conv1d(100, 150, 3, padding=1, bias=False),
            nn.BatchNorm1d(150), nn.ELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.4), nn.Linear(150, 8)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class ShallowConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(5, 40, 13, padding=6, bias=False)
        self.norm = nn.BatchNorm1d(40)
        self.pool = nn.AvgPool1d(4, stride=2)
        self.dropout = nn.Dropout(0.4)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(40, 8)

    def forward(self, x):
        x = self.norm(self.conv(x))
        x = self.pool(x.square())
        x = torch.log(torch.clamp(x, min=1e-6))
        x = self.dropout(x)
        return self.classifier(self.global_pool(x).flatten(1))


class MultiScaleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch3 = nn.Sequential(
            nn.Conv1d(5, 32, 3, padding=1, bias=False),
            nn.BatchNorm1d(32), nn.ReLU()
        )
        self.branch7 = nn.Sequential(
            nn.Conv1d(5, 32, 7, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.ReLU()
        )
        self.branch15 = nn.Sequential(
            nn.Conv1d(5, 32, 15, padding=7, bias=False),
            nn.BatchNorm1d(32), nn.ReLU()
        )
        self.combine = nn.Sequential(
            nn.Conv1d(96, 96, 3, padding=1, bias=False),
            nn.BatchNorm1d(96), nn.ReLU(), nn.MaxPool1d(2),
            nn.Dropout(0.3),
            nn.Conv1d(96, 128, 3, padding=1, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.4), nn.Linear(128, 8)
        )

    def forward(self, x):
        x = torch.cat(
            [self.branch3(x), self.branch7(x), self.branch15(x)], dim=1
        )
        return self.classifier(self.combine(x))


MODELS = {
    "DeepConvNet": DeepConvNet,
    "ShallowConvNet": ShallowConvNet,
    "MultiScaleCNN": MultiScaleCNN,
}


def load_data():
    X_parts = {"train": [], "val": [], "test": []}
    y_parts = {"train": [], "val": [], "test": []}

    for subject in SUBJECTS:
        feature_file = (
            FEATURE_ROOT / f"temporal_spectral_{subject}"
            / f"{subject}_band_power.npy"
        )
        split_dir = SPLIT_ROOT / subject

        if not feature_file.exists():
            raise FileNotFoundError(f"Missing feature file: {feature_file}")

        X = np.load(feature_file)
        if X.shape != (720, 64, 5):
            raise ValueError(f"{subject}: unexpected shape {X.shape}")
        if not np.isfinite(X).all():
            raise ValueError(f"{subject}: NaN or infinite features found")

        # Matches Step 35's label construction.
        y = load_subject_labels(subject)
        used_indices = []

        for split, expected in [
            ("train", 504), ("val", 108), ("test", 108)
        ]:
            index_file = split_dir / f"{split}_indices.npy"
            if not index_file.exists():
                raise FileNotFoundError(f"Missing split file: {index_file}")

            idx = np.load(index_file).astype(int)
            if len(idx) != expected:
                raise ValueError(
                    f"{subject}/{split}: expected {expected}, got {len(idx)}"
                )
            used_indices.extend(idx.tolist())

            # Keep each trial as (5 bands, 64 channel positions).
            X_parts[split].append(X[idx].transpose(0, 2, 1))
            y_parts[split].append(y[idx])

        if len(set(used_indices)) != 720:
            raise ValueError(
                f"{subject}: saved splits overlap or do not cover all trials"
            )

        print(f"Validated {subject}: features {X.shape}")

    X = {k: np.concatenate(v).astype(np.float32) for k, v in X_parts.items()}
    y = {k: np.concatenate(v) for k, v in y_parts.items()}

    expected_sizes = {"train": 6048, "val": 1296, "test": 1296}
    for split, expected in expected_sizes.items():
        if len(X[split]) != expected:
            raise ValueError(f"{split}: expected {expected}, got {len(X[split])}")

    # Match Step 35 flattening order (channel, band) before scaling.
    scaler = StandardScaler()
    train_flat = X["train"].transpose(0, 2, 1).reshape(-1, 320)
    scaler.fit(train_flat)

    for split in ("train", "val", "test"):
        flat = X[split].transpose(0, 2, 1).reshape(-1, 320)
        scaled = scaler.transform(flat).astype(np.float32)
        X[split] = scaled.reshape(-1, 64, 5).transpose(0, 2, 1)

    return X, y


def evaluate(model, loader):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    true, pred = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            total_loss += loss_fn(logits, yb).item() * len(yb)
            true.extend(yb.cpu().numpy())
            pred.extend(logits.argmax(1).cpu().numpy())

    return total_loss / len(loader.dataset), np.array(true), np.array(pred)


def main():
    print("=" * 72)
    print("STEP 84: CNNs USING FIVE-BAND EEG FEATURES")
    print("=" * 72)
    print("Device:", DEVICE)
    print("Input per trial: (5 bands, 64 channels)")
    print("Expected split: 6048 train / 1296 validation / 1296 test")
    print("Output:", OUT.resolve())

    X, y = load_data()
    loaders = {}

    for split in ("train", "val", "test"):
        ds = TensorDataset(
            torch.from_numpy(X[split]),
            torch.from_numpy(y[split]).long(),
        )
        loaders[split] = DataLoader(
            ds, batch_size=BATCH_SIZE,
            shuffle=(split == "train"), num_workers=0
        )

    results = []
    all_predictions = []

    for model_name, factory in MODELS.items():
        print(f"\n--- Training {model_name} ---")
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        model = factory().to(DEVICE)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        criterion = nn.CrossEntropyLoss()

        best_state = None
        best_val_f1 = -1.0
        best_epoch = 0
        wait = 0
        history = []
        start = time.time()

        for epoch in range(1, EPOCHS + 1):
            model.train()
            train_loss = 0.0

            for xb, yb in loaders["train"]:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(yb)

            train_loss /= len(loaders["train"].dataset)
            val_loss, yv, pv = evaluate(model, loaders["val"])
            val_acc = accuracy_score(yv, pv)
            val_f1 = f1_score(yv, pv, average="macro", zero_division=0)

            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "validation_accuracy": val_acc,
                "validation_macro_f1": val_f1,
            })

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1

            if epoch == 1 or epoch % 5 == 0:
                print(
                    f"Epoch {epoch:02d}/{EPOCHS} | "
                    f"train_loss={train_loss:.4f} | "
                    f"val_acc={val_acc:.4f} | val_macro_f1={val_f1:.4f}"
                )

            if wait >= PATIENCE:
                print(f"Early stopping; best epoch: {best_epoch}")
                break

        if best_state is None:
            raise RuntimeError(f"No checkpoint selected for {model_name}")

        model.load_state_dict(best_state)
        _, yt, pt = evaluate(model, loaders["test"])
        test_acc = accuracy_score(yt, pt)
        test_bal_acc = balanced_accuracy_score(yt, pt)
        test_f1 = f1_score(yt, pt, average="macro", zero_division=0)
        runtime = time.time() - start

        model_dir = OUT / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(history).to_csv(
            model_dir / "training_history.csv", index=False
        )
        pd.DataFrame(classification_report(
            yt, pt,
            labels=list(range(8)),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )).T.to_csv(model_dir / "classification_report.csv")

        pd.DataFrame(
            confusion_matrix(yt, pt, labels=list(range(8))),
            index=[f"true_{i}" for i in range(8)],
            columns=[f"pred_{i}" for i in range(8)],
        ).to_csv(model_dir / "confusion_matrix.csv")

        pred_df = pd.DataFrame({
            "true_label_zero_based": yt,
            "predicted_label_zero_based": pt,
            "correct": yt == pt,
            "model": model_name,
        })
        pred_df.to_csv(model_dir / "test_predictions.csv", index=False)
        all_predictions.append(pred_df)
        torch.save(model.state_dict(), model_dir / "best_model.pt")

        results.append({
            "model": model_name,
            "train_trials": 6048,
            "validation_trials": 1296,
            "test_trials": 1296,
            "best_epoch": best_epoch,
            "best_validation_macro_f1": best_val_f1,
            "test_accuracy": test_acc,
            "test_balanced_accuracy": test_bal_acc,
            "test_macro_f1": test_f1,
            "chance_accuracy_reference": 0.125,
            "step35_logistic_regression_test_accuracy": 0.4212962963,
            "difference_vs_step35_percentage_points": (
                test_acc - 0.4212962963
            ) * 100,
            "runtime_seconds": runtime,
        })

        print(
            f"{model_name}: test accuracy={test_acc:.2%}, "
            f"balanced accuracy={test_bal_acc:.2%}, "
            f"macro-F1={test_f1:.2%}, best epoch={best_epoch}"
        )
        del model

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT / "step84_cnn_results.csv", index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(
        OUT / "step84_test_predictions_all_models.csv", index=False
    )

    with open(OUT / "step84_metadata.json", "w") as f:
        json.dump({
            "step": 84,
            "feature_source": "data/features/temporal_spectral_sub-XX/sub-XX_band_power.npy",
            "feature_shape_per_subject": [720, 64, 5],
            "cnn_input_shape": [5, 64],
            "bands": BANDS,
            "subjects": 12,
            "split_trials": {"train": 6048, "validation": 1296, "test": 1296},
            "scaler": "StandardScaler fitted on training data only",
            "model_selection_metric": "validation macro-F1",
            "seed": SEED,
            "device": str(DEVICE),
            "step35_test_accuracy": 0.4212962963,
        }, f, indent=2)

    print("\n" + "=" * 72)
    print("STEP 84 COMPLETE")
    print("=" * 72)
    print(results_df.to_string(index=False))
    print("\nResults saved to:", OUT.resolve())
    print("Existing experiments were not overwritten.")


if __name__ == "__main__":
    main()
