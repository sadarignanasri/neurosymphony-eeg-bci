"""
STEP 86: Improved CNNTransformer training for NeuroSymphony EEG.

Models:
1. ResidualMLP
2. CNNTransformer
3. BandAttentionNet

Uses Step 35 five-band features and the saved Step 34 splits.
StandardScaler is fitted on training data only.
Model selection uses validation macro-F1.
Test evaluation occurs after checkpoint selection.
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
EPOCHS = 120
BATCH_SIZE = 128
LR = 0.0005
WEIGHT_DECAY = 0.001
PATIENCE = 18
N_CLASSES = 8

ROOT = Path(__file__).resolve().parents[2]
FEATURE_ROOT = ROOT / "data/features"
SPLIT_ROOT = ROOT / "data/splits"
OUT = ROOT / "results/eeg/step86_cnn_transformer_corrected"
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

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResidualBlock(nn.Module):
    def __init__(self, width, dropout=0.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(width, width),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.BatchNorm1d(width),
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.block(x))


class ResidualMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(320, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.25),
            ResidualBlock(256),
            ResidualBlock(256),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(128, 8),
        )

    def forward(self, x):
        return self.net(x.flatten(1))


class CNNTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(5, 64)
        self.position = nn.Parameter(torch.zeros(1, 64, 64))
        layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0.2,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(64)
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 8),
        )

    def forward(self, x):
        # Input: batch x 5 bands x 64 channels
        x = x.transpose(1, 2)  # batch x 64 channel positions x 5 bands
        x = self.embedding(x) + self.position
        x = self.norm(self.encoder(x))
        pooled = torch.cat([x.mean(dim=1), x.max(dim=1).values], dim=1)
        return self.head(pooled)


class BandAttentionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.band_attention = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 5),
        )
        self.channel_encoder = nn.Sequential(
            nn.Conv1d(5, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Conv1d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96, 48),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(48, 8),
        )

    def forward(self, x):
        # Learn band importance separately for each trial.
        channel_summary = x.mean(dim=1)
        weights = torch.softmax(
            self.band_attention(channel_summary), dim=1
        )
        weighted_bands = x * weights.unsqueeze(-1)
        return self.head(self.channel_encoder(weighted_bands))


MODELS = {
    "CNNTransformer": CNNTransformer,
}


def load_data():
    features = {"train": [], "val": [], "test": []}
    labels = {"train": [], "val": [], "test": []}

    for subject in SUBJECTS:
        fp = (
            FEATURE_ROOT / f"temporal_spectral_{subject}"
            / f"{subject}_band_power.npy"
        )
        sd = SPLIT_ROOT / subject

        if not fp.exists():
            raise FileNotFoundError(f"Missing feature file: {fp}")

        X = np.load(fp)
        if X.shape != (720, 64, 5):
            raise ValueError(f"{subject}: unexpected shape {X.shape}")
        if not np.isfinite(X).all():
            raise ValueError(f"{subject}: NaN or infinite features")

        # Same label construction as Step 35.
        y = load_subject_labels(subject)
        used = []

        for split, expected in [
            ("train", 504), ("val", 108), ("test", 108)
        ]:
            ip = sd / f"{split}_indices.npy"
            if not ip.exists():
                raise FileNotFoundError(f"Missing split: {ip}")

            idx = np.load(ip).astype(int)
            if len(idx) != expected:
                raise ValueError(
                    f"{subject}/{split}: expected {expected}, got {len(idx)}"
                )
            used.extend(idx.tolist())
            # CNN input layout: trials x bands x channels.
            features[split].append(X[idx].transpose(0, 2, 1))
            labels[split].append(y[idx])

        if len(set(used)) != 720:
            raise ValueError(f"{subject}: split indices overlap or are incomplete")

        print(f"Validated {subject}")

    X = {k: np.concatenate(v).astype(np.float32) for k, v in features.items()}
    y = {k: np.concatenate(v) for k, v in labels.items()}

    expected = {"train": 6048, "val": 1296, "test": 1296}
    for split, count in expected.items():
        if len(X[split]) != count:
            raise ValueError(f"{split}: expected {count}, found {len(X[split])}")

    # Fit the scaler on training data only, matching Step 35 feature order.
    scaler = StandardScaler()
    scaler.fit(X["train"].transpose(0, 2, 1).reshape(-1, 320))

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

    return (
        total_loss / len(loader.dataset),
        np.asarray(true),
        np.asarray(pred),
    )


def main():
    print("=" * 72)
    print("STEP 86: CNNTRANSFORMER IMPROVEMENT")
    print("=" * 72)
    print("Device:", DEVICE)
    print("Output:", OUT.resolve())

    X, y = load_data()
    loaders = {}

    for split in ("train", "val", "test"):
        ds = TensorDataset(
            torch.from_numpy(X[split]),
            torch.from_numpy(y[split]).long(),
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=(split == "train"),
            num_workers=0,
        )

    results = []
    all_predictions = []

    for name, factory in MODELS.items():
        print(f"\n--- Training {name} ---")
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        model = factory().to(DEVICE)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            threshold=0.001,
            min_lr=1e-6,
        )
        criterion = nn.CrossEntropyLoss()
        best_state = None
        best_f1 = -1.0
        best_epoch = 0
        patience_count = 0
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item() * len(yb)

            train_loss /= len(loaders["train"].dataset)
            val_loss, vy, vp = evaluate(model, loaders["val"])
            scheduler.step(val_loss)
            val_acc = accuracy_score(vy, vp)
            val_f1 = f1_score(vy, vp, average="macro", zero_division=0)

            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "validation_accuracy": val_acc,
                "validation_macro_f1": val_f1,
                "learning_rate": optimizer.param_groups[0]["lr"],
            })

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                patience_count = 0
            else:
                patience_count += 1

            if epoch == 1 or epoch % 5 == 0:
                print(
                    f"Epoch {epoch:02d}/{EPOCHS} | "
                    f"train_loss={train_loss:.4f} | "
                    f"val_acc={val_acc:.4f} | val_macro_f1={val_f1:.4f} | "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )

            if patience_count >= PATIENCE:
                print(f"Early stopping; best epoch: {best_epoch}")
                break

        if best_state is None:
            raise RuntimeError(f"No checkpoint selected for {name}")

        model.load_state_dict(best_state)

        # Evaluate test data once, after validation-based checkpoint selection.
        _, yt, yp = evaluate(model, loaders["test"])
        test_acc = accuracy_score(yt, yp)
        test_bal = balanced_accuracy_score(yt, yp)
        test_f1 = f1_score(yt, yp, average="macro", zero_division=0)
        elapsed = time.time() - start

        md = OUT / name
        md.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(md / "training_history.csv", index=False)

        pd.DataFrame(classification_report(
            yt, yp,
            labels=list(range(8)),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )).T.to_csv(md / "classification_report.csv")

        pd.DataFrame(
            confusion_matrix(yt, yp, labels=list(range(8))),
            index=[f"true_{i}" for i in range(8)],
            columns=[f"pred_{i}" for i in range(8)],
        ).to_csv(md / "confusion_matrix.csv")

        pred_df = pd.DataFrame({
            "true_label_zero_based": yt,
            "predicted_label_zero_based": yp,
            "correct": yt == yp,
            "model": name,
        })
        pred_df.to_csv(md / "test_predictions.csv", index=False)
        all_predictions.append(pred_df)
        torch.save(model.state_dict(), md / "best_model.pt")

        results.append({
            "model": name,
            "best_epoch": best_epoch,
            "best_validation_macro_f1": best_f1,
            "test_accuracy": test_acc,
            "test_balanced_accuracy": test_bal,
            "test_macro_f1": test_f1,
            "chance_reference": 0.125,
            "runtime_seconds": elapsed,
        })

        print(
            f"{name}: test_acc={test_acc:.2%}, "
            f"balanced_acc={test_bal:.2%}, macro_f1={test_f1:.2%}, "
            f"best_epoch={best_epoch}, runtime={elapsed:.1f}s"
        )
        del model

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUT / "step86_cnn_transformer_results.csv", index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(
        OUT / "step86_test_predictions.csv", index=False
    )

    with open(OUT / "step86_metadata.json", "w") as f:
        json.dump({
            "step": 86,
            "models": list(MODELS.keys()),
            "subjects": 12,
            "features_per_trial": 320,
            "input_layout": "5 bands x 64 channel positions",
            "split_sizes": {"train": 6048, "validation": 1296, "test": 1296},
            "scaler": "StandardScaler fitted on training data only",
            "selection_metric": "validation macro-F1",
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "seed": SEED,
            "device": str(DEVICE),
        }, f, indent=2)

    print("\n" + "=" * 72)
    print("STEP 86 COMPLETE")
    print("=" * 72)
    print(result_df.to_string(index=False))
    print("\nResults saved to:", OUT.resolve())
    print("Existing experiments were not overwritten.")


if __name__ == "__main__":
    main()
