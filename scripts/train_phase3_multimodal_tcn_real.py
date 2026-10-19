"""
Neuro-Symphony Phase 3 — Real Multimodal TCN Training.

This script trains the complete Phase 3 temporal representation
pipeline on the real ds004514 dataset.

Pipeline:

    EEG
      ↓
    EEG TCN
      ↓
    EEG temporal representation
      ↓
                    ┐
                    ├──> Multimodal representation
                    │
    fNIRS           │
      ↓             │
    fNIRS TCN ──────┘
      ↓
    fNIRS temporal representation
                    ↓
              Classifier
                    ↓
              Task prediction

The EEG TCN, fNIRS TCN, multimodal representation, and
classifier are optimized jointly using classification loss.

This is a within-subject Phase 3 validation.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.acquisition.eeg_loader import (
    EEGData,
    EEGSourceKind,
)

from src.acquisition.fnirs_loader import (
    FNIRSData,
    FNIRSSourceKind,
)

from src.acquisition.recording import (
    RecordingLayout,
    RecordingModality,
)

from src.epochs.epoch_extractor import (
    EpochConfig,
    EpochEvent,
    extract_eeg_epochs,
    extract_fnirs_epochs,
)

from src.models.multimodal_tcn_classifier import (
    MultimodalTCNClassifier,
    MultimodalTCNClassifierConfig,
    multimodal_classification_loss,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path(
    "/Users/gnanasrisadari/Desktop/ds004514"
)

SUBJECT = "sub-01"

EEG_FILE = (
    DATASET_ROOT
    / SUBJECT
    / "eeg"
    / f"{SUBJECT}_task-eeg_eeg.bdf"
)

FNIRS_FILE = (
    DATASET_ROOT
    / SUBJECT
    / "nirs"
    / f"{SUBJECT}_task-nirs_nirs.snirf"
)

EEG_EVENTS_FILE = (
    DATASET_ROOT
    / SUBJECT
    / "eeg"
    / f"{SUBJECT}_task-eeg_events.tsv"
)

FNIRS_EVENTS_FILE = (
    DATASET_ROOT
    / SUBJECT
    / "nirs"
    / f"{SUBJECT}_task-nirs_events.tsv"
)


TASK_LABELS = (
    "animal_tactile_imagery_task",
    "animal_visual_imagery_task",
    "animal_auditory_imagery_task",
    "animal_silent_naming_task",
    "tool_tactile_imagery_task",
    "tool_visual_imagery_task",
    "tool_auditory_imagery_task",
    "tool_silent_naming_task",
)


RANDOM_SEED = 42

TEST_SIZE = 0.20

BATCH_SIZE = 8

NUM_EPOCHS = 20

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ============================================================
# DEVICE
# ============================================================

def select_device() -> torch.device:

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# EVENT LOADING
# ============================================================

def load_task_events(
    path: Path,
) -> list[EpochEvent]:

    table = pd.read_csv(
        path,
        sep="\t",
    )

    events: list[EpochEvent] = []

    for _, row in table.iterrows():

        trial_type = str(
            row["trial_type"]
        )

        if trial_type not in TASK_LABELS:
            continue

        value = None

        if "value" in table.columns:

            raw_value = row["value"]

            if pd.notna(raw_value):
                value = raw_value

        events.append(
            EpochEvent(
                onset=float(
                    row["onset"]
                ),
                trial_type=trial_type,
                value=value,
                subject=SUBJECT,
            )
        )

    return events


# ============================================================
# EEG LOADING
# ============================================================

def load_eeg() -> EEGData:

    print(
        "[1] Loading EEG..."
    )

    raw = mne.io.read_raw_bdf(
        EEG_FILE,
        preload=True,
        verbose="ERROR",
    )

    data = np.asarray(
        raw.get_data(),
        dtype=np.float64,
    )

    eeg = EEGData(
        recording_id=(
            f"{SUBJECT}_task-eeg"
        ),
        recording_path=EEG_FILE,
        modality=RecordingModality.EEG,
        layout=RecordingLayout.BIDS,
        subject_id=SUBJECT,
        session_id=None,
        task_id="eeg",
        run_id=None,
        source_kind=EEGSourceKind.FILE,
        adapter_name="real_training_mne",
        sampling_rate=float(
            raw.info["sfreq"]
        ),
        channel_names=tuple(
            raw.ch_names
        ),
        data=data,
        channel_types=tuple(
            raw.get_channel_types()
        ),
        eeg_metadata=None,
        metadata={},
    )

    print(
        f"EEG channels: "
        f"{eeg.n_channels}"
    )

    print(
        f"EEG sampling rate: "
        f"{eeg.sampling_rate:.6f} Hz"
    )

    return eeg


# ============================================================
# fNIRS LOADING
# ============================================================

def load_fnirs() -> FNIRSData:

    print(
        "[2] Loading fNIRS..."
    )

    raw = mne.io.read_raw_snirf(
        FNIRS_FILE,
        preload=True,
        verbose="ERROR",
    )

    data = np.asarray(
        raw.get_data(),
        dtype=np.float64,
    )

    fnirs = FNIRSData(
        recording_id=(
            f"{SUBJECT}_task-nirs"
        ),
        recording_path=FNIRS_FILE,
        modality=RecordingModality.FNIRS,
        layout=RecordingLayout.BIDS,
        subject_id=SUBJECT,
        session_id=None,
        task_id="nirs",
        run_id=None,
        source_kind=FNIRSSourceKind.FILE,
        adapter_name="real_training_mne",
        sampling_rate=float(
            raw.info["sfreq"]
        ),
        channel_names=tuple(
            raw.ch_names
        ),
        data=data,
        channel_types=tuple(
            raw.get_channel_types()
        ),
        fnirs_metadata=None,
        metadata={},
    )

    print(
        f"fNIRS channels: "
        f"{fnirs.n_channels}"
    )

    print(
        f"fNIRS sampling rate: "
        f"{fnirs.sampling_rate:.6f} Hz"
    )

    return fnirs


# ============================================================
# EPOCH STANDARDIZATION
# ============================================================

def standardize_epochs(
    data: np.ndarray,
    train_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize each channel using training-set statistics.

    Input:
        (epochs, channels, samples)

    Returns:
        standardized_data
        channel_mean
        channel_std

    Statistics are calculated ONLY from the training set to
    prevent test-set information leakage.
    """

    train_data = data[
        train_indices
    ]

    channel_mean = np.mean(
        train_data,
        axis=(0, 2),
        keepdims=True,
    )

    channel_std = np.std(
        train_data,
        axis=(0, 2),
        keepdims=True,
    )

    channel_std = np.maximum(
        channel_std,
        1e-6,
    )

    standardized = (
        data - channel_mean
    ) / channel_std

    if not np.isfinite(
        standardized
    ).all():

        raise ValueError(
            "Standardized epochs contain "
            "NaN or infinite values."
        )

    return (
        standardized.astype(
            np.float32
        ),
        channel_mean,
        channel_std,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 72)

    print(
        "NEURO-SYMPHONY PHASE 3 — "
        "REAL MULTIMODAL TCN TRAINING"
    )

    print("=" * 72)

    print()

    print(
        f"Dataset: {DATASET_ROOT}"
    )

    print(
        f"Subject: {SUBJECT}"
    )

    print()

    # ========================================================
    # 1. LOAD SIGNALS
    # ========================================================

    eeg = load_eeg()

    print()

    fnirs = load_fnirs()

    print()

    # ========================================================
    # 2. LOAD TASK EVENTS
    # ========================================================

    print(
        "[3] Loading task events..."
    )

    eeg_events = load_task_events(
        EEG_EVENTS_FILE
    )

    fnirs_events = load_task_events(
        FNIRS_EVENTS_FILE
    )

    print(
        f"EEG task events: "
        f"{len(eeg_events)}"
    )

    print(
        f"fNIRS task events: "
        f"{len(fnirs_events)}"
    )

    if len(eeg_events) != 720:
        raise AssertionError(
            "Expected 720 EEG task events."
        )

    if len(fnirs_events) != 720:
        raise AssertionError(
            "Expected 720 fNIRS task events."
        )

    print(
        "PASS: 720 corresponding task events loaded."
    )

    print()

    # ========================================================
    # 3. EXTRACT EPOCHS
    # ========================================================

    print(
        "[4] Extracting EEG epochs..."
    )

    epoch_config = EpochConfig(
        tmin=0.0,
        tmax=3.0,
        reject_nonfinite=True,
    )

    eeg_epochs = extract_eeg_epochs(
        eeg,
        eeg_events,
        config=epoch_config,
    )

    print(
        f"EEG epochs: "
        f"{len(eeg_epochs)}"
    )

    print(
        "[4b] Extracting fNIRS epochs..."
    )

    fnirs_epochs = extract_fnirs_epochs(
        fnirs,
        fnirs_events,
        config=epoch_config,
    )

    print(
        f"fNIRS epochs: "
        f"{len(fnirs_epochs)}"
    )

    if len(eeg_epochs) != 720:
        raise AssertionError(
            "Expected 720 EEG epochs."
        )

    if len(fnirs_epochs) != 720:
        raise AssertionError(
            "Expected 720 fNIRS epochs."
        )

    print(
        "PASS: 720 synchronized epochs available."
    )

    print()

    # ========================================================
    # 4. CREATE ARRAYS
    # ========================================================

    print(
        "[5] Creating epoch arrays..."
    )

    eeg_array = np.stack(
        [
            epoch.data
            for epoch in eeg_epochs
        ],
        axis=0,
    )

    fnirs_array = np.stack(
        [
            epoch.data
            for epoch in fnirs_epochs
        ],
        axis=0,
    )

    print(
        f"EEG array: "
        f"{eeg_array.shape}"
    )

    print(
        f"fNIRS array: "
        f"{fnirs_array.shape}"
    )

    # ========================================================
    # 5. LABELS
    # ========================================================

    print(
        "[6] Creating task labels..."
    )

    label_to_index = {
        label: index
        for index, label in enumerate(
            TASK_LABELS
        )
    }

    labels = np.asarray(
        [
            label_to_index[
                epoch.trial_type
            ]
            for epoch in eeg_epochs
        ],
        dtype=np.int64,
    )

    if len(labels) != 720:
        raise AssertionError(
            "Expected 720 labels."
        )

    counts = Counter(
        labels.tolist()
    )

    for index, label in enumerate(
        TASK_LABELS
    ):

        print(
            f"{label:40s}: "
            f"{counts[index]}"
        )

        if counts[index] != 90:
            raise AssertionError(
                f"Class '{label}' does not "
                "contain exactly 90 samples."
            )

    print(
        "PASS: All 8 classes contain 90 samples."
    )

    print()

    # ========================================================
    # 6. TRAIN / TEST SPLIT
    # ========================================================

    print(
        "[7] Creating stratified train/test split..."
    )

    indices = np.arange(
        len(labels)
    )

    train_indices, test_indices = (
        train_test_split(
            indices,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            stratify=labels,
        )
    )

    print(
        f"Training samples: "
        f"{len(train_indices)}"
    )

    print(
        f"Testing samples: "
        f"{len(test_indices)}"
    )

    if len(train_indices) != 576:
        raise AssertionError(
            "Expected 576 training samples."
        )

    if len(test_indices) != 144:
        raise AssertionError(
            "Expected 144 test samples."
        )

    print(
        "PASS: Stratified 80/20 split created."
    )

    print()

    # ========================================================
    # 7. STANDARDIZE USING TRAINING DATA ONLY
    # ========================================================

    print(
        "[8] Standardizing EEG epochs..."
    )

    eeg_array, _, _ = standardize_epochs(
        eeg_array,
        train_indices,
    )

    print(
        "PASS: EEG standardized using "
        "training-set statistics."
    )

    print(
        "[8b] Standardizing fNIRS epochs..."
    )

    fnirs_array, _, _ = standardize_epochs(
        fnirs_array,
        train_indices,
    )

    print(
        "PASS: fNIRS standardized using "
        "training-set statistics."
    )

    print()

    # ========================================================
    # 8. CREATE TENSORS
    # ========================================================

    print(
        "[9] Creating PyTorch tensors..."
    )

    eeg_tensor = torch.from_numpy(
        eeg_array
    )

    fnirs_tensor = torch.from_numpy(
        fnirs_array
    )

    labels_tensor = torch.from_numpy(
        labels
    )

    print(
        f"EEG tensor: "
        f"{tuple(eeg_tensor.shape)}"
    )

    print(
        f"fNIRS tensor: "
        f"{tuple(fnirs_tensor.shape)}"
    )

    print()

    # ========================================================
    # 9. DEVICE
    # ========================================================

    print(
        "[10] Selecting computation device..."
    )

    device = select_device()

    print(
        f"Device: {device}"
    )

    print()

    # ========================================================
    # 10. DATASETS
    # ========================================================

    print(
        "[11] Creating train/test datasets..."
    )

    train_dataset = TensorDataset(
        eeg_tensor[
            train_indices
        ],
        fnirs_tensor[
            train_indices
        ],
        labels_tensor[
            train_indices
        ],
    )

    test_dataset = TensorDataset(
        eeg_tensor[
            test_indices
        ],
        fnirs_tensor[
            test_indices
        ],
        labels_tensor[
            test_indices
        ],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print(
        f"Training batches: "
        f"{len(train_loader)}"
    )

    print(
        f"Testing batches: "
        f"{len(test_loader)}"
    )

    print()

    # ========================================================
    # 11. CREATE END-TO-END MODEL
    # ========================================================

    print(
        "[12] Creating trainable multimodal TCN..."
    )

    model = MultimodalTCNClassifier(
        MultimodalTCNClassifierConfig(
            eeg_channels=eeg.n_channels,
            fnirs_channels=fnirs.n_channels,
            num_classes=len(
                TASK_LABELS
            ),
            embedding_dimension=128,
            eeg_hidden_channels=32,
            fnirs_hidden_channels=32,
            eeg_num_blocks=4,
            fnirs_num_blocks=3,
            kernel_size=3,
            dropout=0.1,
            classifier_hidden_dimension=128,
        )
    ).to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{parameter_count:,}"
    )

    print(
        "PASS: End-to-end multimodal TCN created."
    )

    print()

    # ========================================================
    # 12. OPTIMIZER
    # ========================================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # ========================================================
    # 13. TRAINING
    # ========================================================

    print(
        "[13] Training EEG TCN + fNIRS TCN + "
        "classifier jointly..."
    )

    print()

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):

        model.train()

        running_loss = 0.0

        correct = 0

        total = 0

        for (
            batch_eeg,
            batch_fnirs,
            batch_labels,
        ) in train_loader:

            batch_eeg = (
                batch_eeg.to(
                    device
                )
            )

            batch_fnirs = (
                batch_fnirs.to(
                    device
                )
            )

            batch_labels = (
                batch_labels.to(
                    device
                )
            )

            optimizer.zero_grad()

            logits = model(
                batch_eeg,
                batch_fnirs,
            )

            loss = (
                multimodal_classification_loss(
                    logits,
                    batch_labels,
                )
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
                * batch_labels.size(0)
            )

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            correct += int(
                (
                    predictions
                    == batch_labels
                ).sum().item()
            )

            total += (
                batch_labels.size(0)
            )

        train_loss = (
            running_loss
            / total
        )

        train_accuracy = (
            correct
            / total
        )

        # ----------------------------------------------------
        # Test evaluation
        # ----------------------------------------------------

        model.eval()

        test_correct = 0

        test_total = 0

        test_loss_total = 0.0

        with torch.no_grad():

            for (
                batch_eeg,
                batch_fnirs,
                batch_labels,
            ) in test_loader:

                batch_eeg = (
                    batch_eeg.to(
                        device
                    )
                )

                batch_fnirs = (
                    batch_fnirs.to(
                        device
                    )
                )

                batch_labels = (
                    batch_labels.to(
                        device
                    )
                )

                logits = model(
                    batch_eeg,
                    batch_fnirs,
                )

                loss = (
                    multimodal_classification_loss(
                        logits,
                        batch_labels,
                    )
                )

                predictions = torch.argmax(
                    logits,
                    dim=1,
                )

                test_loss_total += (
                    loss.item()
                    * batch_labels.size(0)
                )

                test_correct += int(
                    (
                        predictions
                        == batch_labels
                    ).sum().item()
                )

                test_total += (
                    batch_labels.size(0)
                )

        test_loss = (
            test_loss_total
            / test_total
        )

        test_accuracy = (
            test_correct
            / test_total
        )

        print(
            f"Epoch "
            f"{epoch:02d}/{NUM_EPOCHS} | "
            f"Train Loss: "
            f"{train_loss:.6f} | "
            f"Train Acc: "
            f"{train_accuracy:.4f} | "
            f"Test Loss: "
            f"{test_loss:.6f} | "
            f"Test Acc: "
            f"{test_accuracy:.4f}"
        )

    print()

    print(
        "PASS: Joint TCN/classifier training completed."
    )

    print()

    # ========================================================
    # 14. FINAL TEST PREDICTIONS
    # ========================================================

    print(
        "[14] Generating final test predictions..."
    )

    model.eval()

    all_true = []

    all_pred = []

    all_logits = []

    with torch.no_grad():

        for (
            batch_eeg,
            batch_fnirs,
            batch_labels,
        ) in test_loader:

            batch_eeg = (
                batch_eeg.to(
                    device
                )
            )

            batch_fnirs = (
                batch_fnirs.to(
                    device
                )
            )

            logits = model(
                batch_eeg,
                batch_fnirs,
            )

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            all_true.extend(
                batch_labels.numpy()
            )

            all_pred.extend(
                predictions.cpu().numpy()
            )

            all_logits.append(
                logits.cpu()
            )

    y_true = np.asarray(
        all_true,
        dtype=np.int64,
    )

    y_pred = np.asarray(
        all_pred,
        dtype=np.int64,
    )

    logits = torch.cat(
        all_logits,
        dim=0,
    ).numpy()

    # ========================================================
    # 15. FINAL METRICS
    # ========================================================

    final_accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    print(
        f"Final test accuracy: "
        f"{final_accuracy:.4f}"
    )

    print()

    print(
        "[15] Classification report..."
    )

    print(
        classification_report(
            y_true,
            y_pred,
            labels=np.arange(
                len(TASK_LABELS)
            ),
            target_names=TASK_LABELS,
            zero_division=0,
        )
    )

    # ========================================================
    # 16. CONFUSION MATRIX
    # ========================================================

    print(
        "[16] Confusion matrix..."
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(
            len(TASK_LABELS)
        ),
    )

    print(matrix)

    print()

    # ========================================================
    # 17. REPRESENTATION VALIDATION
    # ========================================================

    print(
        "[17] Extracting learned neural representations..."
    )

    sample_eeg = (
        eeg_tensor[
            test_indices[:8]
        ].to(device)
    )

    sample_fnirs = (
        fnirs_tensor[
            test_indices[:8]
        ].to(device)
    )

    with torch.no_grad():

        outputs = model(
            sample_eeg,
            sample_fnirs,
            return_representations=True,
        )

    eeg_representation = (
        outputs[
            "eeg_representation"
        ]
    )

    fnirs_representation = (
        outputs[
            "fnirs_representation"
        ]
    )

    multimodal_representation = (
        outputs[
            "multimodal_representation"
        ]
    )

    print(
        f"Learned EEG representation: "
        f"{tuple(eeg_representation.shape)}"
    )

    print(
        f"Learned fNIRS representation: "
        f"{tuple(fnirs_representation.shape)}"
    )

    print(
        f"Learned multimodal representation: "
        f"{tuple(multimodal_representation.shape)}"
    )

    if tuple(
        eeg_representation.shape
    ) != (8, 128):

        raise AssertionError(
            "Unexpected learned EEG "
            "representation shape."
        )

    if tuple(
        fnirs_representation.shape
    ) != (8, 128):

        raise AssertionError(
            "Unexpected learned fNIRS "
            "representation shape."
        )

    if tuple(
        multimodal_representation.shape
    ) != (8, 256):

        raise AssertionError(
            "Unexpected learned multimodal "
            "representation shape."
        )

    if not np.isfinite(
        eeg_representation.cpu().numpy()
    ).all():

        raise AssertionError(
            "Learned EEG representation "
            "contains non-finite values."
        )

    if not np.isfinite(
        fnirs_representation.cpu().numpy()
    ).all():

        raise AssertionError(
            "Learned fNIRS representation "
            "contains non-finite values."
        )

    if not np.isfinite(
        multimodal_representation.cpu().numpy()
    ).all():

        raise AssertionError(
            "Learned multimodal representation "
            "contains non-finite values."
        )

    if not np.isfinite(
        logits
    ).all():

        raise AssertionError(
            "Classifier logits contain "
            "non-finite values."
        )

    print(
        "PASS: Learned neural representations "
        "are finite and correctly shaped."
    )

    print()

    # ========================================================
    # 18. SAVE MODEL
    # ========================================================

    output_dir = (
        DATASET_ROOT
        / "phase3_outputs"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        output_dir
        / "sub-01_multimodal_tcn_classifier.pt"
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "config":
                model.config,

            "task_labels":
                TASK_LABELS,

            "test_accuracy":
                final_accuracy,

            "random_seed":
                RANDOM_SEED,
        },
        model_path,
    )

    print(
        f"Saved trained model: "
        f"{model_path}"
    )

    print()

    # ========================================================
    # FINAL
    # ========================================================

    print("=" * 72)

    print(
        "PHASE 3 REAL MULTIMODAL TCN "
        "TRAINING VALIDATION COMPLETED"
    )

    print("=" * 72)

    print()

    print(
        "Learned pipeline:"
    )

    print(
        "EEG → EEG TCN → 128-D temporal representation"
    )

    print(
        "fNIRS → fNIRS TCN → 128-D temporal representation"
    )

    print(
        "EEG + fNIRS → 256-D multimodal representation"
    )

    print(
        "256-D representation → 8-class classifier"
    )

    print(
        f"Final test accuracy: "
        f"{final_accuracy:.4f}"
    )

    print()

    print(
        "NOTE: This is a within-subject validation "
        "using sub-01."
    )


if __name__ == "__main__":
    main()
