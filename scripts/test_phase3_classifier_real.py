"""
Neuro-Symphony Phase 3 — Real TCN Classifier Validation.

Complete Phase 3 prediction pipeline:

    EEG
      ↓
    EEG TCN
      ↓
    128-D temporal representation

    fNIRS
      ↓
    fNIRS TCN
      ↓
    128-D temporal representation

              ↓

    256-D multimodal representation
              ↓
    Train/Test split
              ↓
    Multimodal classifier
              ↓
    8-class task prediction

The classifier itself is dataset-independent.
The ds004514 dataset is used only for real validation.
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

from src.models.tcn import (
    TCNConfig,
    TemporalConvolutionalNetwork,
    extract_tcn_features,
)

from src.models.multimodal import (
    MultimodalRepresentation,
    MultimodalRepresentationConfig,
)

from src.models.classifier import (
    ClassifierConfig,
    MultimodalClassifier,
    classification_loss,
    predict_classes,
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

    events = []

    for _, row in table.iterrows():

        trial_type = str(
            row["trial_type"]
        )

        if trial_type not in TASK_LABELS:
            continue

        events.append(
            EpochEvent(
                onset=float(
                    row["onset"]
                ),
                trial_type=trial_type,
                value=(
                    row["value"]
                    if "value" in table.columns
                    else None
                ),
                subject=SUBJECT,
            )
        )

    return events


# ============================================================
# EEG LOADING
# ============================================================

def load_eeg() -> EEGData:

    print("[1] Loading EEG...")

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
        recording_id=f"{SUBJECT}_task-eeg",
        recording_path=EEG_FILE,
        modality=RecordingModality.EEG,
        layout=RecordingLayout.BIDS,
        subject_id=SUBJECT,
        session_id=None,
        task_id="eeg",
        run_id=None,
        source_kind=EEGSourceKind.FILE,
        adapter_name="real_validation_mne",
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

    print("[2] Loading fNIRS...")

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
        recording_id=f"{SUBJECT}_task-nirs",
        recording_path=FNIRS_FILE,
        modality=RecordingModality.FNIRS,
        layout=RecordingLayout.BIDS,
        subject_id=SUBJECT,
        session_id=None,
        task_id="nirs",
        run_id=None,
        source_kind=FNIRSSourceKind.FILE,
        adapter_name="real_validation_mne",
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
# MAIN
# ============================================================

def main() -> None:

    print("=" * 72)

    print(
        "NEURO-SYMPHONY PHASE 3 — "
        "REAL TCN CLASSIFICATION VALIDATION"
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
    # 1. LOAD DATA
    # ========================================================

    eeg = load_eeg()

    print()

    fnirs = load_fnirs()

    print()

    # ========================================================
    # 2. LOAD EVENTS
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
            "EEG epoch count is not 720."
        )

    if len(fnirs_epochs) != 720:
        raise AssertionError(
            "fNIRS epoch count is not 720."
        )

    print(
        "PASS: 720 EEG and fNIRS epochs extracted."
    )

    print()

    # ========================================================
    # 4. CONVERT EPOCHS TO TENSORS
    # ========================================================

    print(
        "[5] Creating epoch tensors..."
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

    eeg_tensor = torch.from_numpy(
        eeg_array
    ).float()

    fnirs_tensor = torch.from_numpy(
        fnirs_array
    ).float()

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
    # 5. DEVICE
    # ========================================================

    print(
        "[6] Selecting computation device..."
    )

    if torch.backends.mps.is_available():

        device = torch.device(
            "mps"
        )

    else:

        device = torch.device(
            "cpu"
        )

    print(
        f"Device: {device}"
    )

    print()

    # ========================================================
    # 6. CREATE EEG TCN
    # ========================================================

    print(
        "[7] Creating EEG TCN..."
    )

    eeg_tcn = TemporalConvolutionalNetwork(
        TCNConfig(
            input_channels=eeg.n_channels,
            hidden_channels=32,
            embedding_dimension=128,
            kernel_size=3,
            num_blocks=4,
            dropout=0.1,
            use_batch_norm=True,
        )
    ).to(device)

    print(
        "PASS: EEG TCN created."
    )

    print()

    # ========================================================
    # 7. CREATE fNIRS TCN
    # ========================================================

    print(
        "[8] Creating fNIRS TCN..."
    )

    fnirs_tcn = TemporalConvolutionalNetwork(
        TCNConfig(
            input_channels=fnirs.n_channels,
            hidden_channels=32,
            embedding_dimension=128,
            kernel_size=3,
            num_blocks=3,
            dropout=0.1,
            use_batch_norm=True,
        )
    ).to(device)

    print(
        "PASS: fNIRS TCN created."
    )

    print()

    # ========================================================
    # 8. EXTRACT TCN REPRESENTATIONS
    # ========================================================

    print(
        "[9] Extracting EEG temporal representations..."
    )

    eeg_representation = extract_tcn_features(
        eeg_tcn,
        eeg_tensor.to(device),
    )

    print(
        f"EEG representation: "
        f"{tuple(eeg_representation.shape)}"
    )

    print(
        "[10] Extracting fNIRS temporal representations..."
    )

    fnirs_representation = extract_tcn_features(
        fnirs_tcn,
        fnirs_tensor.to(device),
    )

    print(
        f"fNIRS representation: "
        f"{tuple(fnirs_representation.shape)}"
    )

    if tuple(
        eeg_representation.shape
    ) != (720, 128):

        raise AssertionError(
            "Unexpected EEG TCN representation shape."
        )

    if tuple(
        fnirs_representation.shape
    ) != (720, 128):

        raise AssertionError(
            "Unexpected fNIRS TCN representation shape."
        )

    print(
        "PASS: TCN representations are valid."
    )

    print()

    # ========================================================
    # 9. CREATE MULTIMODAL REPRESENTATION
    # ========================================================

    print(
        "[11] Creating multimodal representation..."
    )

    multimodal_model = (
        MultimodalRepresentation(
            MultimodalRepresentationConfig(
                eeg_dimension=128,
                fnirs_dimension=128,
                output_dimension=None,
                dropout=0.0,
            )
        )
        .to(device)
    )

    multimodal_model.eval()

    with torch.no_grad():

        multimodal_representation = (
            multimodal_model(
                eeg_representation,
                fnirs_representation,
            )
        )

    print(
        f"Multimodal representation: "
        f"{tuple(multimodal_representation.shape)}"
    )

    if tuple(
        multimodal_representation.shape
    ) != (720, 256):

        raise AssertionError(
            "Expected multimodal representation "
            "shape (720, 256)."
        )

    print(
        "PASS: 256-D multimodal representation created."
    )

    print()

    # ========================================================
    # 10. CREATE LABELS
    # ========================================================

    print(
        "[12] Creating classification labels..."
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

    print(
        f"Classes: {len(TASK_LABELS)}"
    )

    print(
        f"Samples: {len(labels)}"
    )

    # ========================================================
    # 11. VALIDATE CLASS DISTRIBUTION
    # ========================================================

    print(
        "[13] Validating class distribution..."
    )

    counts = Counter(
        labels.tolist()
    )

    for index, label in enumerate(
        TASK_LABELS
    ):

        print(
            f"{index}: "
            f"{label:40s} "
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
    # 12. TRAIN / TEST SPLIT
    # ========================================================

    print(
        "[14] Creating stratified train/test split..."
    )

    indices = np.arange(
        len(labels)
    )

    train_indices, test_indices = (
        train_test_split(
            indices,
            test_size=0.20,
            random_state=42,
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
            "Unexpected training-set size."
        )

    if len(test_indices) != 144:
        raise AssertionError(
            "Unexpected test-set size."
        )

    print(
        "PASS: Stratified 80/20 split created."
    )

    print()

    # ========================================================
    # 13. PREPARE CLASSIFIER DATA
    # ========================================================

    print(
        "[15] Preparing classifier tensors..."
    )

    train_x = (
        multimodal_representation[
            train_indices
        ]
        .detach()
        .clone()
    )

    test_x = (
        multimodal_representation[
            test_indices
        ]
        .detach()
        .clone()
    )

    train_y = torch.tensor(
        labels[
            train_indices
        ],
        dtype=torch.long,
        device=device,
    )

    test_y = torch.tensor(
        labels[
            test_indices
        ],
        dtype=torch.long,
        device=device,
    )

    print(
        f"Training representation: "
        f"{tuple(train_x.shape)}"
    )

    print(
        f"Testing representation: "
        f"{tuple(test_x.shape)}"
    )

    print()

    # ========================================================
    # 14. CREATE CLASSIFIER
    # ========================================================

    print(
        "[16] Creating classifier..."
    )

    classifier = MultimodalClassifier(
        ClassifierConfig(
            input_dimension=256,
            num_classes=8,
            hidden_dimension=128,
            dropout=0.2,
        )
    ).to(device)

    print(
        "PASS: Classifier created."
    )

    print()

    # ========================================================
    # 15. TRAINING
    # ========================================================

    print(
        "[17] Training classifier..."
    )

    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    epochs = 50

    classifier.train()

    for epoch in range(
        1,
        epochs + 1,
    ):

        optimizer.zero_grad()

        logits = classifier(
            train_x
        )

        loss = classification_loss(
            logits,
            train_y,
        )

        loss.backward()

        optimizer.step()

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == epochs
        ):

            with torch.no_grad():

                predictions = torch.argmax(
                    logits,
                    dim=1,
                )

                train_accuracy = (
                    predictions
                    == train_y
                ).float().mean().item()

            print(
                f"Epoch "
                f"{epoch:02d}/{epochs} | "
                f"Loss: {loss.item():.6f} | "
                f"Train accuracy: "
                f"{train_accuracy:.4f}"
            )

    print(
        "PASS: Classifier training completed."
    )

    print()

    # ========================================================
    # 16. TESTING
    # ========================================================

    print(
        "[18] Evaluating classifier..."
    )

    classifier.eval()

    with torch.no_grad():

        test_logits = classifier(
            test_x
        )

        test_predictions = torch.argmax(
            test_logits,
            dim=1,
        )

    y_true = (
        test_y
        .detach()
        .cpu()
        .numpy()
    )

    y_pred = (
        test_predictions
        .detach()
        .cpu()
        .numpy()
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    print(
        f"Test accuracy: "
        f"{accuracy:.4f}"
    )

    print()

    # ========================================================
    # 17. CLASSIFICATION REPORT
    # ========================================================

    print(
        "[19] Classification report..."
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(
            len(TASK_LABELS)
        ),
        target_names=TASK_LABELS,
        zero_division=0,
    )

    print(
        report
    )

    # ========================================================
    # 18. CONFUSION MATRIX
    # ========================================================

    print(
        "[20] Confusion matrix..."
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
    # 19. PREDICTION INTEGRITY
    # ========================================================

    print(
        "[21] Checking prediction integrity..."
    )

    if not np.isfinite(
        accuracy
    ):

        raise AssertionError(
            "Classification accuracy is not finite."
        )

    if not np.isfinite(
        test_logits
        .detach()
        .cpu()
        .numpy()
    ).all():

        raise AssertionError(
            "Classifier logits contain "
            "non-finite values."
        )

    if len(y_pred) != len(y_true):

        raise AssertionError(
            "Prediction count does not match "
            "test-set size."
        )

    print(
        "PASS: Predictions and logits are valid."
    )

    print()

    # ========================================================
    # 20. FIRST PREDICTIONS
    # ========================================================

    print(
        "[22] First 10 predictions..."
    )

    for index in range(
        min(10, len(y_pred))
    ):

        print(
            f"{index:02d} | "
            f"True: "
            f"{TASK_LABELS[y_true[index]]:40s} | "
            f"Predicted: "
            f"{TASK_LABELS[y_pred[index]]}"
        )

    print()

    # ========================================================
    # FINAL
    # ========================================================

    print("=" * 72)

    print(
        "PHASE 3 REAL TCN CLASSIFICATION "
        "VALIDATION PASSED"
    )

    print("=" * 72)

    print()

    print(
        "Final Phase 3 representation pipeline:"
    )

    print(
        "EEG → EEG TCN → 128-D"
    )

    print(
        "fNIRS → fNIRS TCN → 128-D"
    )

    print(
        "EEG + fNIRS → 256-D multimodal representation"
    )

    print(
        "256-D representation → 8-class classifier"
    )

    print(
        f"Test accuracy: {accuracy:.4f}"
    )


if __name__ == "__main__":
    main()
