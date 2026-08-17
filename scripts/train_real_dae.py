"""
Neuro-Symphony Phase 2 — Real EEG DAE Training.

Uses the already validated Phase 1 -> Phase 2 EEG path.

Pipeline:

Real dataset
    -> Dataset discovery
    -> Subject discovery
    -> Recording discovery
    -> Universal Recording
    -> EEGLoader
    -> Phase 2 EEG preprocessing
    -> temporal windows
    -> noisy/clean pairs
    -> EEG DAE training
    -> evaluation
    -> checkpoint
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.acquisition.dataset_manager import DatasetManager
from src.acquisition.subject_manager import SubjectManager
from src.acquisition.recording import RecordingManager, RecordingModality
from src.acquisition.eeg_loader import EEGLoader, EEGData

from src.preprocessing.phase1_bridge import preprocess_phase1_eeg

from src.models.dae import (
    DAEConfig,
    DenoisingAutoencoder,
)

from src.training.dae_config import DAETrainingConfig
from src.training.dae_dataset import DAEDataset
from src.evaluation.dae_metrics import evaluate_reconstruction


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# DEVELOPMENT SETTINGS
# ============================================================

TEST_DURATION_SECONDS = 60.0


# ============================================================
# TEMPORAL WINDOWING
# ============================================================

def create_windows(
    signal: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """
    Convert signal:

        (channels, samples)

    into:

        (windows, channels, window_size)
    """

    if signal.ndim != 2:
        raise ValueError(
            "Expected signal shape "
            "(channels, samples)."
        )

    channels, samples = signal.shape

    number_of_windows = (
        samples // window_size
    )

    if number_of_windows < 3:
        raise ValueError(
            "Not enough samples to create "
            "train/validation/test windows."
        )

    usable_samples = (
        number_of_windows * window_size
    )

    trimmed = signal[
        :,
        :usable_samples
    ]

    windows = trimmed.reshape(
        channels,
        number_of_windows,
        window_size,
    )

    windows = np.transpose(
        windows,
        (1, 0, 2),
    )

    return np.asarray(
        windows,
        dtype=np.float32,
    )


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def split_windows(
    windows: np.ndarray,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
):
    """Create deterministic disjoint splits."""

    rng = np.random.default_rng(seed)

    indices = np.arange(
        len(windows)
    )

    rng.shuffle(indices)

    shuffled = windows[
        indices
    ]

    total = len(shuffled)

    test_count = max(
        1,
        int(total * test_fraction),
    )

    validation_count = max(
        1,
        int(total * validation_fraction),
    )

    if (
        test_count
        + validation_count
        >= total
    ):
        raise ValueError(
            "Not enough windows for "
            "train/validation/test split."
        )

    test = shuffled[
        :test_count
    ]

    validation = shuffled[
        test_count:
        test_count + validation_count
    ]

    train = shuffled[
        test_count + validation_count:
    ]

    return (
        train,
        validation,
        test,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 75)
    print("PHASE 2 — REAL EEG DAE TRAINING")
    print("=" * 75)


    # ========================================================
    # CONFIGURATION
    # ========================================================

    config = DAETrainingConfig()

    print()
    print("[1] DAE configuration")

    print(
        "Window size          :",
        config.window_size,
    )

    print(
        "Batch size           :",
        config.batch_size,
    )

    print(
        "Epochs               :",
        config.epochs,
    )

    print(
        "Learning rate        :",
        config.learning_rate,
    )

    print(
        "Noise standard dev.  :",
        config.noise_std,
    )

    print(
        "Validation fraction  :",
        config.validation_fraction,
    )

    print(
        "Test fraction        :",
        config.test_fraction,
    )


    # ========================================================
    # PHASE 1 — DATASET DISCOVERY
    # ========================================================

    print()
    print(
        "[2] Discovering datasets..."
    )

    dataset_manager = DatasetManager()

    dataset_map = (
        dataset_manager.discover_datasets()
    )

    datasets = tuple(
        sorted(
            dataset_map.values(),
            key=lambda item:
                item.dataset_name.lower(),
        )
    )

    print(
        "Datasets discovered:",
        len(datasets),
    )

    if not datasets:
        raise RuntimeError(
            "No datasets were discovered."
        )

    for dataset in datasets:
        print(
            "  -",
            dataset.dataset_name,
        )

    print(
        "PASS: Dataset discovery works."
    )


    # ========================================================
    # PHASE 1 — SUBJECT DISCOVERY
    # ========================================================

    print()
    print(
        "[3] Discovering subjects..."
    )

    subject_manager = SubjectManager(
        dataset_manager=dataset_manager,
    )

    all_subjects = []

    for dataset in datasets:

        subject_map = (
            subject_manager.discover_subjects()
        )

        all_subjects.extend(
            subject_map.values()
        )

    subjects = tuple(
        all_subjects
    )

    print(
        "Subjects discovered:",
        len(subjects),
    )

    if not subjects:
        raise RuntimeError(
            "No subjects were discovered."
        )

    print(
        "PASS: Subject discovery works."
    )


    # ========================================================
    # PHASE 1 — RECORDING DISCOVERY
    # ========================================================

    print()
    print(
        "[4] Discovering EEG recordings..."
    )

    recording_manager = RecordingManager()

    eeg_recording_info = None

    for subject in subjects:

        recordings = (
            recording_manager.discover_recordings(
                subject
            )
        )

        for recording in recordings:

            if (
                recording.modality
                is RecordingModality.EEG
            ):
                eeg_recording_info = recording
                break

        if eeg_recording_info is not None:
            break

    if eeg_recording_info is None:
        raise RuntimeError(
            "No EEG recording was discovered."
        )

    print(
        "PASS: EEG recording discovered."
    )

    print(
        "Dataset :",
        eeg_recording_info.dataset_name,
    )

    print(
        "Subject :",
        eeg_recording_info.subject_id,
    )

    print(
        "Recording:",
        eeg_recording_info.recording_id,
    )


    # ========================================================
    # BUILD UNIVERSAL RECORDING
    # ========================================================

    print()
    print(
        "[5] Building Universal Recording..."
    )

    data_files = [
        file_info
        for file_info
        in eeg_recording_info.files
        if file_info.exists
        and file_info.role.value == "DATA"
    ]

    if not data_files:

        fallback_path = Path(
            eeg_recording_info.recording_path
        )

        if not fallback_path.exists():
            raise RuntimeError(
                "No usable EEG data file found."
            )

        data_file = fallback_path

    else:

        data_file = data_files[0].path

    from src.acquisition.recording import (
        EEGRecordingMetadata,
        Recording,
    )

    universal_recording = Recording(
        recording_id=(
            f"{eeg_recording_info.recording_id}"
            "::eeg::0"
        ),
        recording_path=data_file,
        modality=RecordingModality.EEG,
        subject_id=eeg_recording_info.subject_id,
        session_id=eeg_recording_info.session_id,
        task_id=eeg_recording_info.task_id,
        run_id=eeg_recording_info.run_id,
        eeg_metadata=EEGRecordingMetadata(
            metadata={
                "recording_group":
                    eeg_recording_info.recording_id
            }
        ),
        metadata={
            "dataset_name":
                eeg_recording_info.dataset_name,
            "dataset_path":
                eeg_recording_info.dataset_path,
            "subject_path":
                eeg_recording_info.subject_path,
            "adapter_name":
                eeg_recording_info.adapter_name,
        },
    )

    print(
        "Universal Recording:"
    )

    print(
        "  ID       :",
        universal_recording.recording_id,
    )

    print(
        "  Subject  :",
        universal_recording.subject_id,
    )

    print(
        "  Modality :",
        universal_recording.modality.value,
    )

    print(
        "  Path     :",
        universal_recording.recording_path,
    )


    # ========================================================
    # PHASE 1 — EEG LOADING
    # ========================================================

    print()
    print(
        "[6] Loading through real EEGLoader..."
    )

    eeg_loader = EEGLoader()

    eeg_data = eeg_loader.load(
        universal_recording
    )

    print(
        "PASS: Real Phase 1 EEGData received."
    )

    print(
        "  Channels    :",
        eeg_data.n_channels,
    )

    print(
        "  Samples     :",
        eeg_data.n_samples,
    )

    print(
        "  Sampling Hz :",
        eeg_data.sampling_rate,
    )

    print(
        "  Duration    :",
        f"{eeg_data.duration_seconds:.2f}s",
    )

    if not isinstance(
        eeg_data,
        EEGData,
    ):
        raise RuntimeError(
            "EEGLoader did not return EEGData."
        )


    # ========================================================
    # DEVELOPMENT SEGMENT
    # ========================================================

    print()
    print(
        "[7] Selecting development segment..."
    )

    max_samples = int(
        TEST_DURATION_SECONDS
        * eeg_data.sampling_rate
    )

    if eeg_data.n_samples > max_samples:

        short_data = (
            eeg_data.data[
                :,
                :max_samples
            ]
        )

    else:

        short_data = eeg_data.data

    short_eeg_data = replace(
        eeg_data,
        data=np.asarray(
            short_data,
            dtype=np.float64,
        ),
    )

    print(
        "Development duration:",
        f"{short_eeg_data.duration_seconds:.2f}s",
    )

    print(
        "Development samples :",
        short_eeg_data.n_samples,
    )


    # ========================================================
    # PHASE 2 — EEG PREPROCESSING
    # ========================================================

    print()
    print(
        "[8] Running Phase 2 EEG preprocessing..."
    )

    processed_eeg_data, quality_report = (
        preprocess_phase1_eeg(
            short_eeg_data,
            dataset_config=None,
        )
    )

    print(
        "PASS: Phase 2 preprocessing completed."
    )

    print(
        "Processed shape:",
        processed_eeg_data.data.shape,
    )

    print(
        "Processed channels:",
        processed_eeg_data.n_channels,
    )


    # ========================================================
    # SELECT EEG CHANNELS FOR DAE
    # ========================================================

    print()
    print(
        "[9] Selecting EEG channels for DAE..."
    )

    # The real Phase-2 EEGData contains 80 channels.
    # For DAE training we use the 72 EEG/EXG channels:
    #   A1-A32 = 32
    #   B1-B32 = 32
    #   EXG1-EXG8 = 8
    # Total = 72.
    # Auxiliary/status channels are excluded from DAE input.

    eeg_dae_channels = tuple(
        name
        for name in processed_eeg_data.channel_names
        if (
            name.startswith("A")
            or name.startswith("B")
            or name.startswith("EXG")
        )
    )

    eeg_dae_indices = tuple(
        processed_eeg_data.channel_names.index(name)
        for name in eeg_dae_channels
    )

    dae_eeg_data = np.asarray(
        processed_eeg_data.data[
            list(eeg_dae_indices),
            :,
        ],
        dtype=np.float32,
    )

    print(
        "Total Phase-2 channels :",
        processed_eeg_data.n_channels,
    )
    print(
        "DAE EEG channels       :",
        len(eeg_dae_channels),
    )
    print(
        "First channels         :",
        eeg_dae_channels[:5],
    )
    print(
        "Last channels          :",
        eeg_dae_channels[-5:],
    )
    print(
        "DAE input shape        :",
        dae_eeg_data.shape,
    )

    assert len(eeg_dae_channels) == 72, (
        "Expected 72 EEG/EXG channels for DAE training."
    )

    assert dae_eeg_data.shape[0] == 72, (
        "DAE input must contain exactly 72 channels."
    )

    print(
        "PASS: 72 physiological EEG channels selected."
    )


    # ========================================================
    # TEMPORAL WINDOWS
    # ========================================================

    print()
    print(
        "[10] Creating temporal windows..."
    )

    windows = create_windows(
        dae_eeg_data,
        config.window_size,
    )

    print(
        "Windows shape:",
        windows.shape,
    )

    print(
        "Number of windows:",
        len(windows),
    )


    # ========================================================
    # SPLIT
    # ========================================================

    print()
    print(
        "[11] Creating train/validation/test split..."
    )

    (
        train_windows,
        validation_windows,
        test_windows,
    ) = split_windows(
        windows,
        config.validation_fraction,
        config.test_fraction,
        config.random_seed,
    )

    print(
        "Train:",
        len(train_windows),
    )

    print(
        "Validation:",
        len(validation_windows),
    )

    print(
        "Test:",
        len(test_windows),
    )


    # ========================================================
    # DAE DATASETS
    # ========================================================

    print()
    print(
        "[12] Creating DAE datasets..."
    )

    train_dataset = DAEDataset(
        train_windows,
        noise_std=config.noise_std,
        random_seed=config.random_seed,
    )

    validation_dataset = DAEDataset(
        validation_windows,
        noise_std=config.noise_std,
        random_seed=config.random_seed + 1,
    )

    test_dataset = DAEDataset(
        test_windows,
        noise_std=config.noise_std,
        random_seed=config.random_seed + 2,
    )

    print(
        "Train samples:",
        len(train_dataset),
    )

    print(
        "Validation samples:",
        len(validation_dataset),
    )

    print(
        "Test samples:",
        len(test_dataset),
    )


    # ========================================================
    # DATALOADERS
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )


    # ========================================================
    # MODEL
    # ========================================================

    print()
    print(
        "[13] Creating EEG DAE..."
    )

    model = DenoisingAutoencoder(
        DAEConfig(
            input_channels=
                len(eeg_dae_channels),
        )
    )

    print(
        "PASS: EEG DAE created."
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        "Parameters:",
        parameter_count,
    )


    # ========================================================
    # TRAINING SETUP
    # ========================================================

    device = torch.device(
        "cpu"
    )

    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    criterion = nn.MSELoss()


    # ========================================================
    # TRAIN
    # ========================================================

    print()
    print(
        "[14] Training EEG DAE..."
    )

    best_validation_loss = float(
        "inf"
    )

    best_state = None

    for epoch in range(
        1,
        config.epochs + 1,
    ):

        model.train()

        train_loss = 0.0
        train_count = 0

        for noisy, clean in train_loader:

            noisy = noisy.to(device)
            clean = clean.to(device)

            optimizer.zero_grad()

            reconstructed = model(
                noisy
            )

            loss = criterion(
                reconstructed,
                clean,
            )

            loss.backward()

            optimizer.step()

            batch_size = noisy.shape[0]

            train_loss += (
                loss.item()
                * batch_size
            )

            train_count += batch_size

        train_loss /= train_count


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()

        validation_loss = 0.0
        validation_count = 0

        with torch.no_grad():

            for noisy, clean in validation_loader:

                noisy = noisy.to(device)
                clean = clean.to(device)

                reconstructed = model(
                    noisy
                )

                loss = criterion(
                    reconstructed,
                    clean,
                )

                batch_size = noisy.shape[0]

                validation_loss += (
                    loss.item()
                    * batch_size
                )

                validation_count += batch_size

        validation_loss /= validation_count

        print(
            f"Epoch {epoch:03d}/{config.epochs:03d} "
            f"| train_loss={train_loss:.6f} "
            f"| validation_loss={validation_loss:.6f}"
        )

        if validation_loss < best_validation_loss:

            best_validation_loss = (
                validation_loss
            )

            best_state = {
                key:
                    value.detach()
                    .cpu()
                    .clone()

                for key, value
                in model.state_dict().items()
            }


    # ========================================================
    # RESTORE BEST MODEL
    # ========================================================

    if best_state is not None:

        model.load_state_dict(
            best_state
        )


    # ========================================================
    # SAVE CHECKPOINT
    # ========================================================

    checkpoint_dir = (
        PROJECT_ROOT
        / config.checkpoint_dir
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        checkpoint_dir
        / "eeg_dae_real.pt"
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "model_config": {
                "input_channels":
                    len(eeg_dae_channels),
                "hidden_channels": 32,
                "latent_channels": 16,
                "kernel_size": 5,
            },

            "training_config": {
                "window_size":
                    config.window_size,
                "noise_std":
                    config.noise_std,
                "epochs":
                    config.epochs,
                "learning_rate":
                    config.learning_rate,
            },

            "sampling_rate":
                processed_eeg_data.sampling_rate,

            "channel_names":
                eeg_dae_channels,

            "best_validation_loss":
                best_validation_loss,
        },
        checkpoint_path,
    )

    print()
    print(
        "Best validation loss:",
        best_validation_loss,
    )

    print(
        "Checkpoint:",
        checkpoint_path,
    )


    # ========================================================
    # TEST
    # ========================================================

    print()
    print(
        "[15] Evaluating trained EEG DAE..."
    )

    model.eval()

    reconstructed_batches = []
    clean_batches = []

    with torch.no_grad():

        for noisy, clean in test_loader:

            noisy = noisy.to(device)

            reconstructed = model(
                noisy
            )

            reconstructed_batches.append(
                reconstructed.cpu()
            )

            clean_batches.append(
                clean
            )

    reconstructed_tensor = torch.cat(
        reconstructed_batches,
        dim=0,
    )

    clean_tensor = torch.cat(
        clean_batches,
        dim=0,
    )


    # ========================================================
    # METRICS
    # ========================================================

    metrics = evaluate_reconstruction(
        reconstructed_tensor,
        clean_tensor,
    )

    print()

    for name, value in metrics.items():

        print(
            f"{name}: {value}"
        )


    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    assert np.isfinite(
        best_validation_loss
    )

    assert checkpoint_path.exists()

    assert torch.isfinite(
        reconstructed_tensor
    ).all()

    print()
    print("=" * 75)
    print("REAL EEG DAE TRAINING COMPLETE")
    print("=" * 75)

    print()
    print(
        "PASS: Real Phase 1 EEG loaded."
    )

    print(
        "PASS: Real Phase 2 preprocessing completed."
    )

    print(
        "PASS: Temporal windows created."
    )

    print(
        "PASS: DAE training completed."
    )

    print(
        "PASS: Best checkpoint saved."
    )

    print(
        "PASS: Real EEG reconstruction evaluated."
    )

    print()
    print(
        "Checkpoint:",
        checkpoint_path,
    )


if __name__ == "__main__":
    main()
