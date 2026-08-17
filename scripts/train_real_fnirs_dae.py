from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from src.acquisition.dataset_manager import DatasetManager
from src.acquisition.subject_manager import SubjectManager
from src.acquisition.recording import RecordingManager
from src.acquisition.acquisition_engine import Phase1IntegrationEngine
from src.acquisition.fnirs_loader import FNIRSLoader, FNIRSData

from src.preprocessing.fnirs_bridge import preprocess_phase1_fnirs

from src.models.dae import DAEConfig, DenoisingAutoencoder
from src.training.dae_config import DAETrainingConfig
from src.training.dae_dataset import (
    DAEDataset,
    create_windows,
    split_indices,
)
from src.training.dae_trainer import train_dae
from src.evaluation.dae_metrics import evaluate_reconstruction


SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


def main() -> None:

    print("=" * 75)
    print("PHASE 2 — REAL fNIRS DAE TRAINING")
    print("=" * 75)

    config = DAETrainingConfig()

    print()
    print("[1] DAE configuration")
    print("Window size          :", config.window_size)
    print("Batch size           :", config.batch_size)
    print("Epochs               :", config.epochs)
    print("Learning rate        :", config.learning_rate)
    print("Noise standard dev.  :", config.noise_std)
    print("Validation fraction  :", config.validation_fraction)
    print("Test fraction        :", config.test_fraction)

    # ============================================================
    # PHASE 1 — DATASET DISCOVERY
    # ============================================================

    print()
    print("[2] Discovering datasets...")

    dataset_manager = DatasetManager()

    dataset_map = dataset_manager.discover_datasets()

    datasets = tuple(
        sorted(
            dataset_map.values(),
            key=lambda item: item.dataset_name.lower(),
        )
    )

    print("Datasets discovered:", len(datasets))

    for dataset in datasets:
        print("  -", dataset.dataset_name)

    assert datasets
    print("PASS: Dataset discovery works.")

    # ============================================================
    # PHASE 1 — SUBJECT DISCOVERY
    # ============================================================

    print()
    print("[3] Discovering subjects...")

    subject_manager = SubjectManager(
        dataset_manager=dataset_manager
    )

    subject_map = subject_manager.discover_subjects()

    subjects = tuple(
        sorted(
            subject_map.values(),
            key=lambda item: (
                item.dataset_name.lower(),
                item.subject_id.lower(),
            ),
        )
    )

    print("Subjects discovered:", len(subjects))

    assert subjects
    print("PASS: Subject discovery works.")

    # ============================================================
    # PHASE 1 — RECORDING DISCOVERY
    # ============================================================

    print()
    print("[4] Discovering fNIRS recordings...")

    recording_manager = RecordingManager()

    all_recordings = []

    for subject in subjects:
        all_recordings.extend(
            recording_manager.discover_recordings(subject)
        )

    fnirs_recording_info = None

    for recording in all_recordings:
        modality = recording.modality.value.lower()

        if modality == "fnirs":
            fnirs_recording_info = recording
            break

    assert fnirs_recording_info is not None, (
        "No fNIRS recording was discovered."
    )

    print("PASS: fNIRS recording discovered.")
    print("Dataset :", fnirs_recording_info.dataset_name)
    print("Subject :", fnirs_recording_info.subject_id)
    print("Recording:", fnirs_recording_info.recording_id)

    # ============================================================
    # BUILD UNIVERSAL RECORDING
    # ============================================================

    print()
    print("[5] Building Universal Recording...")

    engine = Phase1IntegrationEngine(config=None)

    universal_recording = (
        engine._build_universal_recordings(
            [fnirs_recording_info]
        )[0][0]
    )

    # Some BIDS discovery records point to channels.tsv.
    # The actual fNIRS signal is the SNIRF file.
    recording_path = Path(
        universal_recording.recording_path
    )

    if recording_path.suffix.lower() == ".tsv":
        snirf_path = recording_path.with_name(
            recording_path.name.replace(
                "_channels.tsv",
                "_nirs.snirf",
            )
        )

        if snirf_path.exists():
            universal_recording = replace(
                universal_recording,
                recording_path=snirf_path,
            )

    print("Universal Recording:")
    print("  ID       :", universal_recording.recording_id)
    print("  Subject  :", universal_recording.subject_id)
    print("  Modality :", universal_recording.modality.value)
    print("  Path     :", universal_recording.recording_path)

    assert Path(
        universal_recording.recording_path
    ).exists()

    print("PASS: Real fNIRS signal file exists.")

    # ============================================================
    # PHASE 1 — REAL fNIRS LOADER
    # ============================================================

    print()
    print("[6] Loading through real FNIRSLoader...")

    fnirs_loader = FNIRSLoader()

    fnirs_data = fnirs_loader.load(
        universal_recording
    )

    assert isinstance(
        fnirs_data,
        FNIRSData,
    )

    print("PASS: Real Phase 1 FNIRSData received.")
    print("  Channels    :", fnirs_data.n_channels)
    print("  Samples     :", fnirs_data.n_samples)
    print("  Sampling Hz :", fnirs_data.sampling_rate)
    print("  Duration    :", fnirs_data.duration_seconds)

    # ============================================================
    # DEVELOPMENT SEGMENT
    # ============================================================

    # For the initial Phase 1 -> Phase 2 validation we used
    # 60 seconds. Real DAE training needs substantially more
    # temporal windows, so use the first 30 minutes here.
    TRAIN_DURATION_SECONDS = 30.0 * 60.0

    print()
    print("[7] Selecting real-data DAE training segment...")

    max_samples = int(
        TRAIN_DURATION_SECONDS
        * fnirs_data.sampling_rate
    )

    short_data = fnirs_data.data[
        :,
        :min(
            fnirs_data.n_samples,
            max_samples,
        ),
    ]

    short_fnirs_data = replace(
        fnirs_data,
        data=np.asarray(
            short_data,
            dtype=np.float32,
        ),
    )

    print(
        "Development duration:",
        f"{short_fnirs_data.duration_seconds:.3f}s",
    )

    print(
        "Development samples :",
        short_fnirs_data.n_samples,
    )

    # ============================================================
    # PHASE 2 — fNIRS PREPROCESSING
    # ============================================================

    print()
    print("[8] Running Phase 2 fNIRS preprocessing...")

    fnirs_preprocessing_result = (
        preprocess_phase1_fnirs(
            short_fnirs_data,
        )
    )

    processed_fnirs_data = (
        fnirs_preprocessing_result.data
    )

    quality_report = (
        fnirs_preprocessing_result.quality_report
    )

    print("PASS: Phase 2 preprocessing completed.")

    print(
        "Processed shape:",
        processed_fnirs_data.data.shape,
    )

    print(
        "Processed channels:",
        processed_fnirs_data.n_channels,
    )

    assert np.isfinite(
        processed_fnirs_data.data
    ).all()

    # ============================================================
    # TEMPORAL WINDOWS
    # ============================================================

    print()
    print("[9] Creating temporal windows...")

    # fNIRS is sampled at 7.8125 Hz, so the EEG window size
    # of 512 samples would correspond to about 65.5 seconds.
    # Use a modality-specific 256-sample fNIRS window instead.
    fnirs_window_size = 256

    print(
        "fNIRS window size:",
        fnirs_window_size,
        "samples",
    )

    print(
        "fNIRS window duration:",
        f"{fnirs_window_size / processed_fnirs_data.sampling_rate:.3f}s",
    )

    windows = create_windows(
        processed_fnirs_data.data,
        fnirs_window_size,
    )

    print("Windows shape:", windows.shape)
    print("Number of windows:", len(windows))

    # ============================================================
    # SPLIT
    # ============================================================

    print()
    print("[10] Creating train/validation/test split...")

    split = split_indices(
        len(windows),
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
        random_seed=config.random_seed,
    )

    train_windows = windows[split.train]
    validation_windows = windows[split.validation]
    test_windows = windows[split.test]

    print("Train:", len(train_windows))
    print("Validation:", len(validation_windows))
    print("Test:", len(test_windows))

    # ============================================================
    # DAE DATASETS
    # ============================================================

    print()
    print("[11] Creating fNIRS DAE datasets...")

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

    print("Train samples:", len(train_dataset))
    print("Validation samples:", len(validation_dataset))
    print("Test samples:", len(test_dataset))

    # ============================================================
    # fNIRS DAE
    # ============================================================

    print()
    print("[12] Creating fNIRS DAE...")

    model = DenoisingAutoencoder(
        DAEConfig(
            input_channels=processed_fnirs_data.n_channels,
        )
    )

    print("PASS: fNIRS DAE created.")

    print(
        "Parameters:",
        sum(
            parameter.numel()
            for parameter in model.parameters()
        ),
    )

    # ============================================================
    # TRAIN
    # ============================================================

    print()
    print("[13] Training fNIRS DAE...")

    checkpoint_path = (
        Path("checkpoints")
        / "fnirs_dae_real.pt"
    )

    result = train_dae(
        model,
        train_dataset,
        validation_dataset,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        checkpoint_path=checkpoint_path,
        random_seed=config.random_seed,
        device="cpu",
    )

    print()
    print(
        "Best validation loss:",
        result.best_validation_loss,
    )

    print(
        "Checkpoint:",
        result.checkpoint_path,
    )

    # ============================================================
    # TEST RECONSTRUCTION
    # ============================================================

    print()
    print("[14] Evaluating trained fNIRS DAE...")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        "Best checkpoint loaded successfully."
    )

    model.eval()

    noisy_samples = []
    clean_samples = []

    for index in range(
        len(test_dataset)
    ):
        noisy, clean = test_dataset[index]

        noisy_samples.append(
            noisy.numpy()
        )

        clean_samples.append(
            clean.numpy()
        )

    noisy_tensor = torch.tensor(
        np.stack(noisy_samples),
        dtype=torch.float32,
    )

    clean_tensor = torch.tensor(
        np.stack(clean_samples),
        dtype=torch.float32,
    )

    with torch.no_grad():
        reconstructed_tensor = model(
            noisy_tensor
        )

    metrics = evaluate_reconstruction(
        reconstructed_tensor.numpy(),
        clean_tensor.numpy(),
    )

    print()

    for key, value in metrics.items():
        print(
            f"{key}: {value}"
        )

    print()
    print("=" * 75)
    print("REAL fNIRS DAE TRAINING COMPLETE")
    print("=" * 75)

    print()
    print("PASS: Real Phase 1 fNIRS loaded.")
    print("PASS: Real Phase 2 preprocessing completed.")
    print("PASS: Temporal windows created.")
    print("PASS: fNIRS DAE training completed.")
    print("PASS: Best checkpoint saved.")
    print("PASS: Real fNIRS reconstruction evaluated.")

    print()
    print(
        "Checkpoint:",
        checkpoint_path.resolve(),
    )


if __name__ == "__main__":
    main()
